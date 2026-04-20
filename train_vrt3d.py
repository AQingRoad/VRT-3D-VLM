import os
import re
import PIL
import json
import math
import torch
import pathlib
from typing import Tuple
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset

from transformers.utils import logging
from transformers import AutoProcessor, AutoTokenizer, HfArgumentParser, TrainerCallback

from vrt3d.trainer import VRT3DSFTConfig, VRT3DDataArguments, VRT3DModelConfig
from vrt3d.trainer import VRT3DTrainer
from vrt3d.utils.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_torch_load
monkey_patch_qwen2_5vl_flash_attn()
monkey_patch_torch_load()
import shutil


class KeepLatestGlobalStepsDirCallback(TrainerCallback):
    """
    需求：
    - 每个 checkpoint-* 里都会出现 global_steps/ 或 global_step*/ 目录
    - 只保留“最新一次保存的那个 checkpoint”里的 global_step(s) 目录
    - 其它历史 checkpoint-* 里的 global_step(s) 目录全部删除
    - （可选）也清理 output_dir 根目录下的 global_step(s)
    """

    def __init__(self, output_dir: str, clean_root: bool = True, verbose: bool = True):
        self.output_dir = output_dir
        self.clean_root = clean_root
        self.verbose = verbose
        self._ckpt_re = re.compile(r"^checkpoint-(\d+)$")
        self._gs_re = re.compile(r"^global_steps$|^global_step\d*$")

    def _list_ckpts(self):
        if not os.path.isdir(self.output_dir):
            return []
        ckpts = []
        for d in os.listdir(self.output_dir):
            m = self._ckpt_re.match(d)
            if m:
                ckpts.append((int(m.group(1)), os.path.join(self.output_dir, d)))
        ckpts.sort(key=lambda x: x[0])
        return ckpts

    def _rm_global_steps_in_dir(self, parent: str):
        if not os.path.isdir(parent):
            return 0
        removed = 0
        for name in os.listdir(parent):
            if self._gs_re.match(name):
                p = os.path.join(parent, name)
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                    removed += 1
                    if self.verbose:
                        print(f"🗑️ removed: {p}")
        return removed

    def on_save(self, args, state, control, **kwargs):
        ckpts = self._list_ckpts()
        if not ckpts:
            return

        latest_step, latest_dir = ckpts[-1]
        removed = 0

        if self.clean_root:
            removed += self._rm_global_steps_in_dir(self.output_dir)

        for step, ckpt_dir in ckpts[:-1]:
            removed += self._rm_global_steps_in_dir(ckpt_dir)

        if self.verbose:
            print(f"✅ keep global_step(s) only in latest checkpoint-{latest_step}; removed {removed} dirs total")


def main(script_args, training_args, model_args):
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    assert len(data_files) == len(image_folders), "Number of data files must match number of image folders"

    all_data = []
    for data_file, image_folder in zip(data_files, image_folders):
        if os.path.exists(data_file) is False:
            data = load_dataset("/".join(data_file.split("/")[:-1]), data_files=data_file.split("/")[-1])['train'].to_list()
        else:
            data = [json.loads(i) for i in open(data_file, 'r').readlines()]

        for item in data:
            item['id'] = str(item['id'])
            if 'image' in item:
                if isinstance(item['image'], str):
                    item['image_path'] = [os.path.join(image_folder, item['image'])]
                    del item['image']
                elif isinstance(item['image'], list):
                    item['image_path'] = [
                        os.path.join(os.path.join(image_folder, 'CAM_FRONT_LEFT'),item['image'][0]),
                        os.path.join(os.path.join(image_folder, 'CAM_FRONT'),item['image'][1]),
                        os.path.join(os.path.join(image_folder, 'CAM_FRONT_RIGHT'),item['image'][2])
                    ]
                    del item['image']
                else:
                    raise ValueError(f"Unsupported image type: {type(item['image'])}")
            
            if 'camera_types' in item:
                item['camera_types'] = item['camera_types']
            
            new_conversations = []
            for idx, conv in enumerate(item['conversations']):
                if idx == 0 and conv['from'] not in ['human', 'user']:
                    continue
                new_conversations.append({
                    'role': 'user' if conv['from'] in ['human', 'user'] else 'assistant',
                    'content': [
                        *({'type': 'image', 'text': None} for _ in range(len(item['image_path'])) if len(new_conversations) == 0),
                        {'type': 'text', 'text': conv['value'].replace('<image>', '')}
                    ]
                })

            if not new_conversations:
                print(f"Skip sample {item.get('id', 'unknown')}: no valid conversations found.")
                continue

            if item.get('answer_template') is not None and new_conversations[-1]['role'] != 'assistant':
                new_conversations.append({
                    'role': 'assistant',
                    'content': [
                        {'type': 'text', 'text': item['answer_template']}
                    ]
                })
            item['conversations'] = new_conversations
            item.pop('answer_template', None)
            all_data.append(item)
            
    dataset = Dataset.from_list(all_data)
    
    def make_conversation(example):
        assert all(os.path.exists(p) for p in example['image_path']), f"Image paths do not exist: {example['image_path']}"
        return example

    dataset = dataset.map(make_conversation, num_proc=16)
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']

    trainer_cls = VRT3DTrainer
    print("using trainer:", trainer_cls.__name__)
    training_args.save_only_model = True
    training_args.save_optimizer_and_scheduler = False
    eval_strategy = getattr(training_args, "eval_strategy", None)
    if eval_strategy is None:
        eval_strategy = getattr(training_args, "evaluation_strategy", "no")
    eval_strategy = str(eval_strategy).split(".")[-1]
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        script_args=script_args,
        training_args=training_args,
        model_args=model_args,
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if eval_strategy != "no" else None,
        callbacks=[KeepLatestGlobalStepsDirCallback(output_dir=training_args.output_dir)]
    )
    model = trainer.model
    print("正在打印可训练参数")
    for n,p in model.named_parameters():
        if n.startswith("visual.") or n.startswith("vl_decoder."):
            p.requires_grad = True
    local_checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    resume_arg = None
    if isinstance(training_args.resume_from_checkpoint, str):
        normalized_resume = training_args.resume_from_checkpoint.strip()
        if normalized_resume.lower() == "true" and local_checkpoints:
            resume_arg = True
        elif normalized_resume.lower() not in {"", "false", "none"}:
            resume_arg = normalized_resume
    elif training_args.resume_from_checkpoint is True and local_checkpoints:
        resume_arg = True

    if resume_arg is not None:
        trainer.train(resume_from_checkpoint=resume_arg)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((VRT3DDataArguments, VRT3DSFTConfig, VRT3DModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    main(script_args, training_args, model_args)
