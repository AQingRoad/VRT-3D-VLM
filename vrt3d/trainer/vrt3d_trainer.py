# Copyright 2025 The HuggingFace Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import os
import cv2
import json
import copy
import torch
import PIL.Image
import numpy as np
from packaging import version
from collections import defaultdict
from datasets import Dataset, IterableDataset
from typing import Any, Callable, Optional, Union, Sized
import torch.nn.functional as F

import deepspeed
import transformers
from transformers import (
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from pycocotools import mask
from torch.utils.data import Sampler
from torchvision.ops.boxes import box_area
from torch.utils.data.dataloader import DataLoader
from accelerate.utils import is_peft_model, set_seed
from ..models import VRT3DForConditionalGeneration, VRTTextProcessor, parse_vrt_into_completion
from .vrt3d_sft_config import VRT3DSFTConfig, VRT3DDataArguments, VRT3DModelConfig
import math
if is_wandb_available():
    import wandb


class DepthGeometryBatch:
    def __init__(self, depth, intrinsics):
        self.depth = depth          # (B, H, W)
        self.intrinsics = intrinsics  # (B, 3, 3)

class RepeatRandomSampler(Sampler):
    """
    Make each micro-step (one backward) homogeneous across ranks:
    - either ALL ranks see VRT samples
    - or ALL ranks see NO-VRT samples

    Assumption:
        global_batch_size = bs_per_device * num_processes * gradient_accumulation_steps
    """

    def __init__(
        self,
        data_source: Sized,
        batch_size: int = 1,                   # global batch size
        repeat_count: int = 1,
        seed: Optional[int] = None,
        num_processes: int = 1,
        gradient_accumulation_steps: int = 1,
    ):
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.repeat_count = int(repeat_count)
        self.num_samples = len(data_source)
        self.seed = seed
        self.num_processes = int(num_processes)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)

        assert self.num_processes >= 1
        assert self.gradient_accumulation_steps >= 1
        assert self.batch_size >= 1

        # infer bs per device
        denom = self.num_processes * self.gradient_accumulation_steps
        assert self.batch_size % denom == 0, (
            f"batch_size({self.batch_size}) must be divisible by "
            f"num_processes({self.num_processes}) * gradient_accumulation_steps({self.gradient_accumulation_steps})"
        )
        self.bs_per_device = self.batch_size // denom  # per micro-step per rank
        assert self.bs_per_device >= 1

        # micro-step total samples across all ranks
        self.micro_step_size = self.bs_per_device * self.num_processes

        # RNG
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
            np.random.seed(seed)

        self.__split_data_source_into_whether_has_vrt()

    def __split_data_source_into_whether_has_vrt(self):
        self.data_indices_with_vrt = []
        self.data_indices_without_vrt = []

        # NOTE: this will call __getitem__ on dataset; if __getitem__ is stochastic,
        # the split may drift. Prefer using metadata if possible.
        for idx, data in enumerate(self.data_source):
            if "objects" in data and data["objects"] is not None and len(data["objects"]) > 0:
                self.data_indices_with_vrt.append(idx)
            else:
                self.data_indices_without_vrt.append(idx)

        if len(self.data_source) == 0:
            self.vrt_ratio = 0.0
        else:
            self.vrt_ratio = len(self.data_indices_with_vrt) / len(self.data_source)

        print("VRT on dataset ratio:", self.vrt_ratio)

    def __len__(self) -> int:
        # total indices per epoch = num_samples * repeat_count
        return self.num_samples * self.repeat_count

    def __iter__(self):
        # helper: sample k indices from a list with wrap-around
        def take_k(arr, ptr, k):
            out = []
            n = len(arr)
            for _ in range(k):
                out.append(arr[ptr % n])
                ptr += 1
            return out, ptr

        # number of global batches per repeat
        num_global_batches = math.ceil(self.num_samples / self.batch_size)

        # we schedule VRT/NO-VRT by micro-step count to match vrt_ratio
        # using floor-based dithering:
        # target_vrt_microsteps = floor((t+1)*ratio) - floor(t*ratio)
        global_microstep = 0

        for rep in range(self.repeat_count):
            # shuffle each pool
            np.random.shuffle(self.data_indices_with_vrt)
            np.random.shuffle(self.data_indices_without_vrt)

            vrt_ptr = 0
            novrt_ptr = 0

            # if one side empty, force the other side
            has_vrt = len(self.data_indices_with_vrt) > 0
            has_novrt = len(self.data_indices_without_vrt) > 0
            if not has_vrt and not has_novrt:
                return

            for _ in range(num_global_batches):
                # build one global batch (batch_size indices), but structured as:
                # gradient_accumulation_steps blocks, each block is one micro-step across ranks
                current_batch_indices = []

                for ga in range(self.gradient_accumulation_steps):
                    # decide type for THIS micro-step
                    if has_vrt and has_novrt:
                        want_vrt = (
                            math.floor((global_microstep + 1) * self.vrt_ratio)
                            - math.floor(global_microstep * self.vrt_ratio)
                        ) == 1
                    elif has_vrt:
                        want_vrt = True
                    else:
                        want_vrt = False

                    if want_vrt:
                        block, vrt_ptr = take_k(self.data_indices_with_vrt, vrt_ptr, self.micro_step_size)
                    else:
                        block, novrt_ptr = take_k(self.data_indices_without_vrt, novrt_ptr, self.micro_step_size)

                    global_microstep += 1

                    # block 内全同类，可以安全 shuffle
                    np.random.shuffle(block)

                    # distribute to each device: each rank gets bs_per_device contiguous indices
                    # keep the yield order "rank0 then rank1 ..." like your original code
                    for rank in range(self.num_processes):
                        s = rank * self.bs_per_device
                        e = (rank + 1) * self.bs_per_device
                        rank_chunk = block[s:e]
                        # optional: shuffle within rank chunk (still same type)
                        np.random.shuffle(rank_chunk)
                        current_batch_indices.extend(rank_chunk)

                # Now current_batch_indices has exactly batch_size indices
                # (unless dataset too small; wrap-around handles it)
                for idx in current_batch_indices:
                    yield idx
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from peft.tuners.lora import LoraLayer
def setup_lora(
    model,
    lora_rank = 8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=None,
    bias="none",
    use_4bit=False,
    bnb_4bit_compute_dtype=torch.bfloat16
    ):
    print("🔒 开始冻结LLM主干网络所有参数...")
    for name, param in model.named_parameters():
        # 排除可能需要训练的视觉模块（如果有），仅冻结LLM部分
        if not any(keyword in name for keyword in ['visual', 'waypointDecoder', 'memory', 'vl_decoder']):
            param.requires_grad = False
    """
    为模型配置 LoRA 适配器
    
    Args:
        model: 基础模型
        lora_rank: LoRA 秩（r），越小参数越少
        lora_alpha: LoRA 缩放系数
        lora_dropout: Dropout 概率
        target_modules: 要应用 LoRA 的目标模块名列表
        bias: 是否训练偏置（none/lora_only/all）
        use_4bit: 是否使用 4bit 量化
        bnb_4bit_compute_dtype: 4bit 量化的计算 dtype
    """
    def get_lora_target_modules(model, base_targets):
        lora_targets = []
        for name, module in model.named_modules():
            if (
                any(t in name for t in base_targets)
                and "vl_decoder" not in name
                and "visual" not in name
            ):
                lora_targets.append(name)
        return lora_targets

    if target_modules is None:
        base_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
            "gate_proj", "up_proj", "down_proj"       # FFN 层
        ]
        target_modules = get_lora_target_modules(model,base_modules)
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type="CAUSAL_LM",  # 因果语言模型
        inference_mode=False,
        use_rslora=False  # 可选启用 RSLoRA
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model

class VRT3DTrainer(Trainer):

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        script_args: VRT3DDataArguments = None,
        training_args: VRT3DSFTConfig = None,
        model_args: VRT3DModelConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        **kwargs
    ):
        # Args
        assert script_args is not None and training_args is not None and model_args is not None
        self.script_args = script_args
        args = training_args
        self.model_args = model_args

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        # Remember to modify it in the invernvl
        model_init_kwargs["attn_implementation"] = model_args.attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = 'bfloat16'
        
        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )

        model_cls = VRT3DForConditionalGeneration
        model_config, model_init_kwargs = model_cls.config_class.from_pretrained(model_id, return_unused_kwargs=True, **model_init_kwargs)
        if hasattr(model_config, "vision_config"):
            # 保留模型原生的vision_config（Qwen2_5_VLVisionConfig实例，非字典）
            original_vision_config = model_config.vision_config
        else:
            # 若模型无vision_config，初始化空配置对象（兼容场景）
            from transformers import Qwen2_5_VLVisionConfig
            original_vision_config = Qwen2_5_VLVisionConfig()
        original_vision_config.use_3d_box = args.use_3d_box
        model_config.vision_config = original_vision_config
        model_config.update({
            # vl_decoder config
            "vl_decoder": {
                "name": "ObjectCentricGeometryDecoder",
                "attn_implementation": "flash_attention_2",
                "hidden_size": 1280,
                "intermediate_size": 3420,
                "llm_hidden_state": 2048,
                "num_heads": 16,
                "spatial_merge_size": 2,
                "use_mask_loss": args.use_mask_loss,
                "use_3d_box": args.use_3d_box
            },
            "use_visual_prototype_projection": args.use_visual_prototype_projection,
            "vision_config": original_vision_config,
            "use_decoder": args.use_decoder
        })
        model = model_cls.from_pretrained(model_id, config=model_config, **model_init_kwargs)

        # Freeze vision modules
        if model_args.freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in ['visual']):
                    p.requires_grad = False
        
        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Processing class
        processing_class = AutoProcessor.from_pretrained(model_id, trust_remote_code=model_init_kwargs.get("trust_remote_code", None))
        for component, processing_keyword in [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]:
            processing_component = getattr(processing_class, component, processing_class)
            setattr(processing_component, processing_keyword, getattr(script_args, processing_keyword))
        if getattr(processing_class, "tokenizer",  None) is not None:
            pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.pad_token_id = pad_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
        else:
            assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
            pad_token_id = processing_class.pad_token_id
        processing_class = VRTTextProcessor(processing_class)
        
        # Data collator
        def data_collator(features):
            return features

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        with deepspeed.zero.GatheredParameters([model.model.embed_tokens.weight], enabled=True):
            self.model_embed_token_size = model.model.embed_tokens.weight.shape[0]
        processing_class.prepare(self.model_embed_token_size)

        set_seed(args.seed, device_specific=True)

        if args.use_lora:
            print("📌 开始配置 LoRA 微调...")
            model = setup_lora(
                model
            )


    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            if getattr(model, "language_model", None) is not None:
                # For InternVL; these operations are copied from the original training script of InternVL
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = True
                model.vision_model.encoder.gradient_checkpointing = True
                model.language_model._set_gradient_checkpointing()
                # This line is necessary, otherwise the `model.gradient_checkpointing_enable()` will be executed during the training process, leading to an error since InternVL does not support this operation.
                args.gradient_checkpointing = False
            else:
                model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

    @staticmethod
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    
    @staticmethod
    def box_xyxy_to_cxcywh(x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    def loss_3d_box(self, pred_3d, gt_3d):
        # pred_3d, gt_3d: [N, 8] = cx,cy,cz,w,l,h,sin,cos
        pred = pred_3d.float()
        gt   = gt_3d.float()
        # center/size: SmoothL1 更抗异常值
        l_center = F.smooth_l1_loss(pred_3d[:, 0:3], gt_3d[:, 0:3], beta=1.0, reduction="mean")
        l_size   = F.smooth_l1_loss(pred_3d[:, 3:6], gt_3d[:, 3:6], beta=1.0, reduction="mean")

        # yaw: 用 cos 相似度更“角度化”（避免 MSE 对幅值敏感）
        pred_yaw = pred_3d[:, 6:8]
        gt_yaw   = gt_3d[:, 6:8]
        pred_yaw = pred_yaw / (pred_yaw.norm(dim=-1, keepdim=True) + 1e-6)
        gt_yaw   = gt_yaw   / (gt_yaw.norm(dim=-1, keepdim=True) + 1e-6)
        yaw_cos  = (pred_yaw * gt_yaw).sum(dim=-1).clamp(-1.0, 1.0)
        l_yaw    = (1.0 - yaw_cos).mean()

        return (l_center + l_size) + 0.2 * l_yaw

    @staticmethod
    def box_iou(boxes1, boxes2):
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union

    def generalized_box_iou(self, boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/

        The boxes should be in [x0, y0, x1, y1] format

        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        iou, union = self.box_iou(boxes1, boxes2)

        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]

        return iou - (area - union) / area, iou

    def dice_loss(self, inputs, targets, loss_mask):
        # inputs: preds [N, H, W]
        # targets: gts [N, H, W]
        # loss_mask: whether the value is valid. [N, H, W]
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        loss_mask = loss_mask.flatten(1)
        numerator = 2 * (inputs * targets * loss_mask).sum(dim=-1)
        denominator = (inputs * loss_mask).sum(dim=-1) + (targets * loss_mask).sum(dim=-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / ((loss_mask.sum(dim=-1) > 0).sum() + 1e-5)
    
    def sigmoid_focal_loss(self, inputs, targets, loss_mask, alpha = 0.25, gamma = 2):
        # inputs: preds [N, H, W]
        # targets: gts [N, H, W]
        # loss_mask: whether the value is valid. [N, H, W]
        prob = inputs.sigmoid()
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
        return ((loss * loss_mask).sum(dim=[1, 2]) / (loss_mask.sum(dim=[1, 2]) + 1e-5)).sum() / ((loss_mask.sum(dim=[1, 2]) > 0).sum() + 1e-5)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The VRT3DTrainer does not support returning outputs")
        depth_list = []
        images = []
        K_list = []
        completion_ids = []
        completion_masks = []
        multimodal_inputs = {
            'image_grid_thw': [],
            'pixel_values': []
        }
        completion_max_len = 0

        solutions = []

        for idx, x in enumerate(inputs):
            # input_images
            assert len(x['image_path']) == 1, "current support only an image per sample"

            for img in x['image_path']:
                image = PIL.Image.open(img).convert("RGB")
                image = image.resize((960, 540), PIL.Image.Resampling.LANCZOS)
                w, h = image.size
                if w < 28 or h < 28:
                    if w < h:
                        new_w = 28
                        new_h = int(h * (28 / w))
                    else:
                        new_h = 28
                        new_w = int(w * (28 / h))
                    image = image.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                images.append(image)
                depth_file = os.path.basename(img).replace(".jpg", ".pt")
                depth_path = os.path.join(self.script_args.depth_info_dir, depth_file)
                if not os.path.exists(depth_path):
                    raise FileNotFoundError(
                        f"Depth tensor not found: {depth_path}. "
                        "Set `depth_info_dir` to the directory containing the precomputed *.pt files."
                    )
                try:
                    pack = torch.load(depth_path, map_location="cpu", weights_only=True)
                except TypeError:
                    pack = torch.load(depth_path, map_location="cpu")
                depth_list.append(pack["depth"])
                K_list.append(pack["K"])

            
            # completion
            im_w, im_h = image.size
            patch_w, patch_h = round(im_w / 28), round(im_h / 28)

            pattern = r'(<\|Obj_(\d+)\|>)'
            pattern_without_matching = r'<\|Obj_\d+\|>'

            completion_id = []
            completion_mask = []
            new_objects = []
            
            conversations = copy.deepcopy(x['conversations'])
            for conv_idx, conv in enumerate(conversations):
                # replace <|Obj_xx|> with <|VRT_xxx|>
                for cont_idx, cont in enumerate(conv['content']):
                    if cont['type'] != 'text':
                        continue
                    obj_in_cont = re.findall(pattern, cont['text'])
                    obj_strs = [i[0] for i in obj_in_cont]
                    objs = [x['objects'][int(i[1])] for i in obj_in_cont]
                    cont_parts = re.split(pattern_without_matching, cont['text'])

                    cont_with_vrt = cont_parts[0]
                    for obj_str, cont_part, obj in zip(obj_strs, cont_parts[1:], objs):
                        obj_ = obj.copy()
                        selected_patches = np.array(obj_['patches'])

                        selecting_stategy = 'random' if self.args.random_select_patch else 'border'
                        if 'selecting_stategy' in obj_ and obj_['selecting_stategy'] is not None:
                            selecting_stategy = obj_['selecting_stategy']
                            assert selecting_stategy in ['random', 'border'], 'The selecting stategy `' + obj_['selecting_stategy'] + '` is not implemented.'

                        if selecting_stategy == 'border':
                            selected_patches_x, selected_patches_y = selected_patches % patch_w, selected_patches // patch_w
                            left_patches_m = selected_patches_x == selected_patches_x.min()
                            right_patches_m = selected_patches_x == selected_patches_x.max()
                            top_patches_m = selected_patches_y == selected_patches_y.min()
                            bottom_patches_m = selected_patches_y == selected_patches_y.max()
                            centre_patches_m = (left_patches_m + right_patches_m + top_patches_m + bottom_patches_m) == 0                        
                            left_patches, right_patches, top_patches, bottom_patches, centre_patches = \
                                selected_patches[left_patches_m], \
                                selected_patches[right_patches_m], \
                                selected_patches[top_patches_m], \
                                selected_patches[bottom_patches_m], \
                                selected_patches[centre_patches_m]
                            if centre_patches_m.sum() == 0:
                                centre_patches = selected_patches
                            pick_patch = np.array([np.random.choice(centre_patches), np.random.choice(left_patches), np.random.choice(top_patches), np.random.choice(right_patches), np.random.choice(bottom_patches)])
                        else:
                            if self.args.random_select_patch_num < 0:
                                pick_patch = selected_patches.copy()
                            else:
                                if selected_patches.shape[0] < self.args.random_select_patch_num:
                                    pick_patch = np.random.choice(selected_patches, self.args.random_select_patch_num, replace=True)
                                else:
                                    pick_patch = np.random.choice(selected_patches, self.args.random_select_patch_num, replace=False)
                        
                        obj_['picked'] = pick_patch
                        obj_['use_losses'] = (conv['role'] == 'assistant')
                        new_objects.append(obj_)
                        cont_with_vrt += self.processing_class.pid2vrt(pick_patch) + cont_part
                    cont['text'] = cont_with_vrt

                if conv_idx == 0:
                    conv_str = self.processing_class.apply_chat_template([conv], tokenize=False, add_generation_prompt=False)
                    # preprocess one pass to construct the VRT into tokenizer.
                    conv_ids = self.processing_class(
                        text=conv_str,
                        images=[image],
                        return_tensors='pt',
                        padding=False,
                        add_special_tokens=False
                    )
                    conv_ids = self.processing_class(
                        text=conv_str,
                        images=[image],
                        return_tensors='pt',
                        padding=False,
                        add_special_tokens=False
                    ).to(self.accelerator.device)
                    multimodal_inputs['image_grid_thw'].append(conv_ids['image_grid_thw'])
                    multimodal_inputs['pixel_values'].append(conv_ids['pixel_values'])
                else:
                    conv_str = '<|im_start|>assistant\n%s<|im_end|>\n' % ''.join([i['text'] for i in conv['content']])
                    true_xy_traj = []
                    for single_xy in x['true_traj']:
                        true_xy_traj.append([single_xy[0], single_xy[1]])
                    conv_str = conv_str.replace("<|TRAJ_START|>, <|TRAJ_END|>", '<|TRAJ_START|> '+str(true_xy_traj)+' <|TRAJ_END|>')
                    conv_ids = self.processing_class(
                        text=conv_str,
                        return_tensors='pt',
                        padding=False,
                        add_special_tokens=False
                    ).to(self.accelerator.device)

                completion_id.append(conv_ids['input_ids'])
                conv_mask = torch.zeros_like(conv_ids['input_ids'], dtype=torch.bool)
                if conv['role'] == 'assistant':
                    conv_mask[:, 3:-1] = True
                completion_mask.append(conv_mask)
            
            completion_id = torch.concat(completion_id, dim=-1)  # [1, L]
            completion_mask = torch.concat(completion_mask, dim=-1)  # [1, L]
            solutions.append(new_objects)
            completion_ids.append(completion_id)
            completion_masks.append(completion_mask)
            completion_max_len = max(completion_max_len, int(completion_id.shape[-1]))

        new_completion_ids = completion_ids[0].new_full((len(completion_ids), completion_max_len), self.processing_class.pad_token_id, dtype=completion_ids[0].dtype)
        new_completion_masks = completion_masks[0].new_full((len(completion_masks), completion_max_len), 0, dtype=torch.int64)
        new_attention_masks = torch.zeros_like(new_completion_masks)

        for comp_idx, (completion_id, completion_mask) in enumerate(zip(completion_ids, completion_masks)):
            new_completion_ids[comp_idx:comp_idx+1, :completion_id.shape[-1]] = completion_id
            new_completion_masks[comp_idx:comp_idx+1, :completion_mask.shape[-1]] = completion_mask
            new_attention_masks[comp_idx:comp_idx+1, :completion_mask.shape[-1]] = 1
        
        new_completion_masks = new_completion_masks[:, 1:] # skip the first token, apply on the assistant response.
        multimodal_inputs['image_grid_thw'] = torch.concat(multimodal_inputs['image_grid_thw'], dim=0)
        multimodal_inputs['pixel_values'] = torch.concat(multimodal_inputs['pixel_values'], dim=0)
        if self.args.use_3d_box:
            depth_batch = torch.stack(depth_list, dim=0).to(model.device)
            K_batch = torch.stack(K_list, dim=0).to(model.device)
            depth_batch = depth_batch.squeeze(1)
            K_batch = K_batch.squeeze(1)
            depth_geometry_batch = DepthGeometryBatch(depth_batch, K_batch)
            multimodal_inputs['da3_predictions'] = depth_geometry_batch

        else:
            multimodal_inputs['da3_predictions'] = None

        # prepare for Robust Per-token Cross-Entropy Loss.
        object_use_losses = []
        loss_masks = []
        gt_bboxes = []
        gt_bboxes_3d = []
        vision_patch_nums = torch.nn.functional.pad((multimodal_inputs['image_grid_thw'].cumprod(-1)[:, -1] // (self.model.config.vision_config.spatial_merge_size ** 2)).cumsum(-1), (1, 0), 'constant', 0)
        all_vision_patch_nums = vision_patch_nums[-1]
        for sol, vpn in zip(solutions, vision_patch_nums[:-1]):
            for obj in sol:
                this_object_loss_mask = torch.zeros((obj['picked'].shape[0], all_vision_patch_nums), device=self.accelerator.device, dtype=torch.bool)
                this_object_loss_mask[:, vpn.item() + np.array(obj['patches'])] = True
                this_object_loss_mask[np.arange(obj['picked'].shape[0]), vpn.item() + obj['picked']] = False
                loss_masks.append(this_object_loss_mask)
                # obj['bbox']: x1, y1, x2, y2. Value in [0, 1].
                gt_bboxes.append(obj['bbox'])
                object_use_losses.append(obj['use_losses'])
                gt_bboxes_3d.append(obj['bbox3d_ego'])
        
        if len(loss_masks) == 0:
            loss_masks = torch.zeros(0, all_vision_patch_nums, dtype=torch.bool).to(self.accelerator.device)
        else:
            loss_masks = torch.cat(loss_masks, dim=0)
        loss_masks = torch.nn.functional.pad(loss_masks, (self.model_embed_token_size, 0), 'constant', False)
        gt_bboxes = torch.Tensor(gt_bboxes).to(self.accelerator.device).to(torch.bfloat16)
        gt_bboxes_3d = torch.Tensor(gt_bboxes_3d).to(self.accelerator.device).to(torch.bfloat16)
        if len(gt_bboxes.shape) == 1:
            gt_bboxes = gt_bboxes.unsqueeze(dim=-1).repeat_interleave(4, dim=-1)
            gt_bboxes_3d = gt_bboxes_3d.unsqueeze(dim=-1).repeat_interleave(8, dim=-1)
        object_use_losses = torch.Tensor(object_use_losses).to(self.accelerator.device).to(torch.bfloat16)

        input_ids = new_completion_ids
        attention_mask = new_attention_masks
        batch_size = input_ids.shape[0]

        model_input_ids = self.processing_class.assign_to_global_vrt_id(input_ids.clone(), multimodal_inputs['image_grid_thw'])

        # Get the current policy's log probabilities
        model_output = model(input_ids=model_input_ids, attention_mask=attention_mask, output_hidden_states=True, **multimodal_inputs)
        
        logits = model_output.logits[:, :-1] # (B, L, V)
        target_ids = model_input_ids[:, 1:]
        if self.args.use_sft_vp_mask:
            visual_patch_mask = target_ids >= self.model_embed_token_size
            logits[visual_patch_mask] = logits[visual_patch_mask].masked_fill(loss_masks, float('-inf'))
        
        # decode to bbox
        hidden_states = torch.stack(model_output.hidden_states, dim=1).permute(2, 1, 0, 3).unsqueeze(dim=-2).contiguous() # [BS, Layers, N, Dim] -> [N, Layers, BS, 1, D]
        completions, feats, labels, vps, vps_feats = parse_vrt_into_completion(self.processing_class, input_ids[:, 1:], hidden_states, torch.tensor([False] * batch_size), model_output.past_image_embeds, multimodal_inputs['image_grid_thw']) # hidden_states: [N, Layers, BS, D]

        low_res_image_embeds = model_output.past_image_embeds
        high_res_image_embeds = model_output.past_high_res_image_embeds
        visual_pe = model_output.past_visual_pe
        
        if self.args.use_decoder:
            feats = vps_feats
            decoded_list = model(feats, low_res_image_embeds, high_res_image_embeds, multimodal_inputs['image_grid_thw'], visual_pe, is_main=False)
        else:
            decoded_list = None
        del model_output

        if self.args.use_mask_loss and decoded_list['pred_mask'].shape[0] > 0:
            gt_mask = torch.zeros_like(decoded_list['pred_mask'])
            loss_mask = torch.zeros_like(decoded_list['pred_mask'])

            obj_idx = 0
            for sol in solutions:
                for obj in sol:
                    if 'rle' in obj and obj['rle'] is not None:
                        gt_m = mask.decode(obj['rle'])
                        mask_h, mask_w = decoded_list['pred_mask_valid_hw'][0][obj_idx].item(), decoded_list['pred_mask_valid_hw'][1][obj_idx].item()
                        resized_gt_m = torch.from_numpy(cv2.resize(gt_m.astype(np.float32()), (mask_w * 4, mask_h * 4)) > 0.5).to(gt_mask.dtype).to(gt_mask.device)
                        gt_mask[obj_idx, :mask_h * 4, :mask_w * 4] = resized_gt_m
                        loss_mask[obj_idx, :mask_h * 4, :mask_w * 4] = 1.0
                    obj_idx += 1

            loss_mask[object_use_losses == 0, ...] = 0.
            mask_loss = self.dice_loss(decoded_list['pred_mask'], gt_mask, loss_mask) + self.sigmoid_focal_loss(decoded_list['pred_mask'], gt_mask, loss_mask)
            self._metrics['mask_loss'].append(self.accelerator.gather_for_metrics(mask_loss).mean().item())
        else:
            if self.args.use_decoder:
                mask_loss = decoded_list['pred_mask'].sum() * 0.0
            else:
                mask_loss = 0.0

        # token loss
        logit_log_probs = logits.log_softmax(dim=-1)
        token_log_prob = torch.gather(logit_log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        per_token_loss = -token_log_prob
        sft_loss = ((per_token_loss * new_completion_masks).sum(dim=-1) / (new_completion_masks.sum(dim=-1) + 1e-4)).to(logits.dtype)
        self._metrics['sft_loss'].append(self.accelerator.gather_for_metrics(sft_loss).mean().item())
        
        if self.args.use_bbox_loss:
            # bbox loss
            pred_bboxes = decoded_list['pred_boxes']  # num_bbox, 4 [cx, cy, w, h]
            # gt_bboxes # num_bbox, 4 [x1, y1, x2, y2]
            num_bboxes = gt_bboxes.shape[0]
            giou, iou = self.generalized_box_iou(self.box_cxcywh_to_xyxy(pred_bboxes), gt_bboxes)
            giou = torch.diag(giou).to(pred_bboxes.dtype)
            if num_bboxes == 0:
                bbox_loss = pred_bboxes.sum() * 0.0
            else:
                bbox_loss = 1. - (giou * object_use_losses).sum() / (object_use_losses.sum() + 1e-4)
                bbox_loss += (torch.nn.functional.l1_loss(pred_bboxes, self.box_xyxy_to_cxcywh(gt_bboxes), reduction='none') * object_use_losses[:, None]).sum() / (object_use_losses.sum() + 1e-4)
            self._metrics['bbox_loss'].append(self.accelerator.gather_for_metrics(bbox_loss).mean().item())
            self._metrics['iou'].append(self.accelerator.gather_for_metrics(torch.diag(iou).sum() / (num_bboxes + 1e-4)).mean().item())
            self._metrics['giou'].append(self.accelerator.gather_for_metrics(giou.sum() / (num_bboxes + 1e-4)).mean().item())
        else:
            if self.args.use_decoder:
                bbox_loss = decoded_list['pred_boxes'].sum() * 0.0
            else:
                bbox_loss = 0.0

        if self.args.use_bbox_loss and self.args.use_score_loss and num_bboxes > 0:
            pred_score = decoded_list['pred_score'].sigmoid() * 2. - 1.
            score_loss = (
                torch.nn.functional.mse_loss(
                    pred_score,
                    giou.unsqueeze(1).detach(),
                    reduction='sum'
                ) / num_bboxes
            )
        else:
            # 挂图版本 0
            if self.args.use_decoder:
                score_loss = decoded_list['pred_score'].sum() * 0.0
            else:
                score_loss = 0.0

        if self.args.use_3d_box and self.args.use_decoder  and num_bboxes > 0:
            pred_boxes_3d = decoded_list['pred_box3d']
            real_gt_3d   = gt_bboxes_3d.to(pred_boxes_3d.device).to(pred_boxes_3d.dtype)
            bbox_3d_loss = self.loss_3d_box(pred_boxes_3d, real_gt_3d)
            self._metrics['bbox_3d_loss'].append(self.accelerator.gather_for_metrics(bbox_3d_loss).mean().item())
        else:
            if self.args.use_3d_box and self.args.use_decoder:
                bbox_3d_loss = decoded_list['pred_box3d'].sum() * 0.0
            else:
                bbox_3d_loss = 0.
        

        loss = sft_loss.mean() + bbox_loss + score_loss + mask_loss + bbox_3d_loss
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    
    def _get_train_sampler(self) -> Sampler:
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            batch_size=effective_batch_size,
            repeat_count=1,
            seed=self.args.seed,
            num_processes=self.accelerator.num_processes,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            seed=self.args.seed,
            num_processes=self.accelerator.num_processes,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps
        )


# Backward-compatible aliases for older training scripts.
DA3Predictions = DepthGeometryBatch
PaDTSFTTrainer = VRT3DTrainer
