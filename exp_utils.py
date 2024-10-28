from contextlib import nullcontext
from enum import StrEnum
import os
from typing import Any, Callable, Dict, Iterator, List, Set, ContextManager, Tuple, Type

import timm
import timm.optim
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, CLIPModel, GemmaForCausalLM, LlamaForCausalLM
from torchmultimodal.modules.losses.contrastive_loss_with_temperature import (
    ContrastiveLossWithTemperature,
)
import torch
from torch import nn, optim
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.distributed._composable import checkpoint

DEVICE = "cuda"
BASE_DIR = "/n/holyscratch01/idreos_lab/Users/spurandare/mem-run-estimator"
OUT_DIR = f"{BASE_DIR}/outputs"
gpu_types: Set[str] = {"H100", "A100"}

runtime_est_modes: Set[str] = {"operator-level-cost-model", "operator-level-benchmark", "operator-level-learned-model"}

model_names: Set[str] = {
    "hf_T5",
    "hf_GPT2",
    "timm_vit",
    "hf_clip",
    "llama_v3_1b",
    "gemma_2b",
    "timm_convnext_v2"
}

class Precision(StrEnum):
    FP = "FP"
    MP = "MP"
    HP = "HP"

model_cards: Dict[str, str] = {
    "hf_T5": "t5-large",
    "hf_GPT2": "gpt2-large",
    "llama_v3_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "gemma_2b": "google/gemma-2b",
    "timm_convnext_v2": "convnextv2_base.fcmae_ft_in22k_in1k",
    "timm_vit": "vit_large_patch14_clip_224.laion2b_ft_in12k_in1k",
    "hf_clip": "openai/clip-vit-base-patch32",
}

precision_to_dtype: Dict[Precision, torch.dtype] = {
    Precision.FP : torch.float32,
    Precision.HP: torch.float16,
    Precision.MP: torch.float32
}

model_class: Dict[str, Type] = {
    "hf_T5": AutoModelForSeq2SeqLM,
    "hf_GPT2": AutoModelForCausalLM,
    "llama_v3_1b": LlamaForCausalLM,
    "gemma_2b": GemmaForCausalLM,
    "hf_clip": CLIPModel,
}

model_ac_class: Dict[str, str] = {
    "hf_T5": "T5Block",
    "hf_GPT2": "GPT2Block",
    "llama_v3_1b": "LlamaDecoderLayer",
    "gemma_2b": "GemmaDecoderLayer",
    "timm_convnext_v2": "ConvNeXtBlock",
    "timm_vit": "Block",
    "hf_clip": "CLIPEncoderLayer",
}

def generate_inputs_and_labels(
        bsz: int, vocab_size: int, seq_len: int, dev: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, vocab_size, (bsz, seq_len), dtype=torch.int64, device=dev)
    labels = torch.randint(0, vocab_size, (bsz, seq_len), dtype=torch.int64, device=dev)
    return (input_ids, labels)

def generate_inputs_and_targets(
        bsz: int, im_sz:int, n_classes: int, dtype: torch.dtype, dev: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    input = torch.randn((bsz, 3, im_sz, im_sz), dtype=dtype, device=dev)
    target = torch.randint(0, n_classes, (bsz, ), dtype=torch.int64, device=dev)
    return(input, target)

def generate_multimodal_inputs(
        bsz: int, vocab_size: int, seq_len: int, im_sz:int, dtype: torch.dtype, dev: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_img = torch.randn((bsz, 3, im_sz, im_sz), dtype=dtype, device=dev)
    input_ids = torch.randint(0, vocab_size, (bsz, seq_len), dtype=torch.int64, device=dev)
    attention_mask = torch.ones((bsz, seq_len), dtype=torch.int64, device=dev)
    return (input_img, input_ids, attention_mask)
     

def create_optimizer(param_iter: Iterator) -> optim.Optimizer:
    optimizer = optim.AdamW(
        param_iter,
        lr=1e-4,
        weight_decay=1.0e-4,
        eps=1.0e-6,
    )
    return optimizer

def apply_ac(model: nn.Module, ac_class: str):
    for module in model.modules():
        module_class = module.__class__.__name__
        if module_class == ac_class:
            checkpoint(module, preserve_rng_state=False)

def create_training_setup(
        model_name: str,
        batch_size: int = 2,
        seq_len: int = 128,
        precision: Precision = Precision.HP,
        ac: bool = False,
        image_size: int = 224,
        dev: torch.device = torch.device(DEVICE), 
        init_mode: ContextManager = nullcontext(),
    ) -> Tuple[nn.Module, optim.Optimizer, Callable]:
    dtype = precision_to_dtype[precision]
    amp_context = nullcontext()
    if precision == Precision.MP:
        amp_context = torch.autocast(device_type=DEVICE)
    if model_name in [
        "hf_T5", "hf_GPT2", "llama_v3_1b", "gemma_2b"
    ]:  
        
        model_card = model_cards[model_name]
        model_cls = model_class[model_name]
        config = AutoConfig.from_pretrained(model_card)
        print(config.vocab_size)

        with init_mode:
            with torch.device(dev):
                model = model_cls.from_config(config=config).to(dtype=dtype)
            optimizer = create_optimizer(model.paramaters())
            if ac:
                ac_class = model_ac_class[model_name]
                apply_ac(model, ac_class)

        def hf_train_step(
                model: nn.Module, optim: optim.Optimizer,
            ):
                input_ids, labels = generate_inputs_and_labels(batch_size, model.vocab_size, seq_len, dev)
                inputs = {"input_ids": input_ids, "labels": labels}
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    with amp_context:
                        loss = model(**inputs).loss
                    loss.backward()
                    optim.step()
                    optim.zero_grad()

        return (model, optimizer, hf_train_step)

    elif model_name in ["timm_vit", "timm_convnext_v2"]:
        model_card = model_cards[model_name]
        with init_mode:
            with torch.device(dev):
                model = timm.create_model(model_card, pretrained=False).to(dtype=dtype)
            optimizer = timm.optim.create_optimizer_v2(model, opt="adam")     
            loss_fn = nn.functional.cross_entropy
            if ac:
                ac_class = model_ac_class[model_name]
                apply_ac(model, ac_class)

        def timm_train_step(
                model: nn.Module, optim: optim.Optimizer,
            ):
                n_classes = model.default_cfg['num_classes']
                inputs = generate_inputs_and_targets(batch_size, image_size, n_classes, dtype, dev)
                inp, target = inputs
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    with amp_context:
                        output = model(inp)
                        loss = loss_fn(output, target)
                    loss.backward()
                    optim.step()
                    optim.zero_grad()

        return (model, optimizer, timm_train_step)
    
    elif model_name == "hf_clip":
        model_card = model_cards[model_name]
        model_cls = model_class[model_name]
        config = AutoConfig.from_pretrained(model_card)
        with init_mode:
            with torch.device(dev):
                model = model_cls._from_config(config=config).to(dtype=dtype)
                loss_fn = ContrastiveLossWithTemperature()

            class CLIP(nn.Module):
                def __init__(self, clip_model, loss_mod):
                    super.__init__()
                    self.add_module('clip_model', clip_model)
                    self.add_module('contrastive_loss_with_temp', loss_mod)

                def forward(self, **kwargs):
                    outputs = self.clip_model(**kwargs)
                    loss = self.contrastive_loss_with_temp(outputs.image_embeds, outputs.text_embeds)
                    return loss

            model_with_loss = CLIP(model, loss_fn)
            if ac:
                ac_class = model_ac_class[model_name]
                apply_ac(model_with_loss, ac_class)
            optimizer = create_optimizer(model_with_loss.parameters())
        
        def clip_train_step(
            model: nn.Module, optim: optim.Optimizer,
        ):
                img, ids, attn_mask = generate_multimodal_inputs(
                    batch_size,
                    model.config.text_config.vocab_size,
                    model.config.text_config.max_length,
                    image_size,
                    dtype,
                    dev
                )
                inputs = {'input_ids': ids, 'attention_mask': attn_mask, 'pixel_values': img}
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    with amp_context:
                        loss = model(**inputs)
                    loss.backward()
                    optim.step()
                    optim.zero_grad()

        return (model_with_loss, optimizer, clip_train_step)

    else:
         raise ValueError(f"No setup is available for {model_name}. Please choose from {model_names}")

def write_to_logfile(file_name: str, log_record: str):
    # Create a lock file
    lock_file = file_name + ".lock"
    if os.path.exists(lock_file):
        # If the lock file exists, wait for it to be released
        while os.path.exists(lock_file):
            pass
    else:
        # Create the lock file and write to the file
        with open(lock_file, "w") as f:
            f.write("locked")
        with open(file_name, "a") as f:
            f.write(log_record)
            f.flush()
        # Release the lock file
        os.remove(lock_file)