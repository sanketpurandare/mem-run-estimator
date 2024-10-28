import argparse
import contextlib
import os
import time
from typing import Any, ContextManager, Dict, List, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed._tools import MemTracker, RuntimeEstimator
from torch._subclasses.fake_tensor import FakeTensorMode

from exp_utils import create_training_setup, DEVICE, gpu_types, model_names, Precision, runtime_est_modes

torch.backends.cuda.enable_flash_sdp(enabled=True)

input_configs: Dict[str, List[Dict[str, Any]]] = {
    "hf_T5": [
        {"batch_size": 4, "seq_len": 128, "precision": Precision.FP, "ac": False},
        {"batch_size": 8, "seq_len": 256, "precision": Precision.MP, "ac": False},
        {"batch_size": 16, "seq_len": 512, "precision": Precision.MP, "ac": False},
        {"batch_size": 16, "seq_len": 1024, "precision": Precision.HP, "ac": True},
        {"batch_size": 32, "seq_len": 2048, "precision": Precision.HP, "ac": True},
        {"batch_size": 32, "seq_len": 4096, "precision": Precision.HP, "ac": True}
    ],
    "hf_GPT2": [
        {"batch_size": 4, "seq_len": 128, "precision": Precision.FP, "ac": False},
        {"batch_size": 8, "seq_len": 256, "precision": Precision.MP, "ac": False},
        {"batch_size": 16, "seq_len": 512, "precision": Precision.MP, "ac": False},
        {"batch_size": 16, "seq_len": 1024, "precision": Precision.HP, "ac": True},
        {"batch_size": 32, "seq_len": 2048, "precision": Precision.HP, "ac": True},
        {"batch_size": 32, "seq_len": 4096, "precision": Precision.HP, "ac": True}
    ],
    "timm_vit": [
        {"batch_size": 4, "seq_len": None, "precision": Precision.FP, "ac": False},
        {"batch_size": 8, "seq_len": None, "precision": Precision.MP, "ac": False},
        {"batch_size": 16, "seq_len": None, "precision": Precision.MP, "ac": False},
        {"batch_size": 16, "seq_len": None, "precision": Precision.HP, "ac": True},
        {"batch_size": 32, "seq_len": None, "precision": Precision.HP, "ac": True},
    ],
    "hf_clip": [
        {"batch_size": 4, "seq_len": None, "precision": Precision.FP, "ac": False, "image_size": 224},
        {"batch_size": 8, "seq_len": None, "precision": Precision.MP, "ac": False, "image_size": 224},
        {"batch_size": 16, "seq_len": None, "precision": Precision.MP, "ac": False, "image_size": 384},
        {"batch_size": 16, "seq_len": None, "precision": Precision.HP, "ac": True, "image_size": 384},
        {"batch_size": 32, "seq_len": None, "precision": Precision.HP, "ac": True, "image_size": 512},
    ],
    "llama_v3_1b": [
        {"batch_size": 4, "seq_len": 128, "precision": Precision.FP, "ac": False},
        {"batch_size": 8, "seq_len": 256, "precision": Precision.MP, "ac": False},
        {"batch_size": 16, "seq_len": 512, "precision": Precision.MP, "ac": False},
        {"batch_size": 16, "seq_len": 1024, "precision": Precision.HP, "ac": True},
        {"batch_size": 32, "seq_len": 2048, "precision": Precision.HP, "ac": True},
        {"batch_size": 32, "seq_len": 4096, "precision": Precision.HP, "ac": True}
    ],
    "gemma_2b": [
        {"batch_size": 4, "seq_len": 128, "precision": Precision.FP, "ac": False},
        {"batch_size": 8, "seq_len": 256, "precision": Precision.MP, "ac": False},
        {"batch_size": 16, "seq_len": 512, "precision": Precision.MP, "ac": False},
        {"batch_size": 16, "seq_len": 1024, "precision": Precision.HP, "ac": True},
        {"batch_size": 32, "seq_len": 2048, "precision": Precision.HP, "ac": True},
        {"batch_size": 32, "seq_len": 4096, "precision": Precision.HP, "ac": True}
    ],
    "timm_convnext_v2": [
        {"batch_size": 4, "seq_len": None, "precision": Precision.FP, "ac": False, "image_size": 224},
        {"batch_size": 8, "seq_len": None, "precision": Precision.MP, "ac": False, "image_size": 224},
        {"batch_size": 16, "seq_len": None, "precision": Precision.MP, "ac": False, "image_size": 384},
        {"batch_size": 16, "seq_len": None, "precision": Precision.HP, "ac": True, "image_size": 384},
        {"batch_size": 32, "seq_len": None, "precision": Precision.HP, "ac": True, "image_size": 512},
    ],
}


class Experiment:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 2,
        seq_len: int = 128,
        precision: Precision = Precision.HP,
        ac: bool = False,
        image_size: int = 224,
        dev: torch.device = torch.device(DEVICE), 
        init_mode: ContextManager = contextlib.nullcontext(),
    ):  
        self.execution_ctx = init_mode
        self.device = dev
        self.model, self.optimizer, self.train_step = create_training_setup(
            model_name,
            batch_size,
            seq_len,
            precision,
            ac,
            image_size,
            dev,
            init_mode
        )
        self.model.train()
        param_dtypes = set()
        param_count = 0
        for param in self.model.parameters():
            param_count += param.numel()
            param_dtypes.add(param.dtype)

        print(f"Model has {param_count} parameters.")
        print(f"Model has {param_dtypes} dtypes.")
        print(f"Parameter Memory: {torch.cuda.memory_allocated() / 2**30} GiB")

    def real_execution(self) -> Tuple[float, int, int]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        warm_up_iters, benchmark_iters = 2, 3
        total_iters = warm_up_iters + benchmark_iters
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
        for i in range(5):
            start_events[i].record()
            with self.execution_ctx:
                self.train_step(self.model, self.optimizer)
            end_events[i].record()
        torch.cuda.synchronize()
        iter_time = (
            sum(start_events[i].elapsed_time(end_events[i]) for i in range(warm_up_iters, total_iters)) / benchmark_iters
        )
        mem_stats = torch.cuda.memory_stats()
        peak_active = mem_stats["active_bytes.all.peak"]
        peak_reserved = mem_stats["reserved_bytes.all.peak"]
        print(f"Iter time: {iter_time} ms")
        print(f"Peak Active Memory: {peak_active / 2**30} GiB")
        print(f"Peak Reserved Memory: {peak_reserved / 2**30} GiB")

        return iter_time, peak_active, peak_reserved
    
    def memory_estimation(self) -> Tuple[int, float]:
        iters = 2
        mem_tracker = MemTracker()
        mem_tracker.track_external(self.model, self.optimizer)

        for iter in range(iters):
            track_start_time = time.time()
            with self.execution_ctx:
                with mem_tracker:
                    self.train_step(self.model, self.optimizer)
            track_end_time = time.time()
            if iter == 0:
                mem_tracker.reset_mod_stats()
        peak_tracker = mem_tracker.get_tracker_snapshot("peak")[self.device]["Total"]
        mem_tracker.display_snapshot("peak", units="MiB", tabulate=True)
        tracking_time = (track_end_time - track_start_time) * 1e3
        print(f"Memory Tracking time (ms): {tracking_time}")
        return (peak_tracker, tracking_time)
    
    def runtime_estimation(self, estimate_mode: str) -> Tuple[float, float]:
        runtime_estimator = RuntimeEstimator()
        est_start_time = time.time()
        with self.execution_ctx:
            with runtime_estimator(estimate_mode_type=estimate_mode):
                self.train_step(self.model, self.optimizer)
        torch.cuda.synchronize()
        est_end_time = time.time()
        estimation_time = (est_end_time - est_start_time) * 1e3
        run_est = runtime_estimator.total_runtime
        return (run_est, estimation_time)





    def run(self):
        self.train_step(self.model, self.optimizer)
        print("Successful.")




if __name__ == "__main__":
    # with FakeTensorMode():
    with contextlib.nullcontext():
        with torch.device("cuda"):
            exp = Experiment(model_names[3], model_batch_sizes[model_names[3]])
        exp.init_optimizer_states()
        exp.run()
    # print(f"Memory: {torch.cuda.memory_allocated() / 2**30} GiB")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemma_2b",
        choices=model_names,
        help=f"Model name",
    )    
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="Training batch size"
    )
    parser.add_argument(
        "--seq_len",
        default=20,
        type=int,
        help="Training equence length"
    )
    parser.add_argument(
        "--image_size",
        default=224,
        default=int,
        help="Training image size"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=Precision.HP.value,
        choices=[p.value for p in Precision],
        help=f"Training precision"
    )
    parser.add_argument(
        "--enable_ac", 
        action="store_true", 
        help="Enables activation checkpointing"
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="H100",
        choices=gpu_types,
        help="GPU type to use",
    )
    parser.add_argument(
        "--real_execution", 
        action="store_true", 
        help="Execute a training iteration"
    )
    parser.add_argument(
        "--memory_estimation", 
        action="store_true", 
        help="Estimate training memory"
    )
    parser.add_argument(
        "--runtime_estimation", 
        action="store_true", 
        help="Estimate training runtime"
    )
    parser.add_argument(
        "--runtime_estimation_mode",
        type=str,
        default="operator-level-cost-model",
        choices=runtime_est_modes,
        help="Runtime estimation modes",
    )