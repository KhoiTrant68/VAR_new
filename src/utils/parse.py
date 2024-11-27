import json
import os
import random
import re
import subprocess
import sys
import time
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import torch

try:
    from tap import Tap
except ImportError as e:
    print(
        f"Error: `from tap import Tap` failed. Please run: pip3 install typed-argument-parser",
        file=sys.stderr,
        flush=True,
    )
    time.sleep(5)
    raise e

import dist

from utils import misc


class Args(Tap):
    data_path: str = "/path/to/imagenet"
    exp_name: str = "text"

    # VAE
    vae_compile_mode: int = (
        0  # torch.compile VAE; =0: not compile; 1: compile with 'reduce-overhead'; 2: compile with 'max-autotune'
    )
    # VAR
    var_compile_mode: int = (
        0  # torch.compile VAR; =0: not compile; 1: compile with 'reduce-overhead'; 2: compile with 'max-autotune'
    )
    depth: int = 16  # VAR depth
    # VAR initialization
    auto_init: float = -1  # -1: automated model parameter initialization
    head_weight_multiplier: float = 0.02  # head.w *= head_weight_multiplier
    ada_lin_init_multiplier: float = 0.5  # the multiplier of ada_lin.w's initialization
    ada_lin_gamma_init_multiplier: float = (
        1e-5  # the multiplier of ada_lin.w[gamma channels]'s initialization
    )
    # VAR optimization
    use_fp16: int = 0  # 1: using fp16, 2: bf16
    base_lr: float = 1e-4  # base lr
    scaled_lr: float = None  # lr = base lr * (global_batch_size / 256)
    initial_wd: float = 0.05  # initial wd
    final_wd: float = 0  # final wd, =final_wd or initial_wd
    grad_clip: float = 2.0  # <=0 for not using grad clip
    label_smoothing: float = 0.0  # label smooth

    global_batch_size: int = 768  # global batch size
    batch_size: int = (
        0  # [automatically set; don't specify this] batch size per GPU = round(args.global_batch_size / args.grad_accumulation / dist.get_world_size() / 8) * 8
    )
    glb_batch_size: int = (
        0  # [automatically set; don't specify this] global batch size = args.batch_size * dist.get_world_size()
    )
    grad_accumulation: int = 1  # gradient accumulation

    epochs: int = 250
    warmup_epochs: float = 0
    initial_lr_ratio: float = 0.005  # initial lr ratio at the beginning of lr warm up
    final_lr_ratio: float = 0.01  # final lr ratio at the end of training
    lr_schedule: str = "lin0"  # lr schedule

    optimizer: str = (
        "adamw"  # lion: https://cloud.tencent.com/developer/article/2336657?areaId=106001 lr=5e-5 (0.25x) wd=0.8 (8x); Lion needs a large bs to work
    )
    use_fused_adamw: bool = True  # fused adamw

    # other hps
    use_shared_adaln: bool = False  # whether to use shared adaln
    use_l2_norm_attention: bool = True  # whether to use L2 normalized attention
    use_fused_ops: bool = (
        True  # whether to use fused op like flash attn, xformers, fused MLP, fused LayerNorm, etc.
    )

    # data
    patch_numbers_str: str = "1_2_3_4_5_6_8_10_13_16"
    patch_size: int = 16
    patch_numbers: tuple = (
        None  # [automatically set; don't specify this] = tuple(map(int, args.patch_numbers_str.replace('-', '_').split('_')))
    )
    resolutions: tuple = (
        None  # [automatically set; don't specify this] = tuple(pn * args.patch_size for pn in args.patch_numbers)
    )

    data_load_resolution: int = (
        None  # [automatically set; don't specify this] would be max(patch_numbers) * patch_size
    )
    mid_resolution: float = (
        1.125  # aug: first resize to mid_resolution = 1.125 * data_load_resolution, then crop to data_load_resolution
    )
    horizontal_flip: bool = False  # augmentation: horizontal flip
    workers: int = (
        0  # num workers; 0: auto, -1: don't use multiprocessing in DataLoader
    )

    # progressive training
    progressive_training_ratio: float = (
        0.0  # >0 for use progressive training during [0%, this] of training
    )
    progressive_initial_stage: int = (
        4  # progressive initial stage, 0: from the 1st token map, 1: from the 2nd token map, etc
    )
    progressive_warmup_epochs: float = (
        0  # num of warmup epochs at each progressive stage
    )

    # would be automatically set in runtime
    command: str = " ".join(sys.argv[1:])  # [automatically set; don't specify this]
    branch: str = (
        subprocess.check_output(
            "git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD",
            shell=True,
        )
        .decode("utf-8")
        .strip()
        or "[unknown]"
    )  # [automatically set; don't specify this]
    commit_id: str = (
        subprocess.check_output("git rev-parse HEAD", shell=True)
        .decode("utf-8")
        .strip()
        or "[unknown]"
    )  # [automatically set; don't specify this]
    commit_msg: str = (
        subprocess.check_output("git log -1", shell=True)
        .decode("utf-8")
        .strip()
        .splitlines()
        or ["[unknown]"]
    )[
        -1
    ].strip()  # [automatically set; don't specify this]
    acc_mean: float = None  # [automatically set; don't specify this]
    acc_tail: float = None  # [automatically set; don't specify this]
    L_mean: float = None  # [automatically set; don't specify this]
    L_tail: float = None  # [automatically set; don't specify this]
    vacc_mean: float = None  # [automatically set; don't specify this]
    vacc_tail: float = None  # [automatically set; don't specify this]
    vL_mean: float = None  # [automatically set; don't specify this]
    vL_tail: float = None  # [automatically set; don't specify this]
    grad_norm: float = None  # [automatically set; don't specify this]
    cur_lr: float = None  # [automatically set; don't specify this]
    cur_wd: float = None  # [automatically set; don't specify this]
    cur_it: str = ""  # [automatically set; don't specify this]
    cur_ep: str = ""  # [automatically set; don't specify this]
    remain_time: str = ""  # [automatically set; don't specify this]
    finish_time: str = ""  # [automatically set; don't specify this]

    # environment
    local_output_dir_path: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "local_output"
    )  # [automatically set; don't specify this]
    tensorboard_log_dir_path: str = (
        "...tb-..."  # [automatically set; don't specify this]
    )
    log_file_path: str = "..."  # [automatically set; don't specify this]
    last_checkpoint_path: str = "..."  # [automatically set; don't specify this]

    use_tf32: bool = True  # whether to use TensorFloat32
    device: str = "cpu"  # [automatically set; don't specify this]
    seed: int = None  # seed

    def seed_everything(self, benchmark: bool):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = benchmark
        if self.seed is None:
            torch.backends.cudnn.deterministic = False
        else:
            torch.backends.cudnn.deterministic = True
            seed = self.seed * dist.get_world_size() + dist.get_rank()
            os.environ["PYTHONHASHSEED"] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

    same_seed_for_all_ranks: int = 0  # this is only for distributed sampler

    def get_different_generator_for_each_rank(
        self,
    ) -> Optional[torch.Generator]:  # for random augmentation
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.seed * dist.get_world_size() + dist.get_rank())
        return g

    local_debug: bool = "KEVIN_LOCAL" in os.environ
    debug_nan: bool = False  # 'KEVIN_LOCAL' in os.environ

    def compile_model(self, m, fast):
        if fast == 0 or self.local_debug:
            return m
        return (
            torch.compile(
                m,
                mode={
                    1: "reduce-overhead",
                    2: "max-autotune",
                    3: "default",
                }[fast],
            )
            if hasattr(torch, "compile")
            else m
        )

    def state_dict(self, key_ordered=True) -> Union[OrderedDict, dict]:
        d = (OrderedDict if key_ordered else dict)()
        for k in self.class_variables.keys():
            if k not in {"device"}:  # these are not serializable
                d[k] = getattr(self, k)
        return d

    def load_state_dict(self, d: Union[OrderedDict, dict, str]):
        if isinstance(d, str):  # for compatibility with old version
            d: dict = eval(
                "\n".join(
                    [
                        l
                        for l in d.splitlines()
                        if "<bound" not in l and "device(" not in l
                    ]
                )
            )
        for k in d.keys():
            try:
                setattr(self, k, d[k])
            except Exception as e:
                print(f"k={k}, v={d[k]}")
                raise e

    @staticmethod
    def set_tf32(use_tf32: bool):
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = bool(use_tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
                print(
                    f"[tf32] [precis] torch.get_float32_matmul_precision(): {torch.get_float32_matmul_precision()}"
                )
            print(
                f"[tf32] [ conv ] torch.backends.cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}"
            )
            print(
                f"[tf32] [matmul] torch.backends.cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}"
            )

    def dump_log(self):
        if not dist.is_local_master():
            return
        if "1/" in self.cur_ep:  # first time to dump log
            with open(self.log_file_path, "w") as fp:
                json.dump(
                    {
                        "is_master": dist.is_master(),
                        "name": self.exp_name,
                        "cmd": self.command,
                        "commit": self.commit_id,
                        "branch": self.branch,
                        "tb_log_dir_path": self.tensorboard_log_dir_path,
                    },
                    fp,
                    indent=0,
                )
                fp.write("\n")

        log_dict = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in {
                "it": self.cur_it,
                "ep": self.cur_ep,
                "lr": self.cur_lr,
                "wd": self.cur_wd,
                "grad_norm": self.grad_norm,
                "L_mean": self.L_mean,
                "L_tail": self.L_tail,
                "acc_mean": self.acc_mean,
                "acc_tail": self.acc_tail,
                "vL_mean": self.vL_mean,
                "vL_tail": self.vL_tail,
                "vacc_mean": self.vacc_mean,
                "vacc_tail": self.vacc_tail,
                "remain_time": self.remain_time,
                "finish_time": self.finish_time,
            }.items()
        }
        with open(self.log_file_path, "a") as fp:
            fp.write(f"{log_dict}\n")

    def __str__(self):
        s = []
        for k in self.class_variables.keys():
            if k not in {"device", "dbg_ks_fp"}:  # these are not serializable
                s.append(f"  {k:20s}: {getattr(self, k)}")
        s = "\n".join(s)
        return f"{{\n{s}\n}}\n"


def init_dist_and_get_args():
    for i in range(len(sys.argv)):
        if sys.argv[i].startswith("--local-rank="):
            del sys.argv[i]
            break
    args = Args(explicit_bool=True).parse_args(known_only=True)
    if args.local_debug:
        args.patch_numbers_str = "1_2_3"
        args.seed = 1
        args.ada_lin_init_multiplier = 1e-2
        args.ada_lin_gamma_init_multiplier = 1e-5
        args.use_shared_adaln = False
        args.use_fused_adamw = False
        args.progressive_training_ratio = 0.8
        args.progressive_initial_stage = 1
    else:
        if args.data_path == "/path/to/imagenet":
            raise ValueError(
                f'{"*"*40}  please specify --data_path=/path/to/imagenet  {"*"*40}'
            )

    if args.extra_args:
        print(
            f"=========================== WARNING: UNEXPECTED EXTRA ARGS ==========================="
        )

    os.makedirs(args.local_output_dir_path, exist_ok=True)
    misc.init_distributed_mode(local_out_path=args.local_output_dir_path, timeout=30)

    args.set_tf32(args.use_tf32)
    args.seed_everything(benchmark=args.progressive_training_ratio == 0)

    args.device = dist.get_device()
    args.patch_numbers = tuple(
        map(int, args.patch_numbers_str.replace("-", "_").split("_"))
    )
    args.resolutions = tuple(pn * args.patch_size for pn in args.patch_numbers)
    args.data_load_resolution = max(args.resolutions)

    bs_per_gpu = round(
        args.global_batch_size / args.grad_accumulation / dist.get_world_size()
    )
    args.batch_size = bs_per_gpu
    args.global_batch_size = args.glb_batch_size = (
        args.batch_size * dist.get_world_size()
    )
    args.workers = min(max(0, args.workers), args.batch_size)

    args.scaled_lr = args.grad_accumulation * args.base_lr * args.glb_batch_size / 256
    args.final_wd = args.final_wd or args.initial_wd

    if args.warmup_epochs == 0:
        args.warmup_epochs = args.epochs * 1 / 50

    if args.progressive_warmup_epochs == 0:
        args.progressive_warmup_epochs = args.epochs * 1 / 300
    if args.progressive_training_ratio > 0:
        args.lr_schedule = f"lin{args.progressive_training_ratio:g}"

    args.log_file_path = os.path.join(args.local_output_dir_path, "log.txt")
    args.last_checkpoint_path = os.path.join(
        args.local_output_dir_path, f"ar-ckpt-last.pth"
    )
    _reg_valid_name = re.compile(r"[^\w\-+,.]")
    tb_name = _reg_valid_name.sub(
        "_",
        f"tb-VARd{args.depth}"
        f"__pn{args.patch_numbers_str}"
        f"__b{args.global_batch_size}ep{args.epochs}{args.optimizer[:4]}lr{args.base_lr:g}wd{args.initial_wd:g}",
    )
    args.tensorboard_log_dir_path = os.path.join(args.local_output_dir_path, tb_name)

    return args
