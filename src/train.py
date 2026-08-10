"""Thin ms-swift launcher for the ARGUS SFT and GRPO runs.

The data schema is identical across Qwen2.5-VL 3B, 7B, and 32B, so ``--model`` is
the only knob that changes between sizes: the memory defaults are chosen from the
3b/7b/32b marker in its path.  Everything after ``--`` is forwarded to ms-swift.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

from .inference import ROLES


DEFAULT_MODEL = os.environ.get("QWEN2_5_VL_7B", "Qwen/Qwen2.5-VL-7B-Instruct")

# Checked longest-first so that "32b" is never read as "3b".
MODEL_SIZES = ("32b", "7b", "3b")

SFT_DEFAULTS = {
    "3b": {"batch": 8, "accumulation": 1},
    "7b": {"batch": 2, "accumulation": 4},
    "32b": {"batch": 1, "accumulation": 8},
}


def model_size(model: str, override: str | None) -> str:
    """Infer 3b/7b/32b from the model path so only --model has to be passed."""
    if override:
        return override
    name = Path(model).name.lower()
    for size in MODEL_SIZES:
        if size in name:
            return size
    raise ValueError(
        f"Cannot infer the model size from {model!r}; pass --model-size explicitly"
    )


def gpu_count(gpus: str) -> int:
    devices = [device.strip() for device in gpus.split(",") if device.strip()]
    if not devices:
        raise ValueError("--gpus must contain at least one device")
    return len(devices)


def forwarded(values: Sequence[str]) -> List[str]:
    values = list(values)
    return values[1:] if values and values[0] == "--" else values


def common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Local path or Hub id of the base model. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--model-size",
        default=None,
        choices=SFT_DEFAULTS,
        help="Only needed when 3b/7b/32b cannot be read from --model.",
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--gpus",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        help="Comma-separated CUDA device ids.",
    )
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--deepspeed", default="zero2")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "swift_args", nargs=argparse.REMAINDER, help="Extra ms-swift args after --."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    sft = subparsers.add_parser("sft", help="LoRA SFT for one observer or Judge.")
    common_parser(sft)
    sft.add_argument("--role", required=True, choices=(*ROLES, "judge"))
    sft.add_argument(
        "--with-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For Judge data only. Observers always use video.",
    )
    sft.add_argument("--epochs", type=float, default=1.0)
    sft.add_argument("--learning-rate", type=float, default=1e-4)
    sft.add_argument("--batch-size", type=int, default=None)
    sft.add_argument("--gradient-accumulation", type=int, default=None)
    sft.add_argument("--max-length", type=int, default=None)
    sft.add_argument("--lora-rank", type=int, default=16)
    sft.add_argument("--lora-alpha", type=int, default=32)
    sft.add_argument("--save-steps", type=int, default=100)
    sft.add_argument("--logging-steps", type=int, default=5)

    grpo = subparsers.add_parser("grpo", help="7B Judge GRPO initialized from SFT.")
    common_parser(grpo)
    grpo.add_argument(
        "--adapter", type=Path, default=None, help="Judge SFT LoRA initialization."
    )
    grpo.add_argument(
        "--with-video", action=argparse.BooleanOptionalAction, default=False
    )
    grpo.add_argument("--epochs", type=float, default=1.0)
    grpo.add_argument("--learning-rate", type=float, default=5e-6)
    grpo.add_argument("--batch-size", type=int, default=1)
    grpo.add_argument("--gradient-accumulation", type=int, default=8)
    grpo.add_argument("--max-length", type=int, default=None)
    grpo.add_argument("--max-completion-length", type=int, default=512)
    grpo.add_argument("--num-generations", type=int, default=8)
    grpo.add_argument("--temperature", type=float, default=0.7)
    grpo.add_argument("--beta", type=float, default=0.04)
    grpo.add_argument("--vllm-memory", type=float, default=0.5)
    grpo.add_argument("--lora-rank", type=int, default=16)
    grpo.add_argument("--lora-alpha", type=int, default=32)
    grpo.add_argument(
        "--reward-func",
        action="append",
        choices=("judge_accuracy", "judge_length", "judge_format"),
        default=None,
    )
    grpo.add_argument("--reward-weight", action="append", type=float, default=None)
    grpo.add_argument("--save-steps", type=int, default=50)
    return parser.parse_args(argv)


def sft_command(args: argparse.Namespace) -> List[str]:
    defaults = SFT_DEFAULTS[model_size(args.model, args.model_size)]
    batch = args.batch_size or defaults["batch"]
    accumulation = args.gradient_accumulation or defaults["accumulation"]
    uses_video = args.role != "judge" or args.with_video
    max_length = args.max_length or (4608 if uses_video else 3584)
    command = [
        "swift",
        "sft",
        "--model",
        args.model,
        "--dataset",
        str(args.dataset.expanduser().resolve()),
        "--load_from_cache_file",
        "true",
        "--dataset_shuffle",
        "true",
        "--split_dataset_ratio",
        "0",
        "--eval_strategy",
        "no",
        "--enable_thinking",
        "false",
        "--tuner_type",
        "lora",
        "--target_modules",
        "all-linear",
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--freeze_llm",
        "false",
        "--freeze_vit",
        "true",
        "--freeze_aligner",
        "true",
        "--torch_dtype",
        "bfloat16",
        "--attn_impl",
        "sdpa",
        "--padding_free",
        "false",
        "--gradient_checkpointing",
        "true",
        "--gradient_checkpointing_kwargs",
        '{"use_reentrant": false}',
        "--max_length",
        str(max_length),
        "--num_train_epochs",
        str(args.epochs),
        "--per_device_train_batch_size",
        str(batch),
        "--gradient_accumulation_steps",
        str(accumulation),
        "--learning_rate",
        str(args.learning_rate),
        "--warmup_ratio",
        "0.05",
        "--max_grad_norm",
        "1.0",
        "--deepspeed",
        args.deepspeed,
        "--report_to",
        "tensorboard",
        "--logging_dir",
        str(args.output.expanduser().resolve() / "tensorboard"),
        "--logging_steps",
        str(args.logging_steps),
        "--save_strategy",
        "steps",
        "--save_steps",
        str(args.save_steps),
        "--save_total_limit",
        "3",
        "--save_only_model",
        "false",
        "--dataset_num_proc",
        "8",
        "--dataloader_num_workers",
        "4",
        "--output_dir",
        str(args.output.expanduser().resolve()),
    ]
    if args.resume:
        command.extend(
            [
                "--resume_from_checkpoint",
                str(args.resume.expanduser().resolve()),
                "--resume_only_model",
                "false",
            ]
        )
    return command + forwarded(args.swift_args)


def grpo_command(args: argparse.Namespace) -> List[str]:
    if model_size(args.model, args.model_size) != "7b":
        raise ValueError("The paper's GRPO stage is defined only for Qwen2.5-VL-7B")
    rewards = args.reward_func or [
        "judge_accuracy",
        "judge_length",
        "judge_format",
    ]
    weights = args.reward_weight or (
        [1.0, 0.1, 0.1] if args.reward_func is None else [1.0] * len(rewards)
    )
    if len(rewards) != len(weights):
        raise ValueError("--reward-func and --reward-weight must have the same count")
    max_length = args.max_length or (4608 if args.with_video else 3584)
    plugin = Path(__file__).with_name("grpo_plugin.py").resolve()
    command = [
        "swift",
        "rlhf",
        "--rlhf_type",
        "grpo",
        "--model",
        args.model,
        "--dataset",
        str(args.dataset.expanduser().resolve()),
        "--load_from_cache_file",
        "true",
        "--dataset_shuffle",
        "true",
        "--split_dataset_ratio",
        "0",
        "--eval_strategy",
        "no",
        "--external_plugins",
        str(plugin),
        "--reward_funcs",
        *rewards,
        "--reward_weights",
        *[str(weight) for weight in weights],
        "--enable_thinking",
        "false",
        "--loss_scale",
        "default+ignore_empty_think",
        "--tuner_type",
        "lora",
        "--target_modules",
        "all-linear",
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--freeze_llm",
        "false",
        "--freeze_vit",
        "true",
        "--freeze_aligner",
        "true",
        "--torch_dtype",
        "bfloat16",
        "--attn_impl",
        "sdpa",
        "--padding_free",
        "false",
        "--gradient_checkpointing",
        "true",
        "--gradient_checkpointing_kwargs",
        '{"use_reentrant": false}',
        "--use_vllm",
        "true",
        "--vllm_mode",
        "colocate",
        "--vllm_tensor_parallel_size",
        "1",
        "--vllm_gpu_memory_utilization",
        str(args.vllm_memory),
        "--vllm_max_model_len",
        str(max_length),
        "--vllm_max_lora_rank",
        str(args.lora_rank),
        "--max_length",
        str(max_length),
        "--max_completion_length",
        str(args.max_completion_length),
        "--num_generations",
        str(args.num_generations),
        "--temperature",
        str(args.temperature),
        "--beta",
        str(args.beta),
        "--num_train_epochs",
        str(args.epochs),
        "--per_device_train_batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation),
        "--learning_rate",
        str(args.learning_rate),
        "--warmup_ratio",
        "0.05",
        "--max_grad_norm",
        "1.0",
        "--deepspeed",
        args.deepspeed,
        "--report_to",
        "tensorboard",
        "--logging_steps",
        "1",
        "--save_strategy",
        "steps",
        "--save_steps",
        str(args.save_steps),
        "--save_total_limit",
        "2",
        "--save_only_model",
        "false",
        "--dataset_num_proc",
        "8",
        "--dataloader_num_workers",
        "4",
        "--output_dir",
        str(args.output.expanduser().resolve()),
        "--log_completions",
        "true",
    ]
    if args.resume:
        command.extend(
            [
                "--resume_from_checkpoint",
                str(args.resume.expanduser().resolve()),
                "--resume_only_model",
                "false",
            ]
        )
        if args.adapter:
            command.extend(["--ref_adapters", str(args.adapter.expanduser().resolve())])
    elif args.adapter:
        adapter = str(args.adapter.expanduser().resolve())
        command.extend(["--adapters", adapter, "--ref_adapters", adapter])
    return command + forwarded(args.swift_args)


def training_environment(args: argparse.Namespace, uses_video: bool) -> Dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpus
    environment["NPROC_PER_NODE"] = str(args.nproc_per_node or gpu_count(args.gpus))
    environment["MASTER_PORT"] = str(args.master_port)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if uses_video:
        environment.update(
            {
                "FPS_MIN_FRAMES": "16",
                "FPS_MAX_FRAMES": "16",
                "VIDEO_MIN_TOKEN_NUM": "256",
                "VIDEO_MAX_TOKEN_NUM": "256",
            }
        )
    return environment


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    devices = gpu_count(args.gpus)
    if args.nproc_per_node is not None and not 1 <= args.nproc_per_node <= devices:
        raise ValueError("--nproc-per-node must be between 1 and the number of --gpus")
    if not 1 <= args.master_port <= 65535:
        raise ValueError("--master-port must be between 1 and 65535")
    if args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("--epochs and --learning-rate must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.gradient_accumulation is not None and args.gradient_accumulation < 1:
        raise ValueError("--gradient-accumulation must be positive")
    if args.max_length is not None and args.max_length < 1:
        raise ValueError("--max-length must be positive")
    if args.lora_rank < 1 or args.lora_alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    args.output = args.output.expanduser().resolve()
    if args.resume and not args.resume.expanduser().resolve().is_dir():
        raise FileNotFoundError(args.resume)
    if args.stage == "grpo" and args.adapter:
        if not args.adapter.expanduser().resolve().is_dir():
            raise FileNotFoundError(args.adapter)

    if args.stage == "sft":
        if args.save_steps < 1 or args.logging_steps < 1:
            raise ValueError("save and logging intervals must be positive")
        command = sft_command(args)
        uses_video = args.role != "judge" or args.with_video
    else:
        if args.max_completion_length < 1 or args.num_generations < 2:
            raise ValueError("GRPO completion length must be positive and generations at least 2")
        if args.temperature <= 0 or args.beta < 0 or not 0 < args.vllm_memory <= 1:
            raise ValueError("Invalid GRPO temperature, beta, or vLLM memory setting")
        if args.save_steps < 1:
            raise ValueError("--save-steps must be positive")
        command = grpo_command(args)
        uses_video = args.with_video
    environment = training_environment(args, uses_video)
    rendered_environment = (
        f"CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']} "
        f"NPROC_PER_NODE={environment['NPROC_PER_NODE']} MASTER_PORT={environment['MASTER_PORT']}"
    )
    print(rendered_environment)
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
