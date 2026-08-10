"""Run ARGUS inference from local weights instead of an OpenAI-compatible endpoint.

``src.inference`` talks HTTP to a served model.  This module loads Qwen2.5-VL and
the trained LoRA adapters in-process with transformers + peft, so no ``swift
deploy`` / ``vllm serve`` is required.  Prompts, media loading, resume logic, and
the output schema are imported from ``src.inference`` unchanged, so the records
written here stay compatible with ``src.aggregate``, ``src.prepare``,
``src.quality``, and ``src.evaluate``.

The five adapters are held simultaneously under distinct peft names and selected
per call, so ``argus`` mode routes each observer to its own LoRA and the Judge to
the GRPO LoRA within a single loaded base model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from PIL import Image

from .inference import (
    COT_SYSTEM_PROMPT,
    JUDGE_SYSTEM_WITH_VIDEO,
    JUDGE_SYSTEM_WITHOUT_VIDEO,
    MULTITURN_SYSTEM_PROMPT,
    MULTITURN_TURNS,
    MULTITURN_VERDICT,
    OBSERVER_SPECS,
    ROLES,
    COT_USER_PROMPT,
    SINGLE_SYSTEM_PROMPT,
    SINGLE_USER_PROMPT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    _result_record,
    _slug,
    build_judge_input,
    compute_metrics,
    observations_from_row,
    parse_observation,
    merge_shards,
    read_records,
    record_complete,
    report_one,
    resolve_single_media,
    run_resumable,
    sample_images,
    shard_output_path,
    tagged_text,
    validated_final,
    write_json_atomic,
    _result_key,
)


WEIGHTS_ROOT = Path(r"C:\Users\admin\Desktop\work\argus-hf\weights")
DEFAULT_MODEL_ROOT = WEIGHTS_ROOT / "qwen2_5_vl_7b" / "main"
DEFAULT_JUDGE_LORA = DEFAULT_MODEL_ROOT / "grpo_video" / "lora" / "judge"
DEFAULT_OBSERVER_LORA_ROOT = DEFAULT_MODEL_ROOT / "shared" / "lora"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

JUDGE_ADAPTER = "judge"
TEMPORAL_PATCH_SIZE = 2


def _existing_dir(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{description} is not a directory: {resolved}")
    return resolved


def _peft_dir(path: Path, description: str) -> Path:
    resolved = _existing_dir(path, description)
    if not (resolved / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Not a peft adapter directory: {resolved}")
    return resolved


def adapter_paths(
    judge_lora: Path | None,
    observer_lora_root: Path | None,
    *,
    require_observers: bool,
    require_judge: bool,
    observer_loras: Mapping[str, Path | None] | None = None,
) -> Dict[str, Path]:
    """Map peft adapter name -> checkpoint directory, validating what is needed.

    Each observer can be given its own path; roles left unset fall back to
    ``<observer_lora_root>/<role>``.
    """
    adapters: Dict[str, Path] = {}
    if require_judge:
        adapters[JUDGE_ADAPTER] = _peft_dir(
            judge_lora or DEFAULT_JUDGE_LORA, "Judge LoRA"
        )
    if require_observers:
        explicit = dict(observer_loras or {})
        root: Path | None = None
        if any(explicit.get(role) is None for role in ROLES):
            root = _existing_dir(
                observer_lora_root or DEFAULT_OBSERVER_LORA_ROOT, "Observer LoRA root"
            )
        for role in ROLES:
            given = explicit.get(role)
            if given is not None:
                adapters[role] = _peft_dir(given, f"{role} LoRA")
            else:
                assert root is not None
                path = root / role
                if not (path / "adapter_config.json").is_file():
                    raise FileNotFoundError(f"Missing {role} adapter under {root}")
                adapters[role] = path
    return adapters


def required_adapters(mode: str, role: str, judge_video: bool) -> tuple[bool, bool]:
    """Return (needs the four observer LoRAs, needs the Judge LoRA)."""
    del judge_video  # the Judge LoRA is the same checkpoint either way
    if mode == "observer":
        return role in ROLES, False
    if mode == "judge":
        return False, True
    if mode in {"argus", "multiturn"}:
        return True, True
    return False, True  # single / cot: one generalist pass, served by the Judge LoRA


def even_frames(images: Sequence[Image.Image]) -> List[Image.Image]:
    """Qwen2.5-VL merges frames in pairs, so the count must be even."""
    frames = list(images)
    if not frames:
        raise ValueError("No frames were sampled from the video")
    if len(frames) % TEMPORAL_PATCH_SIZE:
        frames.append(frames[-1].copy())
    return frames


def text_turn(role: str, text: str) -> Dict[str, Any]:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def video_turn(text: str, frames: Sequence[Image.Image]) -> Dict[str, Any]:
    """A user turn whose frames become one <video> block, as in training."""
    if not frames:
        return text_turn("user", text)
    return {
        "role": "user",
        "content": [{"type": "video"}, {"type": "text", "text": text}],
    }


def image_turn(text: str, frames: Sequence[Image.Image]) -> Dict[str, Any]:
    """A user turn whose frames become separate <image> blocks."""
    if not frames:
        return text_turn("user", text)
    return {
        "role": "user",
        "content": [
            *({"type": "image"} for _ in frames),
            {"type": "text", "text": text},
        ],
    }


class LocalQwenVL:
    """Loads one Qwen2.5-VL base model plus every ARGUS LoRA, and generates text."""

    def __init__(
        self,
        base_model: str,
        adapters: Mapping[str, Path],
        *,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_impl: str = "sdpa",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_pixels_per_frame: int = 256 * 28 * 28,
        media: str = "video",
        fps: float = 2.0,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.media = media
        self.fps = fps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.adapter_names: List[str] = []

        torch_dtype = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(
            base_model, max_pixels=max_pixels_per_frame
        )
        for attribute in ("image_processor", "video_processor"):
            processor = getattr(self.processor, attribute, None)
            if processor is not None and hasattr(processor, "max_pixels"):
                processor.max_pixels = max_pixels_per_frame
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
        )
        if adapters:
            from peft import PeftModel

            names = list(adapters)
            first = names[0]
            self.model = PeftModel.from_pretrained(
                self.model, str(adapters[first]), adapter_name=first
            )
            for name in names[1:]:
                self.model.load_adapter(str(adapters[name]), adapter_name=name)
            self.adapter_names = names
            print(f"loaded adapters: {', '.join(names)}")
        self.model.eval()

    def _select(self, adapter: str) -> None:
        if not self.adapter_names:
            return
        if adapter not in self.adapter_names:
            raise ValueError(
                f"Adapter {adapter!r} was not loaded; available: {self.adapter_names}"
            )
        self.model.set_adapter(adapter)

    def _inputs(self, messages: Sequence[Mapping[str, Any]], frames):
        text = self.processor.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        kwargs: Dict[str, Any] = {"text": [text], "return_tensors": "pt", "padding": True}
        if frames:
            if self.media == "video":
                kwargs["videos"] = [list(frames)]
                kwargs["fps"] = [self.fps]
            else:
                kwargs["images"] = list(frames)
        try:
            return self.processor(**kwargs)
        except TypeError:
            kwargs.pop("fps", None)
            return self.processor(**kwargs)

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        frames: Sequence[Image.Image] = (),
        *,
        adapter: str = JUDGE_ADAPTER,
    ) -> tuple[str, Dict[str, Any]]:
        started = time.perf_counter()
        self._select(adapter)
        inputs = self._inputs(messages, frames).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p if self.temperature > 0 else None,
                pad_token_id=self.processor.tokenizer.pad_token_id
                or self.processor.tokenizer.eos_token_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = generated[0][prompt_length:]
        text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip(), {
            "name": adapter,
            "completion_tokens": int(new_tokens.shape[0]),
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }


def infer_one_local(
    row: Mapping[str, Any],
    *,
    mode: str,
    role: str,
    judge_video: bool,
    dataset_root: Path,
    num_frames: int,
    frame_size: int,
    model: LocalQwenVL,
) -> Dict[str, Any]:
    """Mirror of ``inference.infer_one`` backed by locally loaded weights."""
    started = time.perf_counter()
    calls: List[Dict[str, Any]] = []
    responses: Dict[str, str] = {}
    observations: Dict[str, str] = {}
    user_turn = video_turn if model.media == "video" else image_turn
    try:
        needs_frames = mode in {"single", "cot", "multiturn", "argus", "observer"} or (
            mode == "judge" and judge_video
        )
        frames: List[Image.Image] = []
        if needs_frames:
            frames = even_frames(
                sample_images(
                    dataset_root,
                    str(row.get("video_path", "")),
                    num_frames,
                    frame_size,
                )
            )

        if mode in {"single", "cot"}:
            single = mode == "single"
            system = SINGLE_SYSTEM_PROMPT if single else COT_SYSTEM_PROMPT
            query = SINGLE_USER_PROMPT if single else COT_USER_PROMPT
            text, stat = model.generate(
                [text_turn("system", system), user_turn(query, frames)],
                frames,
                adapter=JUDGE_ADAPTER,
            )
            calls.append(stat)
            responses["final"] = text
            if mode == "cot":
                observations = {name: tagged_text(text, name) for name in ROLES}
                missing = [name for name, value in observations.items() if not value]
                if missing:
                    raise ValueError(
                        f"CoT response is missing tags: {', '.join(missing)}"
                    )
            answer, explanation = validated_final(text, f"{mode} response")

        elif mode == "observer":
            spec = OBSERVER_SPECS[role]
            text, stat = model.generate(
                [
                    text_turn("system", spec["system"]),
                    user_turn(spec["query"], frames),
                ],
                frames,
                adapter=role,
            )
            calls.append(stat)
            responses[role] = text
            observations[role] = parse_observation(text)
            if not observations[role]:
                raise ValueError(f"{role} response is missing <observation> tags")
            answer, explanation = "", ""

        elif mode == "judge":
            observations = observations_from_row(row)
            missing = [name for name in ROLES if not observations.get(name)]
            if missing:
                raise ValueError(
                    f"Input row is missing observations: {', '.join(missing)}"
                )
            judge_input = build_judge_input(observations)
            system = (
                JUDGE_SYSTEM_WITH_VIDEO if judge_video else JUDGE_SYSTEM_WITHOUT_VIDEO
            )
            text, stat = model.generate(
                [
                    text_turn("system", system),
                    user_turn(judge_input, frames)
                    if judge_video
                    else text_turn("user", judge_input),
                ],
                frames if judge_video else (),
                adapter=JUDGE_ADAPTER,
            )
            calls.append(stat)
            responses["judge"] = text
            answer, explanation = validated_final(text, "Judge response")

        elif mode == "multiturn":
            history: List[Dict[str, Any]] = [
                text_turn("system", MULTITURN_SYSTEM_PROMPT)
            ]
            for index, name in enumerate(ROLES):
                prompt = MULTITURN_TURNS[name]
                history.append(
                    user_turn(prompt, frames) if index == 0 else text_turn("user", prompt)
                )
                text, stat = model.generate(history, frames, adapter=name)
                calls.append(stat)
                history.append(text_turn("assistant", text))
                responses[name] = text
                observations[name] = parse_observation(text)
                if not observations[name]:
                    raise ValueError(f"{name} response is missing <observation> tags")
            if judge_video:
                history.append(text_turn("user", MULTITURN_VERDICT))
                text, stat = model.generate(
                    history, frames, adapter=JUDGE_ADAPTER
                )
            else:
                text, stat = model.generate(
                    [
                        text_turn("system", JUDGE_SYSTEM_WITHOUT_VIDEO),
                        text_turn("user", build_judge_input(observations)),
                    ],
                    (),
                    adapter=JUDGE_ADAPTER,
                )
            calls.append(stat)
            responses["judge"] = text
            answer, explanation = validated_final(text, "multi-turn Judge response")

        elif mode == "argus":
            # Sequential, not threaded: set_adapter mutates shared model state.
            for name in ROLES:
                spec = OBSERVER_SPECS[name]
                text, stat = model.generate(
                    [
                        text_turn("system", spec["system"]),
                        user_turn(spec["query"], frames),
                    ],
                    frames,
                    adapter=name,
                )
                calls.append(stat)
                responses[name] = text
                observations[name] = parse_observation(text)
                if not observations[name]:
                    raise ValueError(f"{name} response is missing <observation> tags")
            judge_input = build_judge_input(observations)
            system = (
                JUDGE_SYSTEM_WITH_VIDEO if judge_video else JUDGE_SYSTEM_WITHOUT_VIDEO
            )
            text, stat = model.generate(
                [
                    text_turn("system", system),
                    user_turn(judge_input, frames)
                    if judge_video
                    else text_turn("user", judge_input),
                ],
                frames if judge_video else (),
                adapter=JUDGE_ADAPTER,
            )
            calls.append(stat)
            responses["judge"] = text
            answer, explanation = validated_final(text, "ARGUS Judge response")

        else:
            raise ValueError(f"Unsupported inference mode: {mode}")

        return _result_record(
            row,
            mode=mode,
            judge_video=judge_video,
            answer=answer,
            explanation=explanation,
            observations=observations,
            responses=responses,
            calls=calls,
            wall_sec=time.perf_counter() - started,
        )
    except Exception as exc:  # one bad clip must not abort a long run
        return _result_record(
            row,
            mode=mode,
            judge_video=judge_video,
            answer="Error" if mode != "observer" else "",
            observations=observations,
            responses=responses,
            calls=calls,
            wall_sec=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("argus", "single", "cot", "multiturn", "observer", "judge")
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Manifest/report JSON or JSONL.")
    source.add_argument(
        "--video",
        type=Path,
        help=(
            "Run one local video file (.mp4/.avi/.mov/.mkv/.webm), one image, or one "
            "directory of extracted frames, and print the verdict on the terminal."
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--role", choices=ROLES, default="texture", help="Used by observer mode."
    )

    weights = parser.add_argument_group("local weights")
    weights.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base Qwen2.5-VL path or Hub id the adapters were trained on.",
    )
    weights.add_argument(
        "--judge-lora",
        type=Path,
        default=DEFAULT_JUDGE_LORA,
        help=f"Judge adapter directory (default: {DEFAULT_JUDGE_LORA}).",
    )
    weights.add_argument(
        "--observer-lora-root",
        type=Path,
        default=DEFAULT_OBSERVER_LORA_ROOT,
        help=(
            "Directory holding texture/lighting/motion/physics subdirectories, used "
            f"for any role without its own flag (default: {DEFAULT_OBSERVER_LORA_ROOT})."
        ),
    )
    for role in ROLES:
        weights.add_argument(
            f"--{role}-lora",
            type=Path,
            default=None,
            help=f"{role} adapter directory; overrides --observer-lora-root.",
        )
    weights.add_argument(
        "--no-lora",
        action="store_true",
        help="Run the untuned base model, ignoring every adapter.",
    )
    weights.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    weights.add_argument("--device-map", default="auto")
    weights.add_argument("--attn-impl", default="sdpa", choices=("sdpa", "eager", "flash_attention_2"))

    video = parser.add_mutually_exclusive_group()
    video.add_argument(
        "--with-video",
        "--judge-video",
        dest="judge_video",
        action="store_true",
        help="Let the final Judge see the sampled frames (default).",
    )
    video.add_argument(
        "--without-video",
        "--no-judge-video",
        dest="judge_video",
        action="store_false",
        help="Make the final Judge use only the four text reports.",
    )
    parser.set_defaults(judge_video=True)

    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument(
        "--frame-size",
        type=int,
        default=256,
        help="Longest frame edge in pixels; matches the training default.",
    )
    parser.add_argument(
        "--media",
        choices=("video", "images"),
        default="video",
        help="Feed frames as one <video> block (training format) or as <image> blocks.",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--max-pixels-per-frame",
        type=int,
        default=256 * 28 * 28,
        help="Vision token budget per frame; 256*28*28 matches training.",
    )
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode in {"single", "cot"} and not args.judge_video:
        raise ValueError(f"{args.mode} has no report-only Judge; use --with-video")
    if args.mode == "judge" and args.video:
        raise ValueError("judge mode requires --input rows containing four observations")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if args.num_frames < 1 or args.frame_size < 1:
        raise ValueError("--num-frames and --frame-size must be positive")
    if args.max_new_tokens < 1 or args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("Invalid --max-new-tokens, --temperature, or --top-p")
    if args.max_pixels_per_frame < 28 * 28 or args.fps <= 0:
        raise ValueError("--max-pixels-per-frame and --fps must be positive")
    if args.flush_every < 1:
        raise ValueError("--flush-every must be positive")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")

    if args.video:
        if args.num_shards > 1:
            raise ValueError("--num-shards applies to --input manifests, not --video")
        video = resolve_single_media(args.video)
        rows: List[Dict[str, Any]] = [{"video_path": str(video)}]
        input_stem = video.stem
    else:
        rows = read_records(args.input.expanduser().resolve())
        input_stem = args.input.stem
    seen: set[str] = set()
    for index, row in enumerate(rows):
        key = _result_key(row)
        if key in seen:
            raise ValueError(f"Duplicate video_path at input row {index}: {key}")
        seen.add(key)
    input_keys = [_result_key(row) for row in rows]
    rows = [
        row
        for index, row in enumerate(rows)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        rows = rows[: args.limit]

    needs_observers, needs_judge = required_adapters(
        args.mode, args.role, args.judge_video
    )
    adapters = (
        {}
        if args.no_lora
        else adapter_paths(
            args.judge_lora,
            args.observer_lora_root,
            require_observers=needs_observers,
            require_judge=needs_judge,
            observer_loras={
                role: getattr(args, f"{role}_lora") for role in ROLES
            },
        )
    )

    tag = args.mode if args.mode != "observer" else f"observer-{args.role}"
    if args.mode in {"argus", "multiturn", "judge"}:
        tag += "-video" if args.judge_video else "-text"
    label = "base" if args.no_lora else _slug(Path(args.base_model).name) + "-lora"
    merged_output = (
        args.output
        or (DEFAULT_OUTPUT_ROOT / "local" / label / f"{tag}-{input_stem}.jsonl")
    ).expanduser().resolve()
    output = shard_output_path(merged_output, args.num_shards, args.shard_index)
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir() and not args.video:
        raise FileNotFoundError(dataset_root)

    model = LocalQwenVL(
        args.base_model,
        adapters,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_impl=args.attn_impl,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_pixels_per_frame=args.max_pixels_per_frame,
        media=args.media,
        fps=args.fps,
    )

    def process(row: Mapping[str, Any]) -> Dict[str, Any]:
        return infer_one_local(
            row,
            mode=args.mode,
            role=args.role,
            judge_video=args.judge_video,
            dataset_root=dataset_root,
            num_frames=args.num_frames,
            frame_size=args.frame_size,
            model=model,
        )

    results = run_resumable(
        rows,
        process,
        output=output,
        mode=args.mode,
        role=args.role,
        concurrency=1,  # one GPU model, and set_adapter is shared state
        flush_every=args.flush_every,
        overwrite=args.overwrite,
    )
    print(f"wrote {len(results)} records to {output}")
    if args.video and results:
        report_one(results[0])
    if args.mode == "observer":
        complete = sum(record_complete(row, args.mode, args.role) for row in results)
        print(f"valid {args.role} observations: {complete}/{len(results)}")
        return
    merged = merge_shards(merged_output, args.num_shards, input_keys)
    if merged is None and args.num_shards > 1:
        return  # other shards are still running; they will merge and score
    scored = merged if merged is not None else results
    scored_output = merged_output if merged is not None else output
    metrics = compute_metrics(scored)
    write_json_atomic(
        scored_output.with_name(f"{scored_output.stem}_metrics.json"), metrics
    )
    if metrics["total"]:
        print(
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
            f"fake_recall={metrics['fake_recall']:.4f} "
            f"fake_f1={metrics['fake_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
