"""Prepare the three ms-swift datasets used by ARGUS.

The commands are deliberately model-size agnostic: Qwen2.5-VL 3B, 7B, and 32B
consume the same JSONL. Only the training launcher's ``--model-size`` changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

from .inference import (
    JUDGE_SYSTEM_WITHOUT_VIDEO,
    JUDGE_SYSTEM_WITH_VIDEO,
    OBSERVER_SPECS,
    ROLES,
    build_judge_input,
    extract_rationale,
    iter_records,
    observations_from_row,
    sampled_frame_paths,
    strict_zip_records,
    write_records_atomic,
)


VALID_SOURCE_LABELS = {"real", "fake", "efs", "fs", "fr"}


def required(value: Any, description: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        raise ValueError(f"{description} is empty")
    return text


def source_label(value: Any) -> str:
    label = required(value, "label").lower()
    if label not in VALID_SOURCE_LABELS:
        raise ValueError(f"Unsupported source label: {value!r}")
    return label


def binary_label(value: Any) -> str:
    return "real" if source_label(value) == "real" else "fake"


def format_observation(text: str) -> str:
    text = required(text, "observation")
    return f"<observation>\n{text}\n</observation>"


def media_fields(
    dataset_root: Path,
    video_path: str,
    num_frames: int,
    frame_size: int,
) -> Dict[str, Any]:
    selected = sampled_frame_paths(dataset_root, video_path, num_frames)
    template: Dict[str, Any] = {
        "enable_thinking": False,
        "max_pixels": frame_size * frame_size,
    }
    if isinstance(selected, list):
        media: Any = [str(path) for path in selected]
    else:
        media = str(selected)
        template["nframes"] = num_frames
    return {"videos": [media], "chat_template_kwargs": template}


def observer_training_row(
    row: Mapping[str, Any],
    role: str,
    dataset_root: Path,
    num_frames: int,
    frame_size: int,
) -> Dict[str, Any]:
    video_path = required(row.get("video_path"), "video_path")
    observation = required(
        observations_from_row(row).get(role), f"{role} observation for {video_path}"
    )
    spec = OBSERVER_SPECS[role]
    result = {
        "messages": [
            {"role": "system", "content": spec["system"]},
            {"role": "user", "content": "<video>\n" + spec["query"]},
            {"role": "assistant", "content": format_observation(observation)},
        ],
        **media_fields(dataset_root, video_path, num_frames, frame_size),
    }
    return result


def parse_report_specs(values: Sequence[str]) -> Dict[str, Path]:
    reports: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--report must use ROLE=PATH syntax: {value}")
        role, raw_path = value.split("=", 1)
        role = role.strip().lower()
        if role not in ROLES or role in reports:
            raise ValueError(f"Invalid or duplicate report role: {role!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        reports[role] = path
    missing = [role for role in ROLES if role not in reports]
    if missing:
        raise ValueError(f"Missing --report entries: {', '.join(missing)}")
    return reports


def joined_reports(
    authority: Path,
    reports: Mapping[str, Path],
) -> Iterator[tuple[Dict[str, Any], Dict[str, str]]]:
    paths = [authority, *[reports[role] for role in ROLES]]
    for index, rows in enumerate(strict_zip_records(*paths)):
        authority_row = rows[0]
        expected_label = binary_label(authority_row.get("label"))
        observations: Dict[str, str] = {}
        for role, report_row in zip(ROLES, rows[1:]):
            report_label = str(report_row.get("label", "")).strip()
            if report_label and binary_label(report_label) != expected_label:
                raise ValueError(
                    f"Label mismatch at row {index} for {role}: "
                    f"{expected_label} != {report_label}"
                )
            observations[role] = required(
                observations_from_row(report_row).get(role),
                f"{role} report for {authority_row.get('video_path')}",
            )
        yield authority_row, observations


def judge_sft_row(
    authority: Mapping[str, Any],
    observations: Mapping[str, str],
    *,
    with_video: bool,
    dataset_root: Path,
    num_frames: int,
    frame_size: int,
) -> Dict[str, Any]:
    video_path = required(authority.get("video_path"), "video_path")
    explanation = required(
        extract_rationale(authority), f"explanation for {video_path}"
    )
    if "<answer" in explanation.lower():
        raise ValueError(f"Explanation contains an answer tag: {video_path}")
    prompt = build_judge_input(observations)
    result: Dict[str, Any] = {
        "messages": [
            {
                "role": "system",
                "content": JUDGE_SYSTEM_WITH_VIDEO
                if with_video
                else JUDGE_SYSTEM_WITHOUT_VIDEO,
            },
            {
                "role": "user",
                "content": ("<video>\n" + prompt) if with_video else prompt,
            },
            {
                "role": "assistant",
                "content": (
                    f"<explanation>\n{explanation}\n</explanation>\n"
                    f"<answer>{binary_label(authority.get('label'))}</answer>"
                ),
            },
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if with_video:
        media = media_fields(dataset_root, video_path, num_frames, frame_size)
        result["videos"] = media["videos"]
        result["chat_template_kwargs"].update(media["chat_template_kwargs"])
    return result


def judge_grpo_row(
    authority: Mapping[str, Any],
    observations: Mapping[str, str],
    *,
    with_video: bool,
    dataset_root: Path,
    num_frames: int,
    frame_size: int,
) -> Dict[str, Any]:
    video_path = required(authority.get("video_path"), "video_path")
    prompt = build_judge_input(observations)
    result: Dict[str, Any] = {
        "messages": [
            {
                "role": "system",
                "content": JUDGE_SYSTEM_WITH_VIDEO
                if with_video
                else JUDGE_SYSTEM_WITHOUT_VIDEO,
            },
            {
                "role": "user",
                "content": ("<video>\n" + prompt) if with_video else prompt,
            },
        ],
        "solution": binary_label(authority.get("label")),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if with_video:
        media = media_fields(dataset_root, video_path, num_frames, frame_size)
        result["videos"] = media["videos"]
        result["chat_template_kwargs"].update(media["chat_template_kwargs"])
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    observer = subparsers.add_parser(
        "observer", help="Prepare one Observation Agent SFT dataset."
    )
    observer.add_argument("--role", required=True, choices=ROLES)
    observer.add_argument("--source", required=True, type=Path)
    observer.add_argument("--output", required=True, type=Path)
    observer.add_argument("--dataset-root", required=True, type=Path)

    judge = subparsers.add_parser("judge", help="Prepare text/video Judge SFT data.")
    judge.add_argument("--explanations", required=True, type=Path)
    judge.add_argument("--report", action="append", required=True, metavar="ROLE=PATH")
    judge.add_argument("--output", required=True, type=Path)
    judge.add_argument("--dataset-root", type=Path, default=Path("."))
    judge.add_argument(
        "--with-video", action=argparse.BooleanOptionalAction, default=False
    )

    grpo = subparsers.add_parser("grpo", help="Prepare text/video Judge GRPO data.")
    grpo.add_argument("--manifest", required=True, type=Path)
    grpo.add_argument("--report", action="append", required=True, metavar="ROLE=PATH")
    grpo.add_argument("--output", required=True, type=Path)
    grpo.add_argument("--dataset-root", type=Path, default=Path("."))
    grpo.add_argument(
        "--with-video", action=argparse.BooleanOptionalAction, default=False
    )

    for subparser in (observer, judge, grpo):
        subparser.add_argument("--num-frames", type=int, default=16)
        subparser.add_argument("--frame-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".jsonl":
        raise ValueError("Prepared ms-swift datasets must use a .jsonl output path")
    if args.num_frames < 1 or args.frame_size < 1:
        raise ValueError("--num-frames and --frame-size must be positive")

    if args.command == "observer":
        source = args.source.expanduser().resolve()
        dataset_root = args.dataset_root.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if not dataset_root.is_dir():
            raise FileNotFoundError(dataset_root)
        rows = (
            observer_training_row(
                row, args.role, dataset_root, args.num_frames, args.frame_size
            )
            for row in iter_records(source)
        )
    else:
        reports = parse_report_specs(args.report)
        authority = (
            (args.explanations if args.command == "judge" else args.manifest)
            .expanduser()
            .resolve()
        )
        if not authority.is_file():
            raise FileNotFoundError(authority)
        dataset_root = args.dataset_root.expanduser().resolve()
        if args.with_video and not dataset_root.is_dir():
            raise FileNotFoundError(dataset_root)
        joined = joined_reports(authority, reports)
        if args.command == "judge":
            rows = (
                judge_sft_row(
                    authority_row,
                    observations,
                    with_video=args.with_video,
                    dataset_root=dataset_root,
                    num_frames=args.num_frames,
                    frame_size=args.frame_size,
                )
                for authority_row, observations in joined
            )
        else:
            rows = (
                judge_grpo_row(
                    authority_row,
                    observations,
                    with_video=args.with_video,
                    dataset_root=dataset_root,
                    num_frames=args.num_frames,
                    frame_size=args.frame_size,
                )
                for authority_row, observations in joined
            )

    count = 0

    def counted_rows() -> Iterator[Mapping[str, Any]]:
        nonlocal count
        for row in rows:
            count += 1
            if count % 1000 == 0:
                print(f"prepared {count} rows", flush=True)
            yield row

    write_records_atomic(output, counted_rows())
    print(f"wrote {count} rows to {output}")


if __name__ == "__main__":
    main()
