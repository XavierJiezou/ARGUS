"""Aggregate raw annotator outputs into ARGUS observation and explanation labels.

Every input must follow the manifest order exactly.  The command streams the files
together and fails immediately on a missing, duplicated, or misordered video rather
than silently joining incompatible runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

from .inference import (
    ROLES,
    add_api_arguments,
    binary_label,
    client_from_args,
    extract_rationale,
    iter_records,
    observations_from_row,
    parse_answer,
    parse_observation,
    repair_truncated_jsonl,
    strict_zip_records,
    write_records_atomic,
)


OBSERVATION_SYSTEM = """You are a video forensics expert. Multiple AI models have each analyzed the same video from a specific analytical perspective and produced independent observation reports. Your task is to read all of their reports and synthesize them into a single, coherent, comprehensive observation.

Aggregate multi-agent observations with the video label (real or fake) in mind: use the ground-truth label solely as internal guidance to resolve conflicts between models when their reports disagree.

Rules:
- Write ONE unified observation paragraph (100-200 words).
- Do NOT list models by name (e.g. "Model 1 said..."). Present the observations as a single cohesive expert analysis.
- Resolve conflicts: when models disagree, prioritise observations consistent with the video label and note discrepancies as ambiguous cues.
- Focus on concrete forensic signals and visual evidence mentioned by the models.
- Do NOT output a real/fake verdict. Only output the observation text.
- Do NOT mention or reference the ground-truth label, the video's real/fake status, or whether the video is real or fake in your output. The observation must read as if the observer does not know the verdict.
- Do NOT wrap your output in any tags. Output plain text only."""

EXPLANATION_SYSTEM = """You are a video forensics expert consolidating multiple annotator rationales into one reference explanation for face-video forgery detection.

Write one coherent, evidence-grounded paragraph that is consistent with the supplied ground-truth verdict. Retain only concrete visual evidence from the reports, resolve contradictions, remove unsupported claims and repetition, and make the causal link between the evidence and the verdict clear.

When one or more annotators predicted the correct verdict, prioritize those correct rationales. When all annotators were wrong, re-evaluate the supplied consolidated texture, lighting, motion, and physical-plausibility observations against the ground-truth verdict instead of copying an incorrect conclusion.

Return only the rationale text. Do not mention model names, voting, the aggregation process, or the ground-truth label as metadata. Do not output XML tags or a separate answer."""

BLIND_EXPLANATION_SYSTEM = """You are a video forensics expert consolidating several independent rationales about the same face video.

You are deliberately not given the annotators' predicted answers or the video's ground-truth verdict. Synthesize only the concrete visual evidence contained in the rationales into one coherent reference explanation. Preserve genuine disagreements as ambiguity, remove repetition and unsupported claims, and do not invent evidence.

Do not infer, state, or hint at a final real/fake answer. Do not mention model names, voting, labels, or the aggregation process. Return only one plain-text rationale paragraph without XML tags."""

# dimension_name / dimension_description as instantiated in the paper appendix.
ROLE_NAMES = {
    "texture": "texture and detail",
    "lighting": "lighting",
    "motion": "motion",
    "physics": "physical plausibility",
}

ROLE_DESCRIPTIONS = {
    "texture": "skin texture, edge sharpness, blending, material consistency, and fine detail",
    "lighting": "light direction, highlights, shadows, reflections, and illumination coherence",
    "motion": "temporal continuity, movement, flicker, jitter, and action plausibility",
    "physics": "hair and clothing dynamics, occlusion, perspective, geometry, and deformation",
}


def required(value: Any, description: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        raise ValueError(f"{description} is empty")
    return text


def gold_binary(value: Any) -> str:
    return binary_label(required(value, "label"))


def parse_sources(values: Sequence[str]) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    names: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"--input must use NAME=PATH syntax: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser().resolve()
        if not name or name in names:
            raise ValueError(f"Invalid or duplicate source name: {name!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        names.add(name)
        sources.append((name, path))
    if not sources:
        raise ValueError("At least one --input NAME=PATH is required")
    return sources


def sanitize_plain(text: str, description: str) -> str:
    value = required(text, description)
    observation = parse_observation(value, allow_plain=True)
    if observation:
        value = observation
    explanation = re.search(
        r"<explanation>\s*(.*?)\s*</explanation>", value, re.IGNORECASE | re.DOTALL
    )
    if explanation:
        value = explanation.group(1).strip()
    if re.search(r"</?answer\b", value, re.IGNORECASE):
        raise ValueError(f"{description} unexpectedly contains an answer tag")
    return required(value, description)


def observation_prompt(
    label: str,
    role: str,
    reports: Sequence[tuple[str, str]],
) -> str:
    blocks = [
        (
            f"This video is {label}.\n"
            f"Below are {len(reports)} observation reports from forensic models that "
            f"analyzed the {ROLE_NAMES[role]} of this video. Each report focuses on "
            f"{ROLE_DESCRIPTIONS[role]}."
        ),
    ]
    blocks.extend(
        f"--- Model {index} ({name}) ---\n{text}"
        for index, (name, text) in enumerate(reports, start=1)
    )
    return "\n\n".join(blocks)


def explanation_prompt(
    label: str,
    reports: Sequence[tuple[str, str, str]],
) -> str:
    blocks = [
        f"Ground-truth verdict: {label}.",
        "Consolidate the following annotator rationales that predicted this verdict correctly.",
    ]
    blocks.extend(
        f"--- Rationale {index} ({name}) ---\nPredicted answer: {answer}\n{text}"
        for index, (name, answer, text) in enumerate(reports, start=1)
    )
    return "\n\n".join(blocks)


def all_wrong_prompt(
    label: str,
    observations: Mapping[str, str],
    reports: Sequence[tuple[str, str, str]],
) -> str:
    blocks = [
        f"Ground-truth verdict: {label}.",
        (
            "All annotators predicted the wrong verdict. Re-evaluate their rationales "
            "against the consolidated forensic observations and explain the ground-truth verdict."
        ),
        "Consolidated observations:\n"
        + "\n".join(f"{role.capitalize()}: {observations[role]}" for role in ROLES),
    ]
    blocks.extend(
        f"--- Initial rationale {index} ({name}) ---\nPredicted answer: {answer or 'unparseable'}\n{text}"
        for index, (name, answer, text) in enumerate(reports, start=1)
    )
    return "\n\n".join(blocks)


def blind_prompt(reports: Sequence[tuple[str, str, str]]) -> str:
    blocks = [
        "Synthesize the concrete visual evidence in these independent rationales."
    ]
    blocks.extend(
        f"--- Rationale {index} ---\n{text}"
        for index, (_name, _answer, text) in enumerate(reports, start=1)
    )
    return "\n\n".join(blocks)


def metadata(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: row.get(key, "")
        for key in ("video_path", "split", "label", "dataset", "source")
        if key in row or key == "video_path"
    }


def validate_source_labels(
    manifest_row: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    names: Sequence[str],
) -> None:
    expected = gold_binary(manifest_row.get("label"))
    for name, row in zip(names, source_rows):
        raw_label = str(row.get("label", "")).strip()
        if raw_label and gold_binary(raw_label) != expected:
            raise ValueError(
                f"Label mismatch for {manifest_row.get('video_path')}: manifest={expected}, "
                f"{name}={raw_label}"
            )


def aggregate_observations(
    bundle: tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    source_names: Sequence[str],
    client,
) -> Dict[str, Any]:
    manifest_row, source_rows = bundle
    validate_source_labels(manifest_row, source_rows, source_names)
    label = gold_binary(manifest_row.get("label"))
    aggregated: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    usage: list[Dict[str, Any]] = []
    for role in ROLES:
        reports: list[tuple[str, str]] = []
        for name, row in zip(source_names, source_rows):
            text = observations_from_row(row).get(role, "")
            if text:
                reports.append((name, text))
        if not reports:
            raise ValueError(f"No {role} reports for {manifest_row.get('video_path')}")
        result = client.chat(
            [
                {"role": "system", "content": OBSERVATION_SYSTEM},
                {"role": "user", "content": observation_prompt(label, role, reports)},
            ]
        )
        aggregated[role] = sanitize_plain(result.text, f"aggregated {role} observation")
        counts[role] = len(reports)
        usage.append(
            {
                "name": role,
                "completion_tokens": result.completion_tokens,
                "elapsed_sec": result.elapsed_sec,
            }
        )
    return {
        **metadata(manifest_row),
        "observations": aggregated,
        **{f"{role}_response": aggregated[role] for role in ROLES},
        "num_models_used": counts,
        "usage": usage,
    }


def raw_explanations(
    source_names: Sequence[str], source_rows: Sequence[Mapping[str, Any]]
) -> list[tuple[str, str, str]]:
    reports: list[tuple[str, str, str]] = []
    for name, row in zip(source_names, source_rows):
        rationale = extract_rationale(row)
        if rationale:
            answer = parse_answer(str(row.get("response", "")))
            if answer == "Error":
                answer = parse_answer(str(row.get("answer", "")))
            reports.append((name, "" if answer == "Error" else answer, rationale))
    return reports


def aggregate_explanation(
    bundle: tuple[
        Mapping[str, Any], Sequence[Mapping[str, Any]], Mapping[str, Any] | None
    ],
    source_names: Sequence[str],
    all_wrong_policy: str,
    client,
) -> Dict[str, Any]:
    manifest_row, source_rows, observation_row = bundle
    validate_source_labels(manifest_row, source_rows, source_names)
    label = gold_binary(manifest_row.get("label"))
    reports = raw_explanations(source_names, source_rows)
    if not reports:
        raise ValueError(f"No rationales for {manifest_row.get('video_path')}")
    correct = [report for report in reports if report[1] == label]
    all_wrong = not correct
    if correct:
        system = EXPLANATION_SYSTEM
        prompt = explanation_prompt(label, correct)
        used = len(correct)
    elif all_wrong_policy == "drop":
        return {
            **metadata(manifest_row),
            "aggregated_explanation": "",
            "num_models_used": 0,
            "all_wrong": True,
            "dropped": True,
        }
    elif all_wrong_policy == "blind":
        system = BLIND_EXPLANATION_SYSTEM
        prompt = blind_prompt(reports)
        used = len(reports)
    else:
        if observation_row is None:
            raise ValueError(
                "--observations is required for --all-wrong-policy ground-truth"
            )
        observations = observations_from_row(observation_row)
        missing = [role for role in ROLES if not observations[role]]
        if missing:
            raise ValueError(
                f"Missing consolidated observations for {manifest_row.get('video_path')}: {missing}"
            )
        system = EXPLANATION_SYSTEM
        prompt = all_wrong_prompt(label, observations, reports)
        used = len(reports)
    result = client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    )
    explanation = sanitize_plain(result.text, "aggregated explanation")
    return {
        **metadata(manifest_row),
        "aggregated_explanation": explanation,
        "num_models_used": used,
        "all_wrong": all_wrong,
        "dropped": False,
        "usage": {
            "completion_tokens": result.completion_tokens,
            "elapsed_sec": result.elapsed_sec,
        },
    }


def batched(values: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def existing_prefix(path: Path) -> list[str]:
    if not path.exists():
        return []
    prefix: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Output is not valid JSONL at {path}:{line_number}; use --overwrite"
                ) from exc
            prefix.append(
                required(row.get("video_path"), f"output row {line_number} video_path")
            )
    if len(prefix) != len(set(prefix)):
        raise ValueError(f"Output contains duplicate video_path values: {path}")
    return prefix


def append_rows(
    output: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    flush_every: int,
) -> int:
    written = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written % max(1, flush_every) == 0:
                handle.flush()
                os.fsync(handle.fileno())
        handle.flush()
        os.fsync(handle.fileno())
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("observations", "explanations"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--input", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--observations",
        type=Path,
        default=None,
        help="Consolidated observations used only for all-wrong explanation samples.",
    )
    parser.add_argument(
        "--all-wrong-policy",
        choices=("ground-truth", "blind", "drop"),
        default="ground-truth",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    add_api_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if args.flush_every < 1:
        raise ValueError("--flush-every must be positive")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    sources = parse_sources(args.input)
    names = [name for name, _path in sources]
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".jsonl":
        raise ValueError("Aggregation output must use .jsonl for resumable streaming")
    if args.overwrite:
        output.unlink(missing_ok=True)
    repair_truncated_jsonl(output)
    prefix = existing_prefix(output)

    observation_path: Path | None = None
    if args.observations:
        observation_path = args.observations.expanduser().resolve()
        if not observation_path.is_file():
            raise FileNotFoundError(observation_path)
    if (
        args.kind == "explanations"
        and args.all_wrong_policy == "ground-truth"
        and observation_path is None
    ):
        raise ValueError(
            "--observations is required for ground-truth all-wrong aggregation"
        )

    paths = [manifest, *[path for _name, path in sources]]
    if args.kind == "explanations" and observation_path is not None:
        paths.append(observation_path)

    def pending_rows() -> Iterator[Any]:
        seen = 0
        for index, rows in enumerate(strict_zip_records(*paths)):
            if args.limit is not None and index >= args.limit:
                break
            video_path = required(
                rows[0].get("video_path"), f"manifest row {index} video_path"
            )
            seen += 1
            if index < len(prefix):
                if prefix[index] != video_path:
                    raise ValueError(
                        f"Existing output is not a manifest prefix at row {index}: "
                        f"{prefix[index]} != {video_path}; use --overwrite"
                    )
                continue
            source_rows = list(rows[1 : 1 + len(sources)])
            if args.kind == "observations":
                yield rows[0], source_rows
            else:
                observation_row = rows[-1] if observation_path is not None else None
                yield rows[0], source_rows, observation_row
        if len(prefix) > seen:
            raise ValueError(
                "Existing output is longer than the selected manifest input"
            )

    client = client_from_args(args)
    total_written = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for batch in batched(pending_rows(), max(1, args.concurrency * 2)):
            if args.kind == "observations":
                produced = pool.map(
                    lambda bundle: aggregate_observations(bundle, names, client), batch
                )
            else:
                produced = pool.map(
                    lambda bundle: aggregate_explanation(
                        bundle, names, args.all_wrong_policy, client
                    ),
                    batch,
                )
            total_written += append_rows(output, produced, flush_every=args.flush_every)
            print(
                f"aggregated {len(prefix) + total_written} rows "
                f"({total_written} in this run)",
                flush=True,
            )
    rows = iter_records(output) if output.exists() else ()
    write_records_atomic(output, rows)
    print(f"wrote {len(prefix) + total_written} rows to {output}")


if __name__ == "__main__":
    main()
