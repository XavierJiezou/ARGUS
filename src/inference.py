"""Unified OpenAI-compatible inference for every ARGUS reasoning mode.

This module intentionally keeps the API client, prompts, media loading, JSON/JSONL
I/O, resume logic, and the six supported inference workflows in one place.  The
public CLI is therefore one command with a small set of parameters instead of a
directory of near-identical scripts.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

import requests
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "FaceVid-Forensics-100K"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"

ROLES = ("texture", "lighting", "motion", "physics")
REAL_LABELS = {"real"}
FAKE_LABELS = {"fake", "efs", "fs", "fr"}
FRAME_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

ANSWER_RE = re.compile(r"<answer>\s*(real|fake)\s*</answer>", re.IGNORECASE)
EXPLANATION_RE = re.compile(
    r"<explanation>\s*(.*?)\s*</explanation>", re.IGNORECASE | re.DOTALL
)
OBSERVATION_RE = re.compile(
    r"<observation>\s*(.*?)\s*</observation>", re.IGNORECASE | re.DOTALL
)
VERDICT_RE = re.compile(r"\b(real|fake)\b", re.IGNORECASE)


def binary_label(value: Any) -> str:
    label = str(value if value is not None else "").strip().lower()
    if label in REAL_LABELS:
        return "real"
    if label in FAKE_LABELS:
        return "fake"
    raise ValueError(f"Unsupported label: {value!r}")


def observer_query(role: str) -> str:
    return (
        f"Observe the provided video from the perspective of {role} analysis. "
        "List the specific visual cues you noticed. "
        "Be concrete and precise - mention frame ranges, regions, or objects "
        "where you see potential issues or where everything appears normal. "
        "Put your report inside <observation></observation> tags. "
        "Do not make a final real/fake judgment and do not use <answer> tags. "
        "Keep your response concise (within 200 words)."
    )


OBSERVER_SPECS: Dict[str, Dict[str, str]] = {
    "texture": {
        "system": (
            "You are a texture and detail analysis expert in a video forensics team. "
            "Your task is to carefully observe the video and report any anomalies related to "
            "skin texture, edge sharpness, blending artifacts, material consistency, "
            "and fine-grained detail stability across frames. "
            "Focus ONLY on texture-related observations. "
            "Do NOT make a final real/fake judgment. "
            "Return only a concise report inside <observation></observation> tags."
        ),
        "query": observer_query("texture and detail"),
    },
    "lighting": {
        "system": (
            "You are a lighting analysis expert in a video forensics team. "
            "Your task is to carefully observe the video and report any anomalies related to "
            "light source direction, highlight consistency, shadow placement, specular reflections, "
            "and overall illumination coherence. "
            "Focus ONLY on lighting-related observations. "
            "Do NOT make a final real/fake judgment. "
            "Return only a concise report inside <observation></observation> tags."
        ),
        "query": observer_query("lighting"),
    },
    "motion": {
        "system": (
            "You are a motion analysis expert in a video forensics team. "
            "Your task is to carefully observe the video and report any anomalies related to "
            "inter-frame motion continuity, unnatural movements, temporal flickering, "
            "or physically implausible actions. "
            "Focus ONLY on motion-related observations. "
            "Do NOT make a final real/fake judgment. "
            "Return only a concise report inside <observation></observation> tags."
        ),
        "query": observer_query("motion"),
    },
    "physics": {
        "system": (
            "You are a physical plausibility analysis expert in a video forensics team. "
            "Your task is to carefully observe the video and report any anomalies related to "
            "hair dynamics, clothing behavior, occlusion ordering, perspective correctness, "
            "and geometric deformation. "
            "Focus ONLY on physics-related observations. "
            "Do NOT make a final real/fake judgment. "
            "Return only a concise report inside <observation></observation> tags."
        ),
        "query": observer_query("physical plausibility"),
    },
}

# Every prompt below is kept verbatim in the paper appendix; edit both together.
SINGLE_SYSTEM_PROMPT = """You are an expert video analyst.
Please think about the question as if you were a human pondering deeply. It's encouraged to include self-reflection or verification in the reasoning process. Put the explanation of your judgment within <explanation></explanation> tags. Finally, give the final verdict within <answer></answer> tags."""
SINGLE_USER_PROMPT = (
    "Is this video real or fake?"
    "\n\nThe following images are uniformly sampled frames from the video."
)
# The CoT baseline names the subject and the frame order, as its system prompt does.
COT_USER_PROMPT = (
    "Is this face video real or fake?"
    "\n\nThe following images are uniformly sampled frames from the video, in order."
)

COT_SYSTEM_PROMPT = """You are an expert face video forensics analyst. You are shown frames uniformly sampled from a video, in order. Determine whether the face video is Real or Fake based ONLY on the visible evidence in these frames.

Think step by step. Work through the following four analysis steps in order. In each step, report concrete, specific visual cues - mention frame ranges, facial regions, or objects where you see potential manipulation artifacts, or where everything appears consistent and natural. Do NOT give a real/fake verdict inside these four steps.

1. Texture and detail: skin texture, edge sharpness, blending artifacts around the face boundary, material consistency, and fine-grained detail stability across frames.
2. Lighting: light source direction, highlight consistency, shadow placement, specular reflections on skin and eyes, and overall illumination coherence between the face and the scene.
3. Motion: inter-frame motion continuity, unnatural movement, temporal flickering or jitter, and physically implausible actions across frames.
4. Physical plausibility: hair dynamics, clothing behavior, occlusion ordering, perspective correctness, facial geometry, and geometric deformation.

After the four steps, weigh the four observations together and reach a final decision. If clear manipulation artifacts appear in one or more steps, answer fake. If the face is texturally, temporally, and physically consistent with the scene, answer real.

Output your response strictly in the following format, with every tag present and non-empty:

<texture>your texture and detail observations</texture>
<lighting>your lighting observations</lighting>
<motion>your motion observations</motion>
<physics>your physical plausibility observations</physics>
<explanation>a brief rationale that synthesizes the four observations above into your decision</explanation>
<answer>real</answer>

Output requirements:
* All six tags are required and must not be empty.
* Each of <texture>, <lighting>, <motion>, <physics> must contain at least one concrete visual observation and must NOT contain a real/fake judgment.
* <explanation> must contain at least one sentence and must be grounded in the observations above.
* The answer must be exactly either <answer>real</answer> or <answer>fake</answer>, with no other text inside the answer tag.
* Base every observation only on the visible evidence in the frames. Do not use audio, speech, or metadata as evidence.
* Do not output anything outside these tags."""

MULTITURN_SYSTEM_PROMPT = (
    "You are an expert face video forensics analyst. Your task is to determine whether "
    "the face video is real or fake based only on the visible evidence in the provided frames."
)

MULTITURN_TURNS = {
    "texture": (
        "These are frames uniformly sampled from the video, in order. From the perspective of texture and "
        "detail, observe the video and report any anomalies related to skin texture, edge sharpness, "
        "blending artifacts around the face boundary, material consistency, and fine-grained detail "
        "stability across frames. Be concrete - mention frame ranges, facial regions, or objects. "
        "Put your report inside <observation></observation> tags. Do not make a real/fake judgment yet."
    ),
    "lighting": (
        "Now from the perspective of lighting, report any anomalies related to light source direction, "
        "highlight consistency, shadow placement, specular reflections on skin and eyes, and overall "
        "illumination coherence between the face and the scene. Be concrete - mention frame ranges, "
        "facial regions, or objects. Put your report inside <observation></observation> tags. "
        "Do not make a real/fake judgment yet."
    ),
    "motion": (
        "Now from the perspective of motion, report any anomalies related to inter-frame motion "
        "continuity, unnatural movements, temporal flickering or jitter, and physically implausible "
        "actions across frames. Be concrete - mention frame ranges, facial regions, or objects. "
        "Put your report inside <observation></observation> tags. Do not make a real/fake judgment yet."
    ),
    "physics": (
        "Now from the perspective of physical plausibility, report any anomalies related to hair "
        "dynamics, clothing behavior, occlusion ordering, perspective correctness, facial geometry, "
        "and geometric deformation. Be concrete - mention frame ranges, facial regions, or objects. "
        "Put your report inside <observation></observation> tags. Do not make a real/fake judgment yet."
    ),
}

MULTITURN_VERDICT = (
    "Based on your four observations above and the video frames, decide whether the face video is "
    "real or fake. First put a brief explanation that synthesizes your four observations within "
    "<explanation></explanation> tags; do not put any answer or label inside the explanation. "
    "Then output exactly either <answer>real</answer> or <answer>fake</answer>, with no other text "
    "inside the answer tag."
)

JUDGE_SYSTEM_WITHOUT_VIDEO = (
    "You are the final judge for binary video forgery classification. "
    "You cannot access any images or video. Use only the four expert analysis reports supplied as text. "
    "Put a brief explanation of your judgment within <explanation></explanation> tags; "
    "do not put any <answer> tags or final label inside the explanation. "
    "Then output exactly one tagged lowercase label: <answer>real</answer> or <answer>fake</answer>."
)
JUDGE_SYSTEM_WITH_VIDEO = (
    "You are the final judge for binary video forgery classification. "
    "You are given the video frames together with four expert analysis reports "
    "(texture, lighting, motion, physical plausibility) supplied as text. "
    "Weigh both the visual evidence and the four reports. "
    "Put a brief explanation of your judgment within <explanation></explanation> tags; "
    "do not put any <answer> tags or final label inside the explanation. "
    "Then output exactly one tagged lowercase label: <answer>real</answer> or <answer>fake</answer>."
)


def build_judge_input(observations: Mapping[str, str]) -> str:
    missing = [role for role in ROLES if not str(observations.get(role, "")).strip()]
    if missing:
        raise ValueError(f"Missing observations: {', '.join(missing)}")
    return (
        f"Texture analysis report:\n{observations['texture']}\n\n"
        f"Lighting analysis report:\n{observations['lighting']}\n\n"
        f"Motion analysis report:\n{observations['motion']}\n\n"
        f"Physical plausibility analysis report:\n{observations['physics']}\n\n"
        "Return a brief explanation in <explanation></explanation> tags. Do not put any <answer> "
        "tags or final label inside the explanation. Then output exactly <answer>real</answer> "
        "or <answer>fake</answer>."
    )


def tagged_text(text: str, tag: str) -> str:
    match = re.search(
        rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def parse_observation(text: str, *, allow_plain: bool = False) -> str:
    match = OBSERVATION_RE.search(text or "")
    if match and match.group(1).strip():
        return match.group(1).strip()
    plain = str(text or "").strip()
    if allow_plain and plain and not plain.lower().startswith("<error>"):
        return plain
    return ""


def parse_answer(text: str) -> str:
    strict = ANSWER_RE.search(text or "")
    if strict:
        return strict.group(1).lower()
    tokens = VERDICT_RE.findall(text or "")
    return tokens[-1].lower() if tokens else "Error"


def parse_final(text: str) -> tuple[str, str]:
    explanation = EXPLANATION_RE.search(text or "")
    return parse_answer(text), explanation.group(1).strip() if explanation else ""


def validated_final(text: str, description: str) -> tuple[str, str]:
    answer, explanation = parse_final(text)
    if answer not in {"real", "fake"}:
        raise ValueError(f"{description} is missing a real/fake answer")
    if not explanation:
        raise ValueError(f"{description} is missing a non-empty explanation")
    return answer, explanation


def extract_rationale(row: Mapping[str, Any]) -> str:
    for field in ("aggregated_explanation", "explanation", "rationale"):
        value = str(row.get(field, "") or "").strip()
        if value:
            wrapped = EXPLANATION_RE.search(value)
            return wrapped.group(1).strip() if wrapped else value
    response = str(row.get("response", "") or "")
    wrapped = EXPLANATION_RE.search(response)
    return wrapped.group(1).strip() if wrapped else ""


# ---------------------------------------------------------------------------
# JSON and JSONL
# ---------------------------------------------------------------------------


def iter_json_array(
    path: Path, chunk_size: int = 1024 * 1024
) -> Iterator[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    finished = False
    with path.open("r", encoding="utf-8") as handle:
        while not finished:
            chunk = handle.read(chunk_size)
            eof = chunk == ""
            buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError(f"Expected a JSON array: {path}")
                    buffer = buffer[1:]
                    started = True
                    continue
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                if buffer.startswith("]"):
                    if buffer[1:].strip():
                        raise ValueError(f"Unexpected content after JSON array: {path}")
                    finished = True
                    break
                if not buffer:
                    break
                try:
                    row, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"Invalid or truncated JSON array: {path}")
                    break
                if not isinstance(row, dict):
                    raise ValueError(f"Every record must be an object: {path}")
                yield row
                buffer = buffer[end:]
            if eof and not finished:
                raise ValueError(f"JSON array is missing its closing bracket: {path}")


def iter_records(path: Path) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        first = ""
        while True:
            char = handle.read(1)
            if char == "":
                return
            if not char.isspace():
                first = char
                break
    if first == "[":
        yield from iter_json_array(path)
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Every JSONL record must be an object: {path}:{line_number}"
                )
            yield row


def repair_truncated_jsonl(path: Path) -> bool:
    """Drop only an incomplete final JSONL line left by an interrupted write."""
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".jsonl":
        return False
    last_good_offset = 0
    truncate_at: int | None = None
    truncated_line = 0
    with path.open("rb") as handle:
        line_number = 0
        while True:
            line = handle.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                last_good_offset = handle.tell()
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if handle.read().strip():
                    raise ValueError(
                        f"Invalid JSONL before the end of {path}:{line_number}: {exc}"
                    ) from exc
                truncate_at = last_good_offset
                truncated_line = line_number
                break
            if not isinstance(row, dict):
                raise ValueError(
                    f"Every JSONL record must be an object: {path}:{line_number}"
                )
            last_good_offset = handle.tell()
    if truncate_at is None:
        return False
    with path.open("r+b") as handle:
        handle.truncate(truncate_at)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"repaired truncated final line {truncated_line} in {path}")
    return True


def read_records(path: Path) -> list[Dict[str, Any]]:
    return list(iter_records(path))


def _replace_with_retry(source: Path, destination: Path) -> None:
    for attempt in range(10):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2 + random.random() * 0.2)


def write_records_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        if path.suffix.lower() == ".jsonl":
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(list(rows), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _replace_with_retry(temporary, path)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _replace_with_retry(temporary, path)


def strict_zip_records(*paths: Path) -> Iterator[tuple[Dict[str, Any], ...]]:
    """Zip files by position and fail on length or video_path disagreement."""
    iterators = [iter_records(path) for path in paths]
    sentinel = object()
    for index, group in enumerate(zip_longest(*iterators, fillvalue=sentinel)):
        if any(row is sentinel for row in group):
            raise ValueError(f"Input files have different lengths near row {index}")
        rows = tuple(group)  # type: ignore[arg-type]
        video_paths = [str(row.get("video_path", "")).strip() for row in rows]
        if not video_paths[0] or len(set(video_paths)) != 1:
            raise ValueError(f"video_path mismatch at row {index}: {video_paths}")
        yield rows


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------


def uniform_indices(count: int, wanted: int) -> list[int]:
    if count <= 0:
        raise ValueError("The video contains no frames")
    if wanted <= 0:
        raise ValueError("--num-frames must be positive")
    if count <= wanted:
        return list(range(count))
    if wanted == 1:
        return [count // 2]
    return [round(index * (count - 1) / (wanted - 1)) for index in range(wanted)]


def _natural_key(path: Path) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def locate_media(dataset_root: Path, video_path: str) -> tuple[str, Path]:
    raw = Path(str(video_path))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        if raw.parts and raw.parts[0].lower() == "videos":
            relative = Path(*raw.parts[1:])
            candidates.extend((dataset_root / "frames" / relative, dataset_root / raw))
        candidates.extend((dataset_root / raw, dataset_root / "frames" / raw))
    for candidate in candidates:
        if candidate.is_dir():
            return "frames", candidate.resolve()
        if candidate.is_file():
            if candidate.suffix.lower() in FRAME_SUFFIXES:
                return "image", candidate.resolve()
            if candidate.suffix.lower() in VIDEO_SUFFIXES:
                return "video", candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"No frames or video found for {video_path}; checked: {rendered}"
    )


def resolve_single_media(path: Path) -> Path:
    """Validate a ``--video`` argument: one video file or one frame directory."""
    media = path.expanduser().resolve()
    if not media.exists():
        raise FileNotFoundError(media)
    if media.is_dir():
        frames = [
            entry
            for entry in media.iterdir()
            if entry.is_file() and entry.suffix.lower() in FRAME_SUFFIXES
        ]
        if not frames:
            raise ValueError(
                f"{media} holds no {'/'.join(sorted(FRAME_SUFFIXES))} frames; pass a "
                "video file or a directory of extracted frames"
            )
        return media
    suffix = media.suffix.lower()
    if suffix in VIDEO_SUFFIXES or suffix in FRAME_SUFFIXES:
        return media
    raise ValueError(
        f"Unsupported media type {suffix or '(none)'} for {media}; expected one of "
        f"{', '.join(sorted(VIDEO_SUFFIXES | FRAME_SUFFIXES))} or a frame directory"
    )


def sampled_frame_paths(
    dataset_root: Path, video_path: str, num_frames: int
) -> list[Path] | Path:
    kind, media = locate_media(dataset_root, video_path)
    if kind == "video":
        return media
    if kind == "image":
        return [media]
    frames = sorted(
        (
            path
            for path in media.iterdir()
            if path.is_file() and path.suffix.lower() in FRAME_SUFFIXES
        ),
        key=_natural_key,
    )
    return [
        frames[index].resolve() for index in uniform_indices(len(frames), num_frames)
    ]


def _resize(image: Image.Image, max_size: int) -> Image.Image:
    image = image.convert("RGB")
    if max_size > 0 and max(image.size) > max_size:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image.thumbnail((max_size, max_size), resampling)
    return image


def sample_images(
    dataset_root: Path, video_path: str, num_frames: int, max_size: int
) -> list[Image.Image]:
    selected = sampled_frame_paths(dataset_root, video_path, num_frames)
    if isinstance(selected, list):
        images: list[Image.Image] = []
        for path in selected:
            with Image.open(path) as image:
                images.append(_resize(image.copy(), max_size))
        return images
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise RuntimeError(
            "Reading video files requires decord. Run source create_env.sh, or "
            "provide pre-extracted frames under dataset_root/frames/."
        ) from exc
    reader = VideoReader(str(selected), ctx=cpu(0))
    arrays = reader.get_batch(uniform_indices(len(reader), num_frames)).asnumpy()
    return [_resize(Image.fromarray(array), max_size) for array in arrays]


def image_data_uri(image: Image.Image, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def frame_blocks(images: Sequence[Image.Image]) -> list[Dict[str, Any]]:
    blocks: list[Dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        blocks.append({"type": "text", "text": f"[frame {index}/{len(images)}]"})
        blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_uri(image), "detail": "high"},
            }
        )
    return blocks


def content_with_frames(
    prompt: str, blocks: Sequence[Mapping[str, Any]]
) -> list[Dict[str, Any]]:
    return [{"type": "text", "text": prompt}, *[dict(block) for block in blocks]]


# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatResult:
    text: str
    completion_tokens: int
    elapsed_sec: float


class OpenAIClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        api_key: str = "",
        timeout: float = 180.0,
        retries: int = 5,
        retry_backoff: float = 2.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        requests_per_minute: float = 0.0,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        base = base_url.rstrip("/")
        self.url = (
            base if base.endswith("/chat/completions") else base + "/chat/completions"
        )
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_interval = (
            60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        )
        self.extra_body = dict(extra_body or {})
        self._local = threading.local()
        self._rate_lock = threading.Lock()
        self._next_request_at = 0.0

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"Content-Type": "application/json"})
            if self.api_key:
                session.headers.update({"Authorization": f"Bearer {self.api_key}"})
            self._local.session = session
        return session

    def _wait_for_slot(self) -> None:
        if self.request_interval <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            scheduled = max(now, self._next_request_at)
            self._next_request_at = scheduled + self.request_interval
        delay = scheduled - now
        if delay > 0:
            time.sleep(delay)

    @staticmethod
    def _retry_after(response: requests.Response) -> float:
        value = response.headers.get("Retry-After", "").strip()
        try:
            return max(0.0, float(value))
        except ValueError:
            return 0.0

    @staticmethod
    def _content(data: Mapping[str, Any]) -> str:
        message = data["choices"][0]["message"]
        content = message.get("content", "")
        if isinstance(content, str):
            return content or str(message.get("reasoning_content", "") or "")
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, Mapping)
            )
        return str(content or message.get("reasoning_content", "") or "")

    def chat(self, messages: Sequence[Mapping[str, Any]]) -> ChatResult:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": list(messages),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **self.extra_body,
        }
        last_error = "unknown API error"
        for attempt in range(self.retries + 1):
            retry_after = 0.0
            self._wait_for_slot()
            started = time.perf_counter()
            try:
                response = self._session().post(
                    self.url, json=payload, timeout=self.timeout
                )
                elapsed = time.perf_counter() - started
                if response.status_code in {408, 409, 429, 500, 502, 503, 504}:
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                    retry_after = self._retry_after(response)
                elif response.status_code >= 400:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:1000]}"
                    )
                else:
                    data = response.json()
                    usage = data.get("usage") or {}
                    return ChatResult(
                        self._content(data),
                        int(usage.get("completion_tokens", 0) or 0),
                        round(elapsed, 3),
                    )
            except RuntimeError:
                raise
            except (
                requests.RequestException,
                ValueError,
                KeyError,
                TypeError,
                IndexError,
            ) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.retries:
                delay = max(
                    retry_after, min(30.0, self.retry_backoff * (2**attempt))
                )
                time.sleep(delay + random.random() * min(1.0, delay * 0.2))
        raise RuntimeError(last_error)


def add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model-name",
        "--model",
        dest="model_name",
        default=os.environ.get("MODEL_NAME", "Qwen2.5-VL-7B-Instruct"),
        help="Model name exposed by the endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument(
        "--requests-per-minute",
        type=float,
        default=0.0,
        help="Global request limit shared by all worker threads; 0 disables it.",
    )
    parser.add_argument(
        "--extra-body",
        default="{}",
        help="Optional JSON object merged into every chat-completions request.",
    )


def client_from_args(args: argparse.Namespace) -> OpenAIClient:
    if not str(args.base_url).strip() or not str(args.model_name).strip():
        raise ValueError("--base-url and --model-name must be non-empty")
    if args.max_tokens < 1 or args.temperature < 0:
        raise ValueError("--max-tokens must be positive and --temperature non-negative")
    if args.timeout <= 0 or args.retries < 0 or args.retry_backoff < 0:
        raise ValueError("timeout must be positive; retries and backoff must be non-negative")
    if args.requests_per_minute < 0:
        raise ValueError("--requests-per-minute must be non-negative")
    try:
        extra_body = json.loads(args.extra_body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--extra-body is not valid JSON: {exc}") from exc
    if not isinstance(extra_body, dict):
        raise ValueError("--extra-body must be a JSON object")
    return OpenAIClient(
        args.base_url,
        args.model_name,
        api_key=args.api_key,
        timeout=args.timeout,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        requests_per_minute=args.requests_per_minute,
        extra_body=extra_body,
    )


# ---------------------------------------------------------------------------
# Inference workflows
# ---------------------------------------------------------------------------


def observations_from_row(row: Mapping[str, Any]) -> Dict[str, str]:
    nested = row.get("observations")
    nested = nested if isinstance(nested, Mapping) else {}
    observations: Dict[str, str] = {}
    for role in ROLES:
        candidates = (
            nested.get(role),
            row.get(role),
            row.get(f"{role}_response"),
            row.get(f"{role}_report"),
        )
        value = ""
        for candidate in candidates:
            if candidate is None:
                continue
            value = parse_observation(str(candidate), allow_plain=True)
            if value:
                break
        observations[role] = value
    return observations


def _metadata(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: row.get(key, "")
        for key in ("video_path", "split", "label", "dataset", "source", "id")
        if key in row or key == "video_path"
    }


def _usage(calls: Sequence[Mapping[str, Any]], wall_sec: float) -> Dict[str, Any]:
    return {
        "calls": list(calls),
        "completion_tokens": sum(
            int(call.get("completion_tokens", 0)) for call in calls
        ),
        "elapsed_sec": round(wall_sec, 3),
    }


def _call(
    client: OpenAIClient,
    name: str,
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, Dict[str, Any]]:
    result = client.chat(messages)
    return result.text, {
        "name": name,
        "completion_tokens": result.completion_tokens,
        "elapsed_sec": result.elapsed_sec,
    }


def _result_record(
    row: Mapping[str, Any],
    *,
    mode: str,
    judge_video: bool,
    answer: str = "",
    explanation: str = "",
    observations: Mapping[str, str] | None = None,
    responses: Mapping[str, str] | None = None,
    calls: Sequence[Mapping[str, Any]] = (),
    wall_sec: float = 0.0,
    error: str = "",
) -> Dict[str, Any]:
    observations = dict(observations or {})
    responses = dict(responses or {})
    usage = _usage(calls, wall_sec)
    primary_response = responses.get("judge", responses.get("final", ""))
    if not primary_response and len(responses) == 1:
        primary_response = next(iter(responses.values()))
    record: Dict[str, Any] = {
        **_metadata(row),
        "mode": mode,
        "judge_video": judge_video,
        "answer": answer,
        "explanation": explanation,
        "observations": observations,
        "responses": responses,
        "response": primary_response,
        "completion_tokens": usage["completion_tokens"],
        "elapsed_sec": usage["elapsed_sec"],
        "usage": usage,
    }
    for role in ROLES:
        if role in observations:
            record[role] = observations[role]
        if role in responses:
            record[f"{role}_response"] = responses[role]
    if error:
        record["error"] = error
    return record


def infer_one(
    row: Mapping[str, Any],
    *,
    mode: str,
    role: str,
    judge_video: bool,
    dataset_root: Path,
    num_frames: int,
    frame_size: int,
    client: OpenAIClient,
) -> Dict[str, Any]:
    started = time.perf_counter()
    calls: list[Dict[str, Any]] = []
    responses: Dict[str, str] = {}
    observations: Dict[str, str] = {}
    try:
        needs_images = mode in {"single", "cot", "multiturn", "argus", "observer"}
        needs_images = needs_images or (mode == "judge" and judge_video)
        blocks: list[Dict[str, Any]] = []
        if needs_images:
            images = sample_images(
                dataset_root, str(row.get("video_path", "")), num_frames, frame_size
            )
            blocks = frame_blocks(images)

        if mode == "single":
            text, stat = _call(
                client,
                "single",
                [
                    {"role": "system", "content": SINGLE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": content_with_frames(SINGLE_USER_PROMPT, blocks),
                    },
                ],
            )
            calls.append(stat)
            responses["final"] = text
            answer, explanation = validated_final(text, "single response")

        elif mode == "cot":
            text, stat = _call(
                client,
                "cot",
                [
                    {"role": "system", "content": COT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": content_with_frames(COT_USER_PROMPT, blocks),
                    },
                ],
            )
            calls.append(stat)
            responses["final"] = text
            observations = {name: tagged_text(text, name) for name in ROLES}
            missing = [name for name, value in observations.items() if not value]
            if missing:
                raise ValueError(f"CoT response is missing tags: {', '.join(missing)}")
            answer, explanation = validated_final(text, "CoT response")

        elif mode == "observer":
            spec = OBSERVER_SPECS[role]
            text, stat = _call(
                client,
                role,
                [
                    {"role": "system", "content": spec["system"]},
                    {
                        "role": "user",
                        "content": content_with_frames(spec["query"], blocks),
                    },
                ],
            )
            calls.append(stat)
            responses[role] = text
            observations[role] = parse_observation(text)
            if not observations[role]:
                raise ValueError(f"{role} response is missing <observation> tags")
            answer, explanation = "", ""

        elif mode == "judge":
            observations = observations_from_row(row)
            judge_input = build_judge_input(observations)
            content: Any = judge_input
            if judge_video:
                content = content_with_frames(judge_input, blocks)
            text, stat = _call(
                client,
                "judge",
                [
                    {
                        "role": "system",
                        "content": JUDGE_SYSTEM_WITH_VIDEO
                        if judge_video
                        else JUDGE_SYSTEM_WITHOUT_VIDEO,
                    },
                    {"role": "user", "content": content},
                ],
            )
            calls.append(stat)
            responses["judge"] = text
            answer, explanation = validated_final(text, "Judge response")

        elif mode == "multiturn":
            messages: list[Dict[str, Any]] = [
                {"role": "system", "content": MULTITURN_SYSTEM_PROMPT}
            ]
            for index, name in enumerate(ROLES):
                prompt = MULTITURN_TURNS[name]
                content: Any = (
                    content_with_frames(prompt, blocks) if index == 0 else prompt
                )
                messages.append({"role": "user", "content": content})
                text, stat = _call(client, name, messages)
                calls.append(stat)
                messages.append({"role": "assistant", "content": text})
                responses[name] = text
                observations[name] = parse_observation(text)
                if not observations[name]:
                    raise ValueError(f"{name} response is missing <observation> tags")
            if judge_video:
                messages.append({"role": "user", "content": MULTITURN_VERDICT})
                text, stat = _call(client, "judge", messages)
            else:
                text, stat = _call(
                    client,
                    "judge",
                    [
                        {"role": "system", "content": JUDGE_SYSTEM_WITHOUT_VIDEO},
                        {"role": "user", "content": build_judge_input(observations)},
                    ],
                )
            calls.append(stat)
            responses["judge"] = text
            answer, explanation = validated_final(text, "multi-turn Judge response")

        elif mode == "argus":

            def run_observer(name: str) -> tuple[str, str, str, Dict[str, Any]]:
                spec = OBSERVER_SPECS[name]
                text, stat = _call(
                    client,
                    name,
                    [
                        {"role": "system", "content": spec["system"]},
                        {
                            "role": "user",
                            "content": content_with_frames(spec["query"], blocks),
                        },
                    ],
                )
                observation = parse_observation(text)
                if not observation:
                    raise ValueError(f"{name} response is missing <observation> tags")
                return name, text, observation, stat

            with ThreadPoolExecutor(max_workers=4) as pool:
                observer_results = list(pool.map(run_observer, ROLES))
            for name, text, observation, stat in observer_results:
                responses[name] = text
                observations[name] = observation
                calls.append(stat)
            judge_input = build_judge_input(observations)
            judge_content: Any = judge_input
            if judge_video:
                judge_content = content_with_frames(judge_input, blocks)
            text, stat = _call(
                client,
                "judge",
                [
                    {
                        "role": "system",
                        "content": JUDGE_SYSTEM_WITH_VIDEO
                        if judge_video
                        else JUDGE_SYSTEM_WITHOUT_VIDEO,
                    },
                    {"role": "user", "content": judge_content},
                ],
            )
            calls.append(stat)
            responses["judge"] = text
            answer, explanation = validated_final(text, "ARGUS Judge response")

        else:  # guarded by argparse and useful for direct API callers
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


def record_complete(record: Mapping[str, Any], mode: str, role: str) -> bool:
    if mode == "observer":
        return bool(observations_from_row(record).get(role))
    return str(record.get("answer", "")).lower() in {"real", "fake"}


def _result_key(row: Mapping[str, Any]) -> str:
    value = str(row.get("video_path", "")).strip()
    if not value:
        raise ValueError("Every input row requires a non-empty video_path")
    return value


def run_resumable(
    items: Sequence[Mapping[str, Any]],
    process_one,
    *,
    output: Path,
    mode: str,
    role: str,
    concurrency: int,
    flush_every: int,
    overwrite: bool,
) -> list[Dict[str, Any]]:
    partial = output.with_name(f".{output.name}.partial.jsonl")
    if overwrite:
        output.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)

    known: MutableMapping[str, Dict[str, Any]] = {}
    for source in (output, partial):
        if source.exists():
            repair_truncated_jsonl(source)
            for record in iter_records(source):
                known[_result_key(record)] = record

    pending = [
        item
        for item in items
        if not record_complete(known.get(_result_key(item), {}), mode, role)
    ]
    print(
        f"input={len(items)} complete={len(items) - len(pending)} pending={len(pending)}"
    )
    partial.parent.mkdir(parents=True, exist_ok=True)
    completed_now = 0

    def checkpoint_record(record: Dict[str, Any], checkpoint) -> None:
        nonlocal completed_now
        known[_result_key(record)] = record
        checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
        completed_now += 1
        if completed_now % max(1, flush_every) == 0:
            checkpoint.flush()
            os.fsync(checkpoint.fileno())
        if completed_now % 10 == 0 or completed_now == len(pending):
            print(f"processed {completed_now}/{len(pending)}", flush=True)

    with partial.open("a", encoding="utf-8", newline="\n") as checkpoint:
        if concurrency <= 1:
            for item in pending:
                checkpoint_record(process_one(item), checkpoint)
        else:
            batch_size = max(concurrency * 4, concurrency)
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for start in range(0, len(pending), batch_size):
                    batch = pending[start : start + batch_size]
                    futures = [pool.submit(process_one, item) for item in batch]
                    for future in as_completed(futures):
                        checkpoint_record(future.result(), checkpoint)
        checkpoint.flush()
        os.fsync(checkpoint.fileno())

    ordered = [known[_result_key(item)] for item in items if _result_key(item) in known]
    write_records_atomic(output, ordered)
    partial.unlink(missing_ok=True)
    return ordered


def shard_output_path(output: Path, num_shards: int, shard_index: int) -> Path:
    if num_shards <= 1:
        return output
    return (
        output.parent
        / "shards"
        / (f"{output.stem}.shard{shard_index:05d}-of-{num_shards:05d}{output.suffix}")
    )


def merge_shards(
    output: Path, num_shards: int, keys: Sequence[str]
) -> list[Dict[str, Any]] | None:
    """Combine finished shards into ``output``, or return None while some are missing.

    Every shard process calls this when it finishes, so whichever one happens to
    be last writes the merged file and no separate merge step is needed.
    """
    if num_shards <= 1:
        return None
    merged: Dict[str, Dict[str, Any]] = {}
    for shard_index in range(num_shards):
        path = shard_output_path(output, num_shards, shard_index)
        if not path.is_file():
            print(
                f"{path.name} is not finished yet, so {output.name} was not written; "
                f"rerun any shard once the others are done to merge them"
            )
            return None
        for record in iter_records(path):
            merged.setdefault(_result_key(record), record)
    rows = [merged[key] for key in keys if key in merged]
    write_records_atomic(output, rows)
    print(f"merged {num_shards} shards into {output}")
    return rows


def compute_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labelled: list[tuple[Mapping[str, Any], str]] = []
    for index, row in enumerate(rows):
        raw_label = str(row.get("label", "")).strip()
        if not raw_label:
            continue
        try:
            gold = binary_label(raw_label)
        except ValueError as exc:
            raise ValueError(f"Invalid label at prediction row {index}: {raw_label!r}") from exc
        labelled.append((row, gold))
    tp = fp = fn = tn = 0
    for row, gold in labelled:
        prediction = str(row.get("answer", "")).strip().lower()
        # An unparseable answer counts as wrong, so refusing to answer can never
        # improve a score.
        if gold == "fake":
            tp += int(prediction == "fake")
            fn += int(prediction != "fake")
        else:
            tn += int(prediction == "real")
            fp += int(prediction != "real")
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    present = [
        value
        for value, total in ((recall, tp + fn), (specificity, tn + fp))
        if total
    ]
    return {
        "total": len(labelled),
        "balanced_accuracy": sum(present) / len(present) if present else 0.0,
        "fake_recall": recall,
        "fake_f1": f1,
    }


def report_one(record: Mapping[str, Any]) -> None:
    """Print a single clip's reports and verdict to the terminal."""
    if record.get("error"):
        print(f"\nerror: {record['error']}")
        return
    print(f"\nvideo: {record.get('video_path', '')}")
    for role in ROLES:
        observation = str(record.get(role, "")).strip()
        if observation:
            print(f"\n[{role}]\n{observation}")
    explanation = str(record.get("explanation", "")).strip()
    if explanation:
        print(f"\n[explanation]\n{explanation}")
    answer = str(record.get("answer", "")).strip()
    if answer:
        print(f"\nanswer: {answer}")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "model"


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
    video = parser.add_mutually_exclusive_group()
    video.add_argument(
        "--with-video",
        "--judge-video",
        "--judge-sees-video",
        dest="judge_video",
        action="store_true",
        help="Let the final Judge see the sampled frames (default).",
    )
    video.add_argument(
        "--without-video",
        "--no-judge-video",
        "--no-judge-sees-video",
        dest="judge_video",
        action="store_false",
        help="Make the final Judge use only the four text reports.",
    )
    parser.set_defaults(judge_video=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    add_api_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode in {"single", "cot"} and not args.judge_video:
        raise ValueError(f"{args.mode} has no report-only Judge; use --with-video")
    if args.mode == "judge" and args.video:
        raise ValueError("judge mode requires --input rows containing four observations")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if args.num_frames < 1 or args.frame_size < 1:
        raise ValueError("--num-frames and --frame-size must be positive")
    if args.flush_every < 1:
        raise ValueError("--flush-every must be positive")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")

    if args.video:
        if args.num_shards > 1:
            raise ValueError("--num-shards applies to --input manifests, not --video")
        video = resolve_single_media(args.video)
        rows: list[Dict[str, Any]] = [{"video_path": str(video)}]
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

    tag = args.mode if args.mode != "observer" else f"observer-{args.role}"
    if args.mode in {"argus", "multiturn", "judge"}:
        tag += "-video" if args.judge_video else "-text"
    merged_output = (
        args.output
        or (DEFAULT_OUTPUT_ROOT / _slug(args.model_name) / f"{tag}-{input_stem}.jsonl")
    ).expanduser().resolve()
    output = shard_output_path(merged_output, args.num_shards, args.shard_index)
    dataset_root = args.dataset_root.expanduser().resolve()
    client = client_from_args(args)

    def process(row: Mapping[str, Any]) -> Dict[str, Any]:
        return infer_one(
            row,
            mode=args.mode,
            role=args.role,
            judge_video=args.judge_video,
            dataset_root=dataset_root,
            num_frames=args.num_frames,
            frame_size=args.frame_size,
            client=client,
        )

    results = run_resumable(
        rows,
        process,
        output=output,
        mode=args.mode,
        role=args.role,
        concurrency=args.concurrency,
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
    metrics_path = scored_output.with_name(f"{scored_output.stem}_metrics.json")
    write_json_atomic(metrics_path, metrics)
    if metrics["total"]:
        print(
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
            f"fake_recall={metrics['fake_recall']:.4f} "
            f"fake_f1={metrics['fake_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
