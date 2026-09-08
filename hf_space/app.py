"""Argus Video Forensics Gradio Space.

This app keeps the inference path intentionally small and explicit:

* a video is uniformly sampled into PIL frames;
 * four Observation adapters inspect the same frames;
* one selected Judge adapter receives the reports and optionally the frames;
* the custom API path uses the same prompt contract over OpenAI-compatible Chat
  Completions without ever persisting the request-scoped API key.

The heavyweight Transformers/PEFT imports are lazy.  This keeps API-only mode
usable on a CPU Space and makes importing this module safe for lightweight
tests.
"""

from __future__ import annotations

import base64
import io
import inspect
import os
import re
import subprocess
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import cv2
import gradio as gr
import requests
from PIL import Image

try:
    import spaces as _spaces  # type: ignore
except ImportError:  # local/API-only environments do not need ZeroGPU helpers
    _spaces = None


# ---------------------------------------------------------------------------
# Configuration and checkpoint mapping
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_REPO = "XavierJiezou/argus-models"
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SOURCE_ARGUS = "Argus LoRA"
SOURCE_API = "Custom API"
OBSERVER_ROLES: Tuple[str, ...] = ("texture", "lighting", "motion", "physics")
TRAINING_METHODS: Tuple[str, ...] = ("TRAINING_FREE", "SFT", "GRPO")
TRAINING_METHOD_LABELS: Mapping[str, str] = {
    "TRAINING_FREE": "Training-Free",
    "SFT": "+SFT",
    "GRPO": "+SFT+GRPO",
}
TRAINING_METHOD_CHOICES: Tuple[str, ...] = tuple(
    TRAINING_METHOD_LABELS[method] for method in TRAINING_METHODS
)

# Keep the bundled demo clips small and explicit.  The files are optional so
# that the app still builds if a user deploys only ``app.py`` and the model
# dependencies without copying the example directory.
BUNDLED_EXAMPLE_SPECS: Tuple[Tuple[str, str], ...] = (
    (
        "example/fake/7604593325026331931.mp4",
        "example/fake/7604593325026331931.jpg",
    ),
    (
        "example/real/726.mp4",
        "example/real/726.jpg",
    ),
)

OBSERVER_LORA_SUBFOLDERS: Mapping[str, str] = {
    "lighting": "qwen2_5_vl_7b/main/shared/lora/lighting",
    "motion": "qwen2_5_vl_7b/main/shared/lora/motion",
    "physics": "qwen2_5_vl_7b/main/shared/lora/physics",
    "texture": "qwen2_5_vl_7b/main/shared/lora/texture",
}

JUDGE_LORA_SUBFOLDERS: Mapping[Tuple[str, bool], str] = {
    ("SFT", False): "qwen2_5_vl_7b/main/sft_text/lora/judge",
    ("SFT", True): "qwen2_5_vl_7b/main/sft_video/lora/judge",
    ("GRPO", False): "qwen2_5_vl_7b/main/grpo_text/lora/judge",
    ("GRPO", True): "qwen2_5_vl_7b/main/grpo_video/lora/judge",
}

DEFAULT_NUM_FRAMES = 16
# The trained Observer/Judge recipes use 16 uniformly sampled frames.  Keep
# that budget identical on regular GPU and ZeroGPU runtimes: changing it would
# alter the visual evidence seen by the adapters and can change the verdict.
DEFAULT_MAX_FRAME_SIDE = 256
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_OBSERVER_MAX_NEW_TOKENS = 256
DEFAULT_JUDGE_MAX_NEW_TOKENS = 512
DEFAULT_API_TIMEOUT_SECONDS = 180


def bundled_example_assets() -> Tuple[List[str], List[str]]:
    """Return thumbnail paths and corresponding video paths for bundled demos.

    Example clips are deliberately resolved relative to ``app.py`` rather than
    the process working directory.  Missing files are skipped so API-only or
    code-only deployments remain usable without the optional example assets.
    """

    cards: List[str] = []
    video_paths: List[str] = []
    for video_relative_path, thumbnail_relative_path in BUNDLED_EXAMPLE_SPECS:
        video_path = APP_DIR / video_relative_path
        thumbnail_path = APP_DIR / thumbnail_relative_path
        if video_path.is_file() and thumbnail_path.is_file():
            cards.append(str(thumbnail_path))
            video_paths.append(str(video_path))
    return cards, video_paths


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _default_num_frames() -> int:
    """Return the fixed frame budget used by the trained inference recipe."""

    return DEFAULT_NUM_FRAMES


def normalize_training_method(method: str) -> str:
    """Return the canonical training-method key for UI or API inputs."""

    normalized = str(method or "").strip().upper().replace(" ", "_")
    aliases = {
        "TRAINING_FREE": "TRAINING_FREE",
        "TRAINING-FREE": "TRAINING_FREE",
        "+SFT": "SFT",
        "SFT": "SFT",
        "+SFT+GRPO": "GRPO",
        "GRPO": "GRPO",
    }
    canonical = aliases.get(normalized)
    if canonical not in TRAINING_METHODS:
        raise ValueError("训练方法必须是 Training-Free、+SFT 或 +SFT+GRPO。")
    return canonical


def normalize_bool(value: Any) -> bool:
    """Normalize Gradio booleans and string-like test inputs consistently."""

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _local_adapter_candidate(root: Optional[str], subfolder: str) -> Optional[Path]:
    """Resolve an optional local adapter root used for offline/local checks.

    The Space itself normally uses ``MODEL_REPO``.  Setting ``ARGUS_LORA_ROOT``
    lets a local checkout point at ``weights/qwen2_5_vl_7b`` (or at the parent
    ``weights`` directory) without changing the deployed code.
    """

    if not root:
        return None
    base = Path(root).expanduser()
    candidates = (base / subfolder, base / "qwen2_5_vl_7b" / subfolder)
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "adapter_config.json").exists():
            return candidate.resolve()
    return None


@dataclass(frozen=True)
class AdapterSource:
    repo_id: str
    subfolder: str
    local_path: Optional[Path] = None

    @property
    def display_path(self) -> str:
        if self.local_path:
            return f"local:{self.local_path}"
        return f"{self.repo_id}/{self.subfolder}"


def resolve_adapter_source(subfolder: str, local_root: Optional[str] = None) -> AdapterSource:
    """Return a remote HF source, or an explicit local override if configured."""

    local_path = _local_adapter_candidate(local_root or os.getenv("ARGUS_LORA_ROOT"), subfolder)
    return AdapterSource(repo_id=MODEL_REPO, subfolder=subfolder, local_path=local_path)


def judge_adapter_name(training_method: str, judge_reads_video: bool) -> str:
    method = normalize_training_method(training_method)
    if method == "TRAINING_FREE":
        return "base_model"
    return f"judge_{method.lower()}_{'video' if normalize_bool(judge_reads_video) else 'text'}"


def select_judge_source(
    training_method: str,
    judge_reads_video: bool,
    local_root: Optional[str] = None,
) -> AdapterSource:
    method = normalize_training_method(training_method)
    if method == "TRAINING_FREE":
        raise ValueError("Training-Free 不使用 Judge LoRA。")
    try:
        subfolder = JUDGE_LORA_SUBFOLDERS[(method, normalize_bool(judge_reads_video))]
    except KeyError as exc:  # defensive guard for future mapping edits
        raise ValueError("找不到对应的 Judge checkpoint 映射。") from exc
    return resolve_adapter_source(subfolder, local_root=local_root)


def selected_checkpoint_display(
    model_source: str,
    training_method: str,
    judge_reads_video: bool,
) -> str:
    """Text shown in the UI for the currently selected Judge."""

    if model_source == SOURCE_API:
        return "Custom API: Judge uses specified Model Name (no local checkpoint)"
    try:
        if normalize_training_method(training_method) == "TRAINING_FREE":
            return f"{BASE_MODEL_ID} (Training-Free; no LoRA)"
        return select_judge_source(training_method, normalize_bool(judge_reads_video)).display_path
    except ValueError as exc:
        return f"Configuration error: {exc}"


# ---------------------------------------------------------------------------
# Video sampling
# ---------------------------------------------------------------------------


@dataclass
class VideoSample:
    frames: List[Image.Image]
    frame_indices: List[int]
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 and self.frame_count > 0 else 0.0

    @property
    def summary(self) -> str:
        duration = f"{self.duration_seconds:.1f}s" if self.duration_seconds else "unknown duration"
        return f"{len(self.frames)} frames sampled · {self.width}×{self.height} · {duration}"


def normalize_video_path(video: Any) -> str:
    """Accept Gradio's filepath value and a few common local-test shapes."""

    if video is None or video == "":
        raise ValueError("请先上传一个视频文件。")
    if isinstance(video, (str, os.PathLike)):
        path = os.fspath(video)
    elif isinstance(video, Mapping):
        path = video.get("path") or video.get("name") or video.get("data")
    else:
        path = getattr(video, "path", None) or getattr(video, "name", None)
    if not path or not isinstance(path, (str, os.PathLike)):
        raise ValueError("无法读取上传的视频路径，请重新上传。")
    path = os.fspath(path)
    if not Path(path).is_file():
        raise ValueError("上传的视频文件不存在或已失效，请重新上传。")
    if Path(path).stat().st_size <= 0:
        raise ValueError("上传的视频为空文件。")
    return path


def _resize_frame(rgb_frame: Any, max_side: int) -> Image.Image:
    image = Image.fromarray(rgb_frame).convert("RGB")
    if max(image.size) > max_side:
        ratio = max_side / float(max(image.size))
        image = image.resize(
            (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
            Image.Resampling.LANCZOS,
        )
    return image


def _uniform_indices(frame_count: int, num_frames: int) -> List[int]:
    count = max(1, int(frame_count))
    target = max(1, min(int(num_frames), count))
    if target == 1:
        return [0]
    return sorted({round(i * (count - 1) / (target - 1)) for i in range(target)})


def sample_video_frames(
    video_path: Any,
    num_frames: Optional[int] = None,
    max_side: Optional[int] = None,
) -> VideoSample:
    """Uniformly sample a video into RGB PIL frames.

    Seeking is used when the container reports a frame count.  A bounded
    sequential fallback handles formats where OpenCV cannot seek reliably.
    """

    path = normalize_video_path(video_path)
    target_frames = max(1, num_frames or _default_num_frames())
    target_side = max(64, max_side or _env_int("ARGUS_MAX_FRAME_SIDE", DEFAULT_MAX_FRAME_SIDE))
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError("无法解码该视频。请上传常见的 MP4/H.264 文件。")

    reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: List[Image.Image] = []
    indices: List[int] = []

    try:
        if reported_count > 0:
            wanted = _uniform_indices(reported_count, target_frames)
            for index in wanted:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(_resize_frame(rgb, target_side))
                indices.append(index)

        # Some codecs report a count but fail random seeks.  Re-read once in a
        # bounded pass so a valid video is not rejected unnecessarily.
        if not frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            sequential: List[Tuple[int, Any]] = []
            max_scan = max(256, target_frames * 64)
            index = 0
            while index < max_scan:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                sequential.append((index, frame))
                index += 1
            if sequential:
                selected = _uniform_indices(len(sequential), target_frames)
                for position in selected:
                    original_index, frame = sequential[position]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(_resize_frame(rgb, target_side))
                    indices.append(original_index)
                if reported_count <= 0:
                    reported_count = len(sequential)

        if not frames:
            raise ValueError("视频可以打开但没有可读帧，请检查编码格式。")
        if reported_count <= 0:
            reported_count = max(indices) + 1 if indices else len(frames)
        if width <= 0 or height <= 0:
            width, height = frames[0].size
    finally:
        capture.release()

    return VideoSample(
        frames=frames,
        frame_indices=indices,
        fps=fps,
        frame_count=reported_count,
        width=width,
        height=height,
    )


# Pixel formats browsers decode in a <video> element.  H.264 in 4:4:4, 4:2:2,
# or 10-bit is rejected by Chrome and Firefox even though the codec name and
# container both look correct, which Gradio's own codec allow-list misses.
BROWSER_SAFE_PIX_FMTS: Tuple[str, ...] = ("yuv420p", "yuvj420p", "yuv420p10le")


def _browser_playable(path: str) -> bool:
    """Report whether a browser ``<video>`` element can decode this file.

    Builds on Gradio's container/codec allow-list so the check stays in sync
    with what the frontend player accepts, then additionally rejects pixel
    formats browsers cannot decode.  A probe failure is treated as playable so
    an unusual-but-working file is not transcoded needlessly, matching Gradio's
    behaviour.
    """

    try:
        from gradio import processing_utils

        if not processing_utils.video_is_playable(path):
            return False
    except Exception:
        return True

    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=pix_fmt",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    pix_fmt = (completed.stdout or "").strip().splitlines()
    if completed.returncode != 0 or not pix_fmt:
        return True
    return pix_fmt[0].strip() in BROWSER_SAFE_PIX_FMTS


def transcode_for_preview(video: Any) -> Any:
    """Return a browser-playable copy of an uploaded video when needed.

    ``gr.Video`` only converts videos on the *output* path.  An uploaded file
    is handed to the frontend player untouched, so a decodable-but-not-
    web-playable upload (HEVC/H.265, MPEG-4 Part 2, or an H.264 stream in a
    ``.mov``/``.avi`` container) makes the ``<video>`` element fail and Gradio
    surfaces a bare "Video not playable" error.

    Re-encode those uploads to H.264/yuv420p in an MP4 container.  The original
    file is returned unchanged when it is already playable, when ffmpeg is
    unavailable, or when the conversion fails, so this never turns a working
    upload into a failed one.
    """

    if video is None or video == "":
        return video
    try:
        path = normalize_video_path(video)
    except ValueError:
        # Let the analysis path report the specific problem with its own message.
        return video

    if _browser_playable(path):
        return video

    from gradio import processing_utils

    if not processing_utils.ffmpeg_installed():
        return video

    source = Path(path)
    target = source.with_name(f"{source.stem}_web.mp4")
    if target.resolve() == source.resolve():
        target = source.with_name(f"{source.stem}_web_playable.mp4")
    if target.is_file() and target.stat().st_size > 0:
        return str(target)

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        # H.264 requires even dimensions; pad rather than crop odd-sized input.
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        str(target),
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_env_int("ARGUS_TRANSCODE_TIMEOUT", 180),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return video
    if completed.returncode != 0 or not target.is_file() or target.stat().st_size <= 0:
        return video
    return str(target)


def frame_to_data_url(frame: Image.Image, quality: int = 84) -> str:
    """Encode one sampled frame for an OpenAI-compatible image_url part."""

    buffer = io.BytesIO()
    frame.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# ---------------------------------------------------------------------------
# Prompt contract shared by both inference backends
# ---------------------------------------------------------------------------


OBSERVER_SYSTEM_PROMPTS: Mapping[str, str] = {
    "texture": (
        "You are a texture and detail analysis expert in a video forensics team. "
        "Your task is to carefully observe the video and report any anomalies related to "
        "skin texture, edge sharpness, blending artifacts, material consistency, "
        "and fine-grained detail stability across frames. "
        "Focus ONLY on texture-related observations. "
        "Do NOT make a final real/fake judgment. "
        "Return only a concise report inside <observation></observation> tags."
    ),
    "lighting": (
        "You are a lighting analysis expert in a video forensics team. "
        "Your task is to carefully observe the video and report any anomalies related to "
        "light source direction, highlight consistency, shadow placement, specular reflections, "
        "and overall illumination coherence. "
        "Focus ONLY on lighting-related observations. "
        "Do NOT make a final real/fake judgment. "
        "Return only a concise report inside <observation></observation> tags."
    ),
    "motion": (
        "You are a motion analysis expert in a video forensics team. "
        "Your task is to carefully observe the video and report any anomalies related to "
        "inter-frame motion continuity, unnatural movements, temporal flickering, "
        "or physically implausible actions. "
        "Focus ONLY on motion-related observations. "
        "Do NOT make a final real/fake judgment. "
        "Return only a concise report inside <observation></observation> tags."
    ),
    "physics": (
        "You are a physical plausibility analysis expert in a video forensics team. "
        "Your task is to carefully observe the video and report any anomalies related to "
        "hair dynamics, clothing behavior, occlusion ordering, perspective correctness, "
        "and geometric deformation. "
        "Focus ONLY on physics-related observations. "
        "Do NOT make a final real/fake judgment. "
        "Return only a concise report inside <observation></observation> tags."
    ),
}

OBSERVER_QUERY_ROLES: Mapping[str, str] = {
    "texture": "texture and detail",
    "lighting": "lighting",
    "motion": "motion",
    "physics": "physical plausibility",
}


def observer_system_prompt(role: str) -> str:
    role = role.lower().strip()
    if role not in OBSERVER_SYSTEM_PROMPTS:
        raise ValueError(f"未知 Observation：{role}")
    return OBSERVER_SYSTEM_PROMPTS[role]


def observer_user_prompt(role: str, sample: VideoSample) -> str:
    role = role.lower().strip()
    if role not in OBSERVER_QUERY_ROLES:
        raise ValueError(f"Unknown Observation role: {role}")
    return (
        "<video>\n"
        f"Observe the provided video from the perspective of {OBSERVER_QUERY_ROLES[role]} analysis. "
        "List the specific visual cues you noticed. "
        "Be concrete and precise - mention frame ranges, regions, or objects "
        "where you see potential issues or where everything appears normal. "
        "Put your report inside <observation></observation> tags. "
        "Do not make a final real/fake judgment and do not use <answer> tags. "
        "Keep your response concise (within 200 words)."
    )


# Copied from the Judge dataset builder used to train the shipped adapters.
JUDGE_SYSTEM_TEXT = (
    "You are the final judge for binary video forgery classification. "
    "You cannot access any images or video. Use only the four expert analysis reports supplied as text. "
    "Put a brief explanation of your judgment within <explanation></explanation> tags; "
    "do not put any <answer> tags or final label inside the explanation. "
    "Then output exactly one tagged lowercase label: <answer>real</answer> or <answer>fake</answer>."
)

JUDGE_SYSTEM_VIDEO = (
    "You are the final judge for binary video forgery classification. "
    "You are given the video frames together with four expert analysis reports "
    "(texture, lighting, motion, physical plausibility) supplied as text. "
    "Weigh both the visual evidence and the four reports. "
    "Put a brief explanation of your judgment within <explanation></explanation> tags; "
    "do not put any <answer> tags or final label inside the explanation. "
    "Then output exactly one tagged lowercase label: <answer>real</answer> or <answer>fake</answer>."
)


def _clip_text(value: Any, limit: int = 6000) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[…report truncated for Judge context…]"


def format_reports_for_judge(reports: Mapping[str, str]) -> str:
    """Return the exact four-report prompt used to train the Judge adapters."""

    return (
        "Texture analysis report:\n"
        f"{_clip_text(reports.get('texture', ''))}\n\n"
        "Lighting analysis report:\n"
        f"{_clip_text(reports.get('lighting', ''))}\n\n"
        "Motion analysis report:\n"
        f"{_clip_text(reports.get('motion', ''))}\n\n"
        "Physical plausibility analysis report:\n"
        f"{_clip_text(reports.get('physics', ''))}\n\n"
        "Return a brief explanation in <explanation></explanation> tags. Do not put any <answer> "
        "tags or final label inside the explanation. Then output exactly <answer>real</answer> "
        "or <answer>fake</answer>."
    )


def judge_user_prompt(reports: Mapping[str, str], sample: VideoSample, include_video: bool) -> str:
    include_video = normalize_bool(include_video)
    return ("<video>\n" if include_video else "") + format_reports_for_judge(reports)


_OBSERVATION_TAG_RE = re.compile(
    r"<observation>\s*(.*?)\s*</observation>", re.IGNORECASE | re.DOTALL
)
_EXPLANATION_TAG_RE = re.compile(
    r"<explanation>\s*(.*?)\s*</explanation>", re.IGNORECASE | re.DOTALL
)
_ANSWER_TAG_RE = re.compile(
    r"<answer>\s*(real|fake)\s*</answer>", re.IGNORECASE
)


def normalize_observer_output(value: Any) -> str:
    """Strip the protocol wrapper used by the trained Observer adapters."""

    raw = str(value or "").strip()
    if not raw:
        return raw
    match = _OBSERVATION_TAG_RE.search(raw)
    return (match.group(1) if match else raw).strip()


def normalize_judge_output(value: Any) -> str:
    """Render the trained Judge XML response cleanly in the Gradio panel."""

    raw = str(value or "").strip()
    if not raw:
        return raw
    explanation_match = _EXPLANATION_TAG_RE.search(raw)
    answer_match = _ANSWER_TAG_RE.search(raw)
    if explanation_match or answer_match:
        explanation = (explanation_match.group(1) if explanation_match else "").strip()
        answer = (answer_match.group(1) if answer_match else "").upper()
        pieces: List[str] = []
        if explanation:
            pieces.append(explanation)
        if answer:
            pieces.append(f"**Verdict: {answer}**")
        return "\n\n".join(pieces).strip()
    # Preserve provider/checkpoint output when it does not follow the trained
    # XML contract; do not rewrite it into a different inference protocol.
    return raw


def _qwen_messages(system_prompt: str, user_prompt: str, frames: Optional[Sequence[Image.Image]]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    prompt_text = str(user_prompt)
    if frames:
        # ms-swift replaces the leading <video> placeholder with one video
        # media item. Recreate that contract with the Transformers structured
        # message format and the same requested 256x256 frame size.
        if prompt_text.startswith("<video>\n"):
            prompt_text = prompt_text[len("<video>\n") :]
        content.append(
            {
                "type": "video",
                "video": list(frames),
                "resized_height": 256,
                "resized_width": 256,
            }
        )
    content.append({"type": "text", "text": prompt_text})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


# ---------------------------------------------------------------------------
# Built-in Qwen2.5-VL + PEFT backend
# ---------------------------------------------------------------------------


_TORCH: Any = None


def _import_torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised on minimal CPU envs
            raise RuntimeError("内置模式缺少 PyTorch，请使用带 GPU 的 Hugging Face Space。") from exc
        _TORCH = torch
    return _TORCH


def runtime_status() -> str:
    """Short, user-facing runtime hint; does not load any model."""

    try:
        torch = _import_torch()
        if not torch.cuda.is_available():
            return "当前运行时未检测到 CUDA GPU。内置 Argus LoRA 需要 GPU；Custom API 模式仍可用。"
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        return f"CUDA 可用：{name} · 可用显存约 {free / 2**30:.1f}/{total / 2**30:.1f} GB"
    except Exception:
        return "GPU 状态暂不可用；Custom API 模式无需 GPU。"


class ArgusModelManager:
    """One lazily-created base model with named Observation/Judge adapters."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.model: Any = None
        self.processor: Any = None
        self.compute_dtype: Any = None
        self.loaded_adapter_names: set[str] = set()
        self.loaded_sources: Dict[str, AdapterSource] = {}
        self.load_summary = ""

    def _clear_model_state(self) -> None:
        """Make a failed partial adapter load safe to retry."""

        self.model = None
        self.processor = None
        self.compute_dtype = None
        self.loaded_adapter_names.clear()
        self.loaded_sources.clear()
        self.load_summary = ""

    def _hf_token(self) -> Optional[str]:
        # HF Spaces exposes secrets as environment variables.  Never hard-code
        # or print the token; passing None is valid for public repositories.
        return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    def _choose_quantization(self, torch: Any) -> Tuple[bool, Any, float]:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "内置 Argus LoRA 需要 CUDA GPU。当前没有 GPU，请切换到 Custom API 模式，"
                "或在 Space 设置中选择带 GPU 的硬件。"
            )
        bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        dtype = torch.bfloat16 if bf16_supported else torch.float16
        force = os.getenv("ARGUS_FORCE_4BIT", "").strip().lower()
        if force in {"1", "true", "yes", "on"}:
            use_4bit = True
        elif force in {"0", "false", "no", "off"}:
            use_4bit = False
        else:
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                use_4bit = free_bytes < _env_float("ARGUS_BF16_MIN_FREE_GB", 20.0) * 2**30
            except Exception:
                use_4bit = False
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            free_gb = free_bytes / 2**30
        except Exception:
            free_gb = 0.0
        return use_4bit, dtype, free_gb

    def _processor(self, processor_cls: Any, token: Optional[str]) -> Any:
        kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if token:
            kwargs["token"] = token
        # Keeping visual tokens bounded makes a 24 GB card practical.
        kwargs.update(
            {
                "min_pixels": 256 * 28 * 28,
                "max_pixels": 768 * 28 * 28,
            }
        )
        try:
            return processor_cls.from_pretrained(BASE_MODEL_ID, **kwargs)
        except TypeError:
            kwargs.pop("min_pixels", None)
            kwargs.pop("max_pixels", None)
            return processor_cls.from_pretrained(BASE_MODEL_ID, **kwargs)

    def _download_adapter_locally(self, source: AdapterSource) -> Path:
        from huggingface_hub import snapshot_download  # type: ignore

        token = self._hf_token()
        kwargs: Dict[str, Any] = {
            "repo_id": source.repo_id,
            "allow_patterns": [f"{source.subfolder}/*"],
        }
        if token:
            kwargs["token"] = token
        root = snapshot_download(**kwargs)
        path = Path(root) / source.subfolder
        if not (path / "adapter_config.json").exists():
            raise RuntimeError(f"HF 仓库中找不到 adapter_config.json：{source.display_path}")
        return path

    def _load_first_adapter(self, base_model: Any, peft_cls: Any, name: str, source: AdapterSource) -> Any:
        if source.local_path:
            return peft_cls.from_pretrained(base_model, str(source.local_path), adapter_name=name)
        token = self._hf_token()
        kwargs: Dict[str, Any] = {"adapter_name": name, "subfolder": source.subfolder}
        if token:
            kwargs["token"] = token
        try:
            return peft_cls.from_pretrained(base_model, source.repo_id, **kwargs)
        except (TypeError, ValueError, OSError):
            local_path = self._download_adapter_locally(source)
            return peft_cls.from_pretrained(base_model, str(local_path), adapter_name=name)

    def _load_named_adapter(self, name: str, source: AdapterSource) -> None:
        if name in self.loaded_adapter_names:
            return
        if self.model is None:
            raise RuntimeError("基础模型尚未加载。")
        if source.local_path:
            self.model.load_adapter(str(source.local_path), adapter_name=name)
        else:
            token = self._hf_token()
            kwargs: Dict[str, Any] = {"adapter_name": name, "subfolder": source.subfolder}
            if token:
                kwargs["token"] = token
            try:
                self.model.load_adapter(source.repo_id, **kwargs)
            except (TypeError, ValueError, OSError):
                local_path = self._download_adapter_locally(source)
                self.model.load_adapter(str(local_path), adapter_name=name)
        self.loaded_adapter_names.add(name)
        self.loaded_sources[name] = source

    def _ensure_observer_adapters(self, peft_cls: Any = None) -> None:
        """Wrap the base model with the shared Observation adapters on demand."""

        if self.model is None:
            raise RuntimeError("基础模型尚未加载。")
        if set(OBSERVER_ROLES).issubset(self.loaded_adapter_names):
            return
        if peft_cls is None:
            from peft import PeftModel as peft_cls  # type: ignore

        if not self.loaded_adapter_names:
            first_role = OBSERVER_ROLES[0]
            first_source = resolve_adapter_source(OBSERVER_LORA_SUBFOLDERS[first_role])
            self.model = self._load_first_adapter(self.model, peft_cls, first_role, first_source)
            self.loaded_adapter_names.add(first_role)
            self.loaded_sources[first_role] = first_source

        for role in OBSERVER_ROLES:
            if role not in self.loaded_adapter_names:
                self._load_named_adapter(role, resolve_adapter_source(OBSERVER_LORA_SUBFOLDERS[role]))

    def _load_base_and_observers(self, load_observers: bool = True) -> None:
        torch = _import_torch()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "内置 Argus LoRA 需要 CUDA GPU。当前没有 GPU，请切换到 Custom API 模式，"
                "或在 Space 设置中选择带 GPU 的硬件。"
            )
        try:
            from peft import PeftModel  # type: ignore
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration  # type: ignore
        except ImportError as exc:
            raise RuntimeError("内置模式依赖 transformers、peft、accelerate 和 bitsandbytes，请检查 requirements.txt。") from exc

        use_4bit, dtype, free_gb = self._choose_quantization(torch)
        token = self._hf_token()
        processor = self._processor(AutoProcessor, token)
        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        if token:
            model_kwargs["token"] = token
        if use_4bit:
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                )
            except Exception as exc:
                raise RuntimeError(
                    "当前显存建议使用 4-bit，但 bitsandbytes 不可用。请安装 requirements.txt，"
                    "或改用 24GB 以上 GPU。"
                ) from exc
            model_kwargs["quantization_config"] = quant_config

        try:
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_MODEL_ID, **model_kwargs)
            self.model = base_model
            self.processor = processor
            self.compute_dtype = dtype
            if load_observers:
                self._ensure_observer_adapters(PeftModel)
            self.model.eval()
            gpu_name = torch.cuda.get_device_name(0)
            mode = "4-bit" if use_4bit else "bf16/fp16"
            self.load_summary = f"已加载 Qwen2.5-VL-7B · {mode} · {gpu_name} · 可用显存约 {free_gb:.1f} GB"
        except torch.cuda.OutOfMemoryError as exc:
            self._clear_model_state()
            torch.cuda.empty_cache()
            raise RuntimeError(
                "加载 Argus LoRA 时显存不足。建议使用 24GB 以上 GPU，或设置 ARGUS_FORCE_4BIT=1。"
            ) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self._clear_model_state()
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "加载 Argus LoRA 时显存不足。建议使用 24GB 以上 GPU，或设置 ARGUS_FORCE_4BIT=1。"
                ) from exc
            if any(marker in str(exc).lower() for marker in ("401", "403", "gated", "unauthorized", "forbidden", "repository not found")):
                self._clear_model_state()
                raise RuntimeError("无法访问 Hugging Face 模型或 LoRA 仓库，请在 Space Secret 配置读取权限的 HF_TOKEN。") from exc
            self._clear_model_state()
            raise
        except Exception as exc:
            self._clear_model_state()
            if any(marker in str(exc).lower() for marker in ("401", "403", "gated", "unauthorized", "forbidden", "repository not found")):
                raise RuntimeError("无法访问 Hugging Face 模型或 LoRA 仓库，请在 Space Secret 配置读取权限的 HF_TOKEN。") from exc
            raise

    def ensure_ready(self, training_method: str, judge_reads_video: bool) -> Tuple[str, str]:
        """Load the base model and only the adapters needed by this method."""

        method = normalize_training_method(training_method)
        with self._lock:
            if self.model is None:
                self._load_base_and_observers(load_observers=method != "TRAINING_FREE")
            if method == "TRAINING_FREE":
                return "base_model", f"{BASE_MODEL_ID} (Training-Free; no LoRA)"

            from peft import PeftModel  # type: ignore

            self._ensure_observer_adapters(PeftModel)
            judge_name = judge_adapter_name(method, judge_reads_video)
            judge_source = select_judge_source(method, judge_reads_video)
            self._load_named_adapter(judge_name, judge_source)
            return judge_name, judge_source.display_path

    @staticmethod
    def _move_inputs(inputs: Any, device: Any, torch: Any) -> Any:
        if hasattr(inputs, "items"):
            for key, value in list(inputs.items()):
                if torch.is_tensor(value):
                    inputs[key] = value.to(device)
        elif hasattr(inputs, "to"):
            inputs = inputs.to(device)
        return inputs

    def _model_input_device(self) -> Any:
        """Pick the device used by the first language-model inputs."""

        candidate = getattr(self.model, "device", None)
        if candidate is not None and str(candidate) != "meta":
            return candidate
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return next(self.model.parameters()).device

    def _adapter_context(self, disable: bool = False) -> Any:
        """Temporarily disable all PEFT adapters for Training-Free inference."""

        if not disable or not self.loaded_adapter_names:
            return nullcontext()
        disable_adapter = getattr(self.model, "disable_adapter", None)
        if not callable(disable_adapter):
            raise RuntimeError("当前 PEFT 版本不支持 Training-Free 的无 LoRA 推理。")
        return disable_adapter()

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("内置模型尚未准备好。")
        torch = _import_torch()
        try:
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            image_inputs: Optional[Sequence[Any]] = None
            video_inputs: Optional[Sequence[Any]] = None
            try:
                from qwen_vl_utils import process_vision_info  # type: ignore

                image_inputs, video_inputs = process_vision_info(messages)
                has_image_items = any(
                    isinstance(item, Mapping) and item.get("type") == "image"
                    for message in messages
                    for item in (message.get("content") or [])
                    if isinstance(message.get("content"), list)
                )
                has_video_items = any(
                    isinstance(item, Mapping) and item.get("type") == "video"
                    for message in messages
                    for item in (message.get("content") or [])
                    if isinstance(message.get("content"), list)
                )
                if has_image_items and not image_inputs:
                    raise ValueError("qwen-vl-utils returned no image inputs")
                if has_video_items and not video_inputs:
                    raise ValueError("qwen-vl-utils returned no video inputs")
            except Exception:
                # Direct PIL fallbacks keep local smoke tests independent from
                # optional video decoder backends.
                image_inputs = [
                    item.get("image")
                    for message in messages
                    for item in (message.get("content") or [])
                    if isinstance(item, Mapping) and item.get("type") == "image" and item.get("image") is not None
                ]
                video_inputs = [
                    item.get("video")
                    for message in messages
                    for item in (message.get("content") or [])
                    if isinstance(item, Mapping) and item.get("type") == "video" and item.get("video") is not None
                ]

            processor_kwargs: Dict[str, Any] = {
                "text": [prompt_text],
                "padding": True,
                "return_tensors": "pt",
            }
            if image_inputs:
                processor_kwargs["images"] = image_inputs
            if video_inputs:
                processor_kwargs["videos"] = video_inputs
            inputs = self.processor(**processor_kwargs)
            device = self._model_input_device()
            inputs = self._move_inputs(inputs, device, torch)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens or _env_int(
                        "ARGUS_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS
                    ),
                    do_sample=False,
                )
            input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
            if input_ids is not None:
                trimmed = [output_ids[len(input_ids[i]) :] for i, output_ids in enumerate(generated_ids)]
            else:
                trimmed = generated_ids
            decoded = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            answer = str(decoded[0] if decoded else "").strip()
            if not answer:
                raise RuntimeError("模型返回了空报告，请重试或检查 checkpoint。")
            return answer
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError("推理时显存不足。请减少采样帧数或设置 ARGUS_FORCE_4BIT=1。") from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise RuntimeError("推理时显存不足。请减少采样帧数或设置 ARGUS_FORCE_4BIT=1。") from exc
            raise

    def generate_observer(
        self,
        role: str,
        sample: VideoSample,
        training_free: bool = False,
    ) -> str:
        role = role.lower().strip()
        if role not in OBSERVER_ROLES:
            raise ValueError(f"未知 Observation：{role}")
        with self._lock:
            if self.model is None:
                raise RuntimeError("内置模型尚未准备好，请先开始一次分析以加载模型。")
            with self._adapter_context(training_free):
                if not training_free:
                    self.model.set_adapter(role)
                return self._generate(
                    _qwen_messages(
                        observer_system_prompt(role),
                        observer_user_prompt(role, sample),
                        sample.frames,
                    ),
                    max_new_tokens=_env_int(
                        "ARGUS_OBSERVER_MAX_NEW_TOKENS", DEFAULT_OBSERVER_MAX_NEW_TOKENS
                    ),
                )

    def generate_judge(
        self,
        training_method: str,
        judge_reads_video: bool,
        reports: Mapping[str, str],
        sample: VideoSample,
    ) -> str:
        method = normalize_training_method(training_method)
        reads_video = normalize_bool(judge_reads_video)
        adapter_name = judge_adapter_name(method, reads_video)
        with self._lock:
            if self.model is None:
                raise RuntimeError("内置模型尚未准备好，请先开始一次分析以加载模型。")
            frames = sample.frames if reads_video else None
            with self._adapter_context(method == "TRAINING_FREE"):
                if method != "TRAINING_FREE":
                    if adapter_name not in self.loaded_adapter_names:
                        self._load_named_adapter(adapter_name, select_judge_source(method, reads_video))
                    self.model.set_adapter(adapter_name)
                return self._generate(
                    _qwen_messages(
                        JUDGE_SYSTEM_VIDEO if reads_video else JUDGE_SYSTEM_TEXT,
                        judge_user_prompt(reports, sample, reads_video),
                        frames,
                    ),
                    max_new_tokens=_env_int(
                        "ARGUS_JUDGE_MAX_NEW_TOKENS", DEFAULT_JUDGE_MAX_NEW_TOKENS
                    ),
                )


MODEL_MANAGER = ArgusModelManager()


def _gpu_duration_for_call(*args: Any, **kwargs: Any) -> int:
    """Choose a ZeroGPU reservation from the request's actual workload.

    API-only requests do not use the local model and need only a short
    reservation for video decoding.  Video-aware built-in inference gets the
    larger budget; text-only built-in inference keeps the smaller historical
    budget.  ``ARGUS_GPU_DURATION_SECONDS`` remains an explicit override.
    """

    model_source = kwargs.get("model_source")
    judge_reads_video = kwargs.get("judge_reads_video")
    if model_source is None and len(args) > 1:
        model_source = args[1]
    if judge_reads_video is None and len(args) > 3:
        judge_reads_video = args[3]

    # Keep the effective request below the usual 300-second ZeroGPU ceiling.
    # The installed ``spaces`` package knows the multiplier for the selected
    # GPU family; fall back conservatively when running outside a Space.
    raw_cap = 200
    try:
        from spaces.zero import config as zero_config  # type: ignore

        factor = float(zero_config.get_config().get("duration_factor", 1.0))
        if factor > 0:
            raw_cap = min(300, max(60, int(300 / factor)))
    except Exception:
        pass

    if model_source == SOURCE_API:
        default = 60
    elif normalize_bool(judge_reads_video):
        default = raw_cap
    else:
        default = min(180, raw_cap)
    return min(raw_cap, max(60, _env_int("ARGUS_GPU_DURATION_SECONDS", default)))


def _gpu_task(function: Callable[..., Any]) -> Callable[..., Any]:
    """Use ZeroGPU when the Space provides it; remain a no-op on regular GPUs."""

    if _spaces is None:
        return function
    # ZeroGPU otherwise stops GPU functions after its 60-second default.  A
    # cold Qwen2.5-VL load plus five adapter-backed generations needs a wider
    # window, especially after a Space restart.
    # The dynamic reservation stays at or below the usual 300-second effective
    # per-call ceiling while leaving enough time for the video-aware Judge.
    return _spaces.GPU(duration=_gpu_duration_for_call)(function)


# ---------------------------------------------------------------------------
# OpenAI-compatible API backend
# ---------------------------------------------------------------------------


class APIRequestError(RuntimeError):
    """User-facing API error that deliberately excludes credentials."""


def normalize_chat_completions_url(base_url: str) -> str:
    raw = str(base_url or "").strip().rstrip("/")
    if not raw:
        raise ValueError("请填写 Custom API 的 Base URL。")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Base URL 必须是完整的 http(s) 地址，例如 https://example.com/v1。")
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        return raw
    if path.endswith("/v1"):
        return raw + "/chat/completions"
    return raw + "/v1/chat/completions"


def _api_content_parts(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces: List[str] = []
        for part in content:
            if isinstance(part, Mapping):
                text = part.get("text")
                if text:
                    pieces.append(str(text))
        return "\n".join(pieces).strip()
    return str(content or "").strip()


def call_openai_compatible(
    base_url: str,
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_content: Any,
    *,
    timeout: Optional[float] = None,
) -> str:
    """Call ``/chat/completions`` without persisting the request-scoped key."""

    key = str(api_key or "").strip()
    model = str(model_name or "").strip()
    if not key:
        raise ValueError("请填写 Custom API 的 API Key。")
    if not model:
        raise ValueError("请填写 Custom API 的 Model Name。")
    endpoint = normalize_chat_completions_url(base_url)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_tokens": _env_int("ARGUS_API_MAX_TOKENS", 512),
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    request_timeout = timeout if timeout is not None else _env_float("ARGUS_API_TIMEOUT", DEFAULT_API_TIMEOUT_SECONDS)
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=request_timeout)
    except requests.exceptions.Timeout as exc:
        raise APIRequestError("Custom API 请求超时，请检查 Base URL 或稍后重试。") from exc
    except requests.exceptions.RequestException as exc:
        raise APIRequestError(f"无法连接 Custom API：{exc.__class__.__name__}。") from exc

    if response.status_code >= 400:
        # Do not include request headers or the API key in diagnostics.
        detail = (getattr(response, "text", "") or "").strip().replace("\n", " ")
        detail = detail.replace(key, "[redacted]")
        if len(detail) > 300:
            detail = detail[:300] + "…"
        suffix = f"：{detail}" if detail else "。"
        raise APIRequestError(f"Custom API 返回 HTTP {response.status_code}{suffix}")
    try:
        data = response.json()
    except ValueError as exc:
        raise APIRequestError("Custom API 返回的不是有效 JSON。") from exc
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise APIRequestError("Custom API 响应缺少 choices[0].message.content。") from exc
    answer = _api_content_parts(content)
    if not answer:
        raise APIRequestError("Custom API 返回了空内容。")
    return answer


def build_api_user_content(
    prompt: str,
    frames: Sequence[Image.Image],
    *,
    include_frames: bool,
) -> Any:
    """Build OpenAI-compatible text/image_url content parts."""

    if not include_frames:
        return prompt
    parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, frame in enumerate(frames, start=1):
        parts.append({"type": "text", "text": f"Sampled frame {index}/{len(frames)}:"})
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": frame_to_data_url(frame), "detail": "low"},
            }
        )
    return parts


def run_api_observer(
    role: str,
    sample: VideoSample,
    base_url: str,
    api_key: str,
    model_name: str,
) -> str:
    return call_openai_compatible(
        base_url,
        api_key,
        model_name,
        observer_system_prompt(role),
        build_api_user_content(observer_user_prompt(role, sample), sample.frames, include_frames=True),
    )


def run_api_judge(
    sample: VideoSample,
    reports: Mapping[str, str],
    base_url: str,
    api_key: str,
    model_name: str,
    judge_reads_video: bool,
) -> str:
    reads_video = normalize_bool(judge_reads_video)
    return call_openai_compatible(
        base_url,
        api_key,
        model_name,
        JUDGE_SYSTEM_VIDEO if reads_video else JUDGE_SYSTEM_TEXT,
        build_api_user_content(
            judge_user_prompt(reports, sample, reads_video),
            sample.frames,
            include_frames=reads_video,
        ),
    )


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------


def _progress(progress: Optional[Callable[..., Any]], value: float, desc: str) -> None:
    if progress:
        try:
            progress(value, desc=desc)
        except TypeError:
            progress(value)


def _redact_secret(message: Any, secret: Any) -> str:
    """Keep an accidental provider echo out of user-facing status text."""

    text = str(message)
    token = str(secret or "")
    return text.replace(token, "[redacted]") if token else text


def _blank_result(
    reports: Optional[Mapping[str, str]] = None,
    judge: str = "",
    checkpoint: str = "",
    status: str = "",
) -> Tuple[str, str, str, str, str]:
    reports = reports or {}
    return (
        str(reports.get("texture", "")),
        str(reports.get("lighting", "")),
        str(reports.get("motion", "")),
        str(reports.get("physics", "")),
        str(judge or ""),
    )


JUDGE_PENDING_TEXT = "*Judge inference in progress…*"


def _error_result(
    reports: Mapping[str, str], message: str
) -> Tuple[str, str, str, str, str]:
    """Surface failures in the remaining Judge panel after Status was removed."""

    return _blank_result(
        reports=reports,
        judge=f"**Analysis failed:** {message}",
    )


@_gpu_task
def analyze_video(
    video: Any,
    model_source: str,
    training_method: str,
    judge_reads_video: bool,
    base_url: str,
    api_key: str,
    model_name: str,
    progress: Optional[Callable[..., Any]] = gr.Progress(track_tqdm=False),
) -> Iterable[Tuple[str, str, str, str, str]]:
    """Gradio generator that streams Observation reports as they complete."""

    reports: Dict[str, str] = {}
    judge_result = ""
    reads_video = normalize_bool(judge_reads_video)
    checkpoint = selected_checkpoint_display(model_source, training_method, reads_video)
    try:
        if model_source not in {SOURCE_ARGUS, SOURCE_API}:
            raise ValueError("请选择模型来源。")
        method = normalize_training_method(training_method)
        if model_source == SOURCE_API:
            # Validate API parameters before decoding a potentially large video.
            normalize_chat_completions_url(base_url)
            if not str(api_key or "").strip():
                raise ValueError("请填写 Custom API 的 API Key。")
            if not str(model_name or "").strip():
                raise ValueError("请填写 Custom API 的 Model Name。")

        yield _blank_result(
            checkpoint=checkpoint,
            status="正在读取视频…",
        )
        sample = sample_video_frames(video)
        yield _blank_result(
            checkpoint=checkpoint,
            status="视频读取完成，开始分析。",
        )

        if model_source == SOURCE_ARGUS:
            load_message = (
                "加载基础 Qwen2.5-VL（Training-Free；不加载 LoRA）"
                if method == "TRAINING_FREE"
                else "加载 Argus LoRA（首次运行需要下载）"
            )
            _progress(progress, 0.12, load_message)
            judge_name, checkpoint = MODEL_MANAGER.ensure_ready(method, reads_video)
            status = f"{MODEL_MANAGER.load_summary} · Judge adapter: {judge_name}"
            yield _blank_result(
                checkpoint=checkpoint,
                status=status,
            )
            for offset, role in enumerate(OBSERVER_ROLES):
                _progress(progress, 0.2 + offset * 0.15, f"Observation · {role}")
                reports[role] = normalize_observer_output(
                    MODEL_MANAGER.generate_observer(
                        role,
                        sample,
                        training_free=method == "TRAINING_FREE",
                    )
                )
                yield _blank_result(
                    reports=reports,
                    checkpoint=checkpoint,
                    status=f"已完成 Observation：{role}。",
                )
            _progress(progress, 0.84, "Judge 汇总四份报告")
            # The last Observer update otherwise clears the fifth output while
            # the video-aware Judge is still generating. Keep that panel
            # visibly busy even if a ZeroGPU task is later aborted.
            yield _blank_result(
                reports=reports,
                judge=JUDGE_PENDING_TEXT,
                checkpoint=checkpoint,
                status="Judge inference in progress",
            )
            judge_result = normalize_judge_output(
                MODEL_MANAGER.generate_judge(method, reads_video, reports, sample)
            )
            _progress(progress, 1.0, "分析完成")
            yield _blank_result(
                reports=reports,
                judge=judge_result,
                checkpoint=checkpoint,
                status="分析完成。",
            )
            return

        # Custom API mode: the key is used only by this request and is never
        # stored in global state, status text, files, or logs.
        api_model = str(model_name).strip()
        for offset, role in enumerate(OBSERVER_ROLES):
            _progress(progress, 0.15 + offset * 0.16, f"Custom API Observation · {role}")
            reports[role] = normalize_observer_output(
                run_api_observer(role, sample, base_url, api_key, api_model)
            )
            yield _blank_result(
                reports=reports,
                checkpoint=checkpoint,
                status=f"已完成 Custom API Observation：{role}。",
            )
        _progress(progress, 0.84, "Custom API Judge 汇总")
        yield _blank_result(
            reports=reports,
            judge=JUDGE_PENDING_TEXT,
            checkpoint=checkpoint,
            status="Custom API Judge inference in progress",
        )
        judge_result = normalize_judge_output(
            run_api_judge(
                sample,
                reports,
                base_url,
                api_key,
                api_model,
                reads_video,
            )
        )
        _progress(progress, 1.0, "分析完成")
        yield _blank_result(
            reports=reports,
            judge=judge_result,
            checkpoint=f"Custom API：{api_model} · Judge {'video' if reads_video else 'text'} mode",
            status="分析完成。API Key 仅用于本次请求。",
        )
    except ValueError as exc:
        yield _error_result(reports, f"Invalid parameter or video: {_redact_secret(exc, api_key)}")
    except APIRequestError as exc:
        yield _error_result(reports, f"Custom API error: {_redact_secret(exc, api_key)}")
    except RuntimeError as exc:
        yield _error_result(reports, f"Built-in model or inference error: {_redact_secret(exc, api_key)}")
    except Exception as exc:  # keep the UI readable while preserving a useful class
        yield _error_result(
            reports,
            f"Unexpected {exc.__class__.__name__}: {_redact_secret(exc, api_key)}",
        )


def update_source_visibility(model_source: str) -> Tuple[Any, Any]:
    return (
        gr.update(visible=model_source == SOURCE_ARGUS),
        gr.update(visible=model_source == SOURCE_API),
    )


def update_checkpoint(model_source: str, training_method: str, judge_reads_video: bool) -> str:
    return selected_checkpoint_display(model_source, training_method, judge_reads_video)


def build_demo() -> gr.Blocks:
    # Gradio 5 and 6 calculate the top inset of a nested, containerless
    # column differently. Keep the checkbox baseline aligned with the
    # training-method radio buttons in both the local base environment and
    # the Space runtime without injecting CSS.
    try:
        gradio_major = int(str(getattr(gr, "__version__", "6")).split(".", 1)[0])
    except (TypeError, ValueError):
        gradio_major = 6
    training_checkbox_spacer_height = 22 if gradio_major < 6 else 12

    custom_css = """
    *, body, input, button, select, textarea, .gradio-container, .prose, h1, h2, h3, h4, h5, h6, span, div, label, button span, .tabitem {
        font-family: "Times New Roman", Times, serif !important;
    }
    """

    with gr.Blocks(theme=gr.themes.Default(), title="ARGUS", css=custom_css) as demo:
        gr.Markdown("<h1 style='text-align: center; font-weight: bold;'>ARGUS: Multi-Agent Forensic Reasoning for Generalizable Deepfake Video Detection</h1>")

        # 第一区块：视频输入与参数（左侧 scale=4，右侧 scale=8 黄金比例分栏）
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=320):
                video = gr.Video(
                    label="Upload video",
                    sources=["upload"],
                    height=150,
                )
                example_cards, example_video_paths = bundled_example_assets()
                if example_cards:
                    examples_parameters = inspect.signature(gr.Examples).parameters
                    examples_api_options: Dict[str, Any] = (
                        {"api_visibility": "private", "api_name": None}
                        if "api_visibility" in examples_parameters
                        else {"show_api": False, "api_name": False}
                    )
                    gr.Examples(
                        examples=[[path] for path in example_video_paths],
                        inputs=[video],
                        examples_per_page=len(example_video_paths),
                        label="Bundled examples",
                        cache_examples=False,
                        **examples_api_options,
                    )
                else:
                    gr.Markdown("No bundled examples available.")

                model_source = gr.Radio(
                    choices=[SOURCE_ARGUS, SOURCE_API],
                    value=SOURCE_ARGUS,
                    label="Model source",
                )
                with gr.Column(visible=True) as builtin_settings:
                    training_method = gr.Radio(
                        choices=list(TRAINING_METHOD_CHOICES),
                        value=TRAINING_METHOD_LABELS["SFT"],
                        label="Training method",
                    )
                    judge_reads_video = gr.Checkbox(
                        value=False,
                        label="Judge reads raw video",
                    )
                with gr.Column(visible=False) as api_settings:
                    base_url = gr.Textbox(
                        label="Base URL",
                        placeholder="https://your-endpoint.example.com/v1",
                    )
                    api_key = gr.Textbox(
                        label="API Key",
                        type="password",
                        placeholder="Saved for current request only",
                    )
                    model_name = gr.Textbox(
                        label="Model Name",
                        placeholder="e.g. gpt-4o-mini / qwen-vl-plus",
                    )
                start_button = gr.Button("Start Analysis", variant="primary", size="lg")

            # 右侧：Observation 在上方展示（切换查看），Judge 汇总放在下方（scale=8）
            with gr.Column(scale=8):
                gr.Markdown("### Observation Agent Reports")
                with gr.Tabs():
                    with gr.TabItem("Texture Agent"):
                        texture = gr.Markdown(
                            value="*Awaiting Texture Agent analysis...*",
                            container=True,
                        )
                    with gr.TabItem("Lighting Agent"):
                        lighting = gr.Markdown(
                            value="*Awaiting Lighting Agent analysis...*",
                            container=True,
                        )
                    with gr.TabItem("Motion Agent"):
                        motion = gr.Markdown(
                            value="*Awaiting Motion Agent analysis...*",
                            container=True,
                        )
                    with gr.TabItem("Physics Agent"):
                        physics = gr.Markdown(
                            value="*Awaiting Physics Agent analysis...*",
                            container=True,
                        )

                gr.Markdown("### Judge Agent — Judgement Output")
                judge_result = gr.Markdown(
                    value="*Awaiting Judge final assessment...*",
                    container=True,
                )

        outputs = [texture, lighting, motion, physics, judge_result]

        # gr.Video only converts videos it *outputs*.  An upload is passed to
        # the browser player as-is, so re-encode non-web-playable uploads here
        # instead of letting the <video> element fail with "Video not playable".
        upload_parameters = inspect.signature(video.upload).parameters
        upload_api_options: Dict[str, Any] = (
            {"api_visibility": "private", "api_name": None}
            if "api_visibility" in upload_parameters
            else {"show_api": False, "api_name": False}
        )
        video.upload(
            transcode_for_preview,
            inputs=[video],
            outputs=[video],
            **upload_api_options,
        )

        model_source.change(
            update_source_visibility,
            inputs=[model_source],
            outputs=[builtin_settings, api_settings],
        )
        start_button.click(
            analyze_video,
            inputs=[video, model_source, training_method, judge_reads_video, base_url, api_key, model_name],
            outputs=outputs,
            concurrency_limit=1,
        )

    demo.queue(default_concurrency_limit=1)
    return demo


demo = build_demo()


if __name__ == "__main__":
    port = _env_int("PORT", 7860)
    demo.launch(server_name="0.0.0.0", server_port=port)
