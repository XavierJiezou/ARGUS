"""ms-swift reward functions for the ARGUS Judge GRPO stage."""

from __future__ import annotations

import re
from typing import List

from swift.rewards import ORM, orms


def prediction(text: str) -> str:
    strict = re.search(r"<answer>\s*(real|fake)\s*</answer>", text or "", re.IGNORECASE)
    if strict:
        return strict.group(1).lower()
    tokens = re.findall(r"\b(real|fake)\b", text or "", re.IGNORECASE)
    return tokens[-1].lower() if tokens else ""


def target(label: str) -> str:
    normalized = str(label).strip().lower()
    if normalized == "real":
        return "real"
    if normalized in {"efs", "fs", "fr", "fake"}:
        return "fake"
    raise ValueError(f"Unsupported label: {label!r}")


def explanation(text: str) -> str:
    match = re.search(
        r"<explanation>\s*(.*?)\s*</explanation>", text or "", re.IGNORECASE | re.DOTALL
    )
    return match.group(1).strip() if match else ""


class JudgeAccuracy(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        return [
            1.0 if prediction(completion) == target(label) else 0.0
            for completion, label in zip(completions, solution)
        ]


class JudgeLength(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        rewards: List[float] = []
        for completion in completions:
            words = len(re.findall(r"[A-Za-z0-9]+", explanation(completion)))
            if words < 8:
                score = words / 8.0
            elif words <= 140:
                score = 1.0
            elif words >= 200:
                score = 0.0
            else:
                score = 1.0 - (words - 140) / 60.0
            rewards.append(max(0.0, score))
        return rewards


class JudgeFormat(ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        rewards: List[float] = []
        for completion in completions:
            checks = (
                bool(explanation(completion)),
                bool(
                    re.search(
                        r"<answer>\s*(real|fake)\s*</answer>", completion, re.IGNORECASE
                    )
                ),
                len(re.findall(r"<answer>", completion, re.IGNORECASE)) == 1,
            )
            rewards.append(sum(checks) / len(checks))
        return rewards


orms["judge_accuracy"] = JudgeAccuracy
orms["judge_length"] = JudgeLength
orms["judge_format"] = JudgeFormat
