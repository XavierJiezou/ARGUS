"""Score an ARGUS prediction file."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .inference import compute_metrics, iter_records, write_json_atomic


def indexed_rows(path: Path) -> tuple[list[str], Dict[str, Dict[str, Any]]]:
    order: list[str] = []
    rows: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(iter_records(path)):
        video_path = str(row.get("video_path", "")).strip()
        if not video_path:
            raise ValueError(f"Missing video_path in {path} at row {index}")
        if video_path in rows:
            raise ValueError(f"Duplicate video_path in {path}: {video_path}")
        order.append(video_path)
        rows[video_path] = row
    return order, rows


def grouped_metrics(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    groups: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(field, "")).strip()
        if value:
            groups[value].append(row)
    return {
        name: compute_metrics(group_rows)
        for name, group_rows in sorted(groups.items())
    }


def report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "overall": compute_metrics(rows),
        "by_source": grouped_metrics(rows, "source"),
        "by_dataset": grouped_metrics(rows, "dataset"),
    }


def print_metrics(name: str, values: Mapping[str, Any]) -> None:
    print(
        f"{name}: n={values['total']} "
        f"balanced_accuracy={values['balanced_accuracy']:.4f} "
        f"fake_recall={values['fake_recall']:.4f} "
        f"fake_f1={values['fake_f1']:.4f}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prediction", type=Path, help="Prediction JSON or JSONL written by inference."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the metrics report; defaults to <prediction>_metrics.json.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prediction = args.prediction.expanduser().resolve()
    order, indexed = indexed_rows(prediction)
    rows = [indexed[video_path] for video_path in order]
    result = report(rows)
    print_metrics(str(prediction), result["overall"])
    output = args.output or prediction.with_name(f"{prediction.stem}_metrics.json")
    write_json_atomic(output.expanduser().resolve(), result)


if __name__ == "__main__":
    main()
