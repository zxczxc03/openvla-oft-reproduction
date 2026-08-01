"""Extract LIBERO language-instruction variants into reverse lookup maps.

This script reads LIBERO's benchmark task map and BDDL files without importing
LIBERO itself, then writes the compact runtime map used by the advantage and
RECAP scripts. JSON output remains available when the full task metadata is
needed for inspection.

Example:
    python experiments/robot/libero/extract_libero_language_map.py

By default, this writes ``vla-scripts/libero_language_to_task.py`` so the
consumer scripts can import it directly.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import string
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from typing import Any


DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
LANGUAGE_VARIANT_RE = re.compile(r"_language_(\d+)")
SCENE_PREFIX_RE = re.compile(r"^[A-Z_]+_SCENE\d+_")
PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


def normalize_language(language: str) -> str:
    """Normalize an instruction for forgiving reverse lookup."""
    return " ".join(language.lower().translate(PUNCT_TRANSLATION).split())


def infer_benchmark_dir() -> Path:
    candidates = []

    if os.environ.get("LIBERO_BENCHMARK_DIR"):
        candidates.append(Path(os.environ["LIBERO_BENCHMARK_DIR"]))

    if os.environ.get("LIBERO_ROOT"):
        libero_root = Path(os.environ["LIBERO_ROOT"])
        candidates.extend(
            [
                libero_root / "libero" / "libero" / "benchmark",
                libero_root / "libero" / "benchmark",
            ]
        )

    repo_root = Path(__file__).resolve().parents[3]
    candidates.extend(
        [
            repo_root.parent / "LIBERO-plus" / "libero" / "libero" / "benchmark",
            Path.home() / "LIBERO-plus" / "libero" / "libero" / "benchmark",
        ]
    )

    for candidate in candidates:
        if (candidate / "libero_suite_task_map.py").exists():
            return candidate

    raise FileNotFoundError(
        "Could not infer LIBERO benchmark dir. Pass --benchmark-dir, set LIBERO_BENCHMARK_DIR, "
        "or set LIBERO_ROOT."
    )


def load_libero_task_map(task_map_path: Path) -> dict[str, list[str]]:
    module = ast.parse(task_map_path.read_text(encoding="utf-8"), filename=str(task_map_path))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "libero_task_map":
                    return ast.literal_eval(node.value)

    raise ValueError(f"Could not find libero_task_map assignment in {task_map_path}")


def load_task_classification(classification_path: Path) -> dict[str, dict[str, Any]]:
    if not classification_path.exists():
        return {}

    data = json.loads(classification_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "tasks" in data or "data" in data:
            records = data.get("tasks", data.get("data", []))
        else:
            records = []
            for suite_records in data.values():
                if isinstance(suite_records, list):
                    records.extend(suite_records)
    else:
        records = data

    task_info = {}
    for record in records:
        if isinstance(record, dict) and "name" in record:
            task_info[record["name"]] = {
                key: value for key, value in record.items() if key != "name"
            }
    return task_info


def split_view_suffix(task_name: str) -> tuple[str, str]:
    if "_view_" not in task_name:
        return task_name, ""

    bddl_stem, view_suffix = task_name.split("_view_", 1)
    return bddl_stem, f"_view_{view_suffix}"


def extract_language_from_bddl(bddl_path: Path) -> str:
    text = bddl_path.read_text(encoding="utf-8")
    marker = "(:language"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"No :language field found in {bddl_path}")

    index = start + len(marker)
    depth = 1
    language_chars = []
    while index < len(text):
        char = text[index]
        if char == "(":
            depth += 1
            language_chars.append(char)
        elif char == ")":
            depth -= 1
            if depth == 0:
                break
            language_chars.append(char)
        else:
            language_chars.append(char)
        index += 1

    if depth != 0:
        raise ValueError(f"Unclosed :language field in {bddl_path}")

    return " ".join("".join(language_chars).split())


def canonicalize_task_name(task_name: str) -> str:
    no_scene = SCENE_PREFIX_RE.sub("", task_name)
    return no_scene.replace("_", " ")


def language_variant_id(task_stem: str) -> int | None:
    match = LANGUAGE_VARIANT_RE.search(task_stem)
    return int(match.group(1)) if match else None


def build_maps(
    benchmark_dir: Path,
    bddl_root: Path,
    suites: list[str],
    language_only: bool,
) -> dict[str, Any]:
    task_map = load_libero_task_map(benchmark_dir / "libero_suite_task_map.py")
    task_classification = load_task_classification(benchmark_dir / "task_classification.json")

    language_to_tasks = defaultdict(list)
    missing_bddl_files = []
    total_tasks_seen = 0
    total_language_tasks_seen = 0

    for suite in suites:
        if suite not in task_map:
            raise KeyError(f"Suite {suite!r} is not in libero_task_map. Available suites: {sorted(task_map)}")

        for task_index, task_name in enumerate(task_map[suite]):
            total_tasks_seen += 1
            if "_language_" in task_name:
                total_language_tasks_seen += 1
            elif language_only:
                continue

            bddl_stem, view_suffix = split_view_suffix(task_name)
            bddl_file = f"{bddl_stem}.bddl"
            bddl_path = bddl_root / suite / bddl_file
            if not bddl_path.exists():
                missing_bddl_files.append(str(bddl_path))
                continue

            language = extract_language_from_bddl(bddl_path)
            if "_language_" in bddl_stem:
                base_task = bddl_stem.split("_language_", 1)[0]
            else:
                base_task = bddl_stem

            language_key = language.lower()
            entry = {
                "language": language,
                "suite": suite,
                "task_index": task_index,
                "task_name": task_name,
                "base_task": base_task,
                "canonical_task": canonicalize_task_name(base_task),
                "language_variant": language_variant_id(bddl_stem),
                "bddl_file": bddl_file,
                "bddl_path": str(bddl_path),
                "view_suffix": view_suffix,
                "classification": task_classification.get(task_name),
            }
            language_to_tasks[language_key].append(entry)

    language_to_tasks = dict(sorted(language_to_tasks.items()))
    normalized_language_to_tasks = defaultdict(list)
    for language_key, entries in language_to_tasks.items():
        normalized_language_to_tasks[normalize_language(language_key)].extend(entries)

    language_to_base_tasks = {
        language: sorted({entry["base_task"] for entry in entries}) for language, entries in language_to_tasks.items()
    }
    normalized_language_to_base_tasks = {
        language: sorted({entry["base_task"] for entry in entries})
        for language, entries in sorted(normalized_language_to_tasks.items())
    }

    return {
        "metadata": {
            "benchmark_dir": str(benchmark_dir),
            "bddl_root": str(bddl_root),
            "suites": suites,
            "language_only": language_only,
            "num_tasks_seen": total_tasks_seen,
            "num_language_tasks_seen": total_language_tasks_seen,
            "num_language_keys": len(language_to_tasks),
            "num_normalized_language_keys": len(normalized_language_to_tasks),
            "num_missing_bddl_files": len(missing_bddl_files),
            "missing_bddl_files": missing_bddl_files,
        },
        "language_to_tasks": language_to_tasks,
        "normalized_language_to_tasks": dict(sorted(normalized_language_to_tasks.items())),
        "language_to_base_tasks": language_to_base_tasks,
        "normalized_language_to_base_tasks": normalized_language_to_base_tasks,
    }


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_python(data: dict[str, Any], path: Path) -> None:
    metadata = {
        key: data["metadata"][key]
        for key in (
            "suites",
            "language_only",
            "num_tasks_seen",
            "num_language_tasks_seen",
            "num_language_keys",
            "num_normalized_language_keys",
            "num_missing_bddl_files",
        )
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            '"""Generated by experiments/robot/libero/extract_libero_language_map.py; do not edit manually."""',
            "# ruff: noqa",
            "from __future__ import annotations",
            "",
            "import string",
            "",
            "# fmt: off",
            f"METADATA = {pformat(metadata, width=120)}",
            "",
            f"NORMALIZED_LANGUAGE_TO_BASE_TASKS = {pformat(data['normalized_language_to_base_tasks'], width=120)}",
            "# fmt: on",
            "",
            inspect_normalize_language_source(),
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def inspect_normalize_language_source() -> str:
    return """\
_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


def normalize_language(language: str) -> str:
    return " ".join(language.lower().translate(_PUNCT_TRANSLATION).split())
"""


def parse_args() -> argparse.Namespace:
    default_output_prefix = Path(__file__).resolve().parents[3] / "vla-scripts" / "libero_language_to_task"
    parser = argparse.ArgumentParser(description="Extract LIBERO language instructions into reverse lookup maps.")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Path to LIBERO benchmark dir. Can also be supplied through LIBERO_BENCHMARK_DIR or LIBERO_ROOT.",
    )
    parser.add_argument(
        "--bddl-root",
        type=Path,
        default=None,
        help="Path to LIBERO bddl_files dir. Defaults to <benchmark-dir>/../bddl_files.",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        default=list(DEFAULT_SUITES),
        help=f"Task suites to scan. Defaults to: {' '.join(DEFAULT_SUITES)}.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=default_output_prefix,
        help=f"Output prefix. Extensions are added from --format. Default: {default_output_prefix}",
    )
    parser.add_argument(
        "--format",
        choices=["json", "py", "both"],
        default="py",
        help="Output format. Python output is compact; JSON contains the full task metadata.",
    )
    parser.add_argument(
        "--include-all-tasks",
        action="store_true",
        help="Include non-language tasks too. By default only task names containing _language_ are exported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_dir = (args.benchmark_dir or infer_benchmark_dir()).resolve()
    bddl_root = (args.bddl_root or benchmark_dir.parent / "bddl_files").resolve()

    data = build_maps(
        benchmark_dir=benchmark_dir,
        bddl_root=bddl_root,
        suites=args.suites,
        language_only=not args.include_all_tasks,
    )

    written_paths = []
    if args.format in {"json", "both"}:
        json_path = args.output_prefix.with_suffix(".json")
        write_json(data, json_path)
        written_paths.append(json_path)
    if args.format in {"py", "both"}:
        py_path = args.output_prefix.with_suffix(".py")
        write_python(data, py_path)
        written_paths.append(py_path)

    metadata = data["metadata"]
    print(f"Extracted {metadata['num_language_keys']} language keys from {metadata['num_language_tasks_seen']} language tasks.")
    if metadata["num_missing_bddl_files"]:
        print(f"Skipped {metadata['num_missing_bddl_files']} tasks with missing BDDL files.")
    for path in written_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
