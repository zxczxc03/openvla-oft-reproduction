#!/usr/bin/env python3
"""
Convert a LIBERO / LIBERO-Plus eval log into summary images and CSV files.

Outputs:
  - eval_summary_table.png     leaderboard-style table
  - eval_category_bar.png      category success-rate bar chart
  - eval_cumulative_curve.png  cumulative success-rate curve
  - eval_summary.csv           category-level numbers
  - eval_episodes.csv          per-episode parsed records
  - eval_failures.csv          failed episodes only

Usage:
  python libero_plus_log_to_image.py \
      --log_path ./experiments/logs/EVAL-xxx.txt \
      --model_name OpenVLA-OFT \
      --output_dir ./eval_figures

Category mapping is based on the task-description patterns in LIBERO-Plus logs:
  Camera      : contains " view " but not noise/light
  Noise       : contains " noise "
  Light       : contains " light "
  Background  : ends with "levelK sampleN"
  Layout      : ends with "add N"
  Robot       : ends with "table N" or "tb N"
  Language    : the remaining paraphrased / language-variation tasks

If your task names use different tags, edit infer_category().
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATEGORY_ORDER = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]


def infer_category(task: str) -> str:
    """Infer LIBERO-Plus category from a logged task description."""
    t = f" {task.strip()} "

    # Check explicit perturbation tags first. Noise tasks often also contain a camera/view tag.
    if " noise " in t:
        return "Noise"
    if " light " in t:
        return "Light"
    if " view " in t:
        return "Camera"

    # Background / layout tags used by the uploaded LIBERO-Plus-style log.
    if re.search(r"\blevel\d+\s+sample\d+\s*$", task):
        return "Background"
    if re.search(r"\badd\s+\d+\s*$", task):
        return "Layout"

    # Robot/domain/layout variants in the uploaded log are tagged as table/tb.
    if re.search(r"\b(?:table|tb)\s+\d+\s*$", task):
        return "Robot"

    # Everything else is usually a language paraphrase in LIBERO-Plus logs.
    return "Language"


def strip_variant_suffix(task: str) -> str:
    """Remove common LIBERO-Plus suffixes to get a coarse base task string."""
    patterns = [
        r"\s+view\s+.*$",
        r"\s+noise\s+\d+\s*$",
        r"\s+light\s+\d+\s*$",
        r"\s+level\d+\s+sample\d+\s*$",
        r"\s+add\s+\d+\s*$",
        r"\s+(?:table|tb)\s+\d+\s*$",
    ]
    base = task
    for pat in patterns:
        base = re.sub(pat, "", base)
    return base.strip()


def parse_eval_log(log_path: str | os.PathLike) -> List[Dict[str, object]]:
    """Parse per-episode task/success records from an eval log."""
    records: List[Dict[str, object]] = []
    current: Dict[str, object] = {}

    task_re = re.compile(r"^Task:\s*(.*)$")
    success_re = re.compile(r"^Success:\s*(True|False)\s*$")
    episode_re = re.compile(r"episode=(\d+)--success=(True|False)")
    completed_re = re.compile(r"^# episodes completed so far:\s*(\d+)")
    successes_re = re.compile(r"^# successes:\s*(\d+)\s*\(([^)]+)\)")

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()

            m = task_re.match(line)
            if m:
                current = {"task": m.group(1).strip()}
                continue

            m = episode_re.search(line)
            if m and current:
                current["episode_from_video"] = int(m.group(1))
                current["success_from_video"] = (m.group(2) == "True")
                continue

            m = success_re.match(line)
            if m and current:
                current["success"] = (m.group(1) == "True")
                current["episode_index"] = len(records) + 1
                task = str(current.get("task", ""))
                current["category"] = infer_category(task)
                current["base_task"] = strip_variant_suffix(task)
                records.append(current)
                current = {}
                continue

            m = completed_re.match(line)
            if m and records:
                records[-1]["episodes_completed_so_far"] = int(m.group(1))
                continue

            m = successes_re.match(line)
            if m and records:
                records[-1]["successes_so_far"] = int(m.group(1))
                records[-1]["success_percent_so_far_text"] = m.group(2)
                continue

    return records


def summarize(records: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"episodes": 0, "successes": 0, "success_rate": 0.0})

    total_eps = 0
    total_succ = 0
    for r in records:
        cat = str(r.get("category", "Unknown"))
        success = bool(r.get("success", False))
        stats[cat]["episodes"] += 1
        stats[cat]["successes"] += int(success)
        total_eps += 1
        total_succ += int(success)

    for cat, d in stats.items():
        d["success_rate"] = d["successes"] / d["episodes"] if d["episodes"] else 0.0

    stats["Total"] = {
        "episodes": total_eps,
        "successes": total_succ,
        "success_rate": total_succ / total_eps if total_eps else 0.0,
    }
    return stats


def _pct(stats: Dict[str, Dict[str, float]], key: str) -> str:
    if key not in stats or stats[key]["episodes"] == 0:
        return "-"
    return f"{100.0 * stats[key]['success_rate']:.1f}"


def save_summary_csv(stats: Dict[str, Dict[str, float]], output_path: str | os.PathLike) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "episodes", "successes", "success_rate", "success_percent"])
        for cat in CATEGORY_ORDER + ["Total"]:
            if cat in stats:
                d = stats[cat]
                writer.writerow([cat, int(d["episodes"]), int(d["successes"]), f"{d['success_rate']:.6f}", f"{100*d['success_rate']:.2f}"])


def save_episode_csv(records: List[Dict[str, object]], output_path: str | os.PathLike) -> None:
    fieldnames = [
        "episode_index",
        "episode_from_video",
        "category",
        "success",
        "task",
        "base_task",
        "episodes_completed_so_far",
        "successes_so_far",
        "success_percent_so_far_text",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def save_failures_csv(records: List[Dict[str, object]], output_path: str | os.PathLike) -> None:
    failures = [r for r in records if not bool(r.get("success", False))]
    save_episode_csv(failures, output_path)


def plot_summary_table(stats: Dict[str, Dict[str, float]], output_path: str | os.PathLike, model_name: str = "Model") -> None:
    columns = ["Model"] + CATEGORY_ORDER + ["Total"]
    row = [model_name] + [_pct(stats, c) for c in CATEGORY_ORDER] + [_pct(stats, "Total")]

    fig_w = max(11.5, 1.25 * len(columns))
    fig, ax = plt.subplots(figsize=(fig_w, 2.2), dpi=220)
    ax.axis("off")

    total = stats.get("Total", {"episodes": 0, "successes": 0, "success_rate": 0.0})
    title = (
        f"LIBERO-Plus Eval Summary  |  "
        f"{int(total['successes'])}/{int(total['episodes'])} successes  |  "
        f"Total {100*total['success_rate']:.1f}%"
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    table = ax.table(cellText=[row], colLabels=columns, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#BFC7D5")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#EEF3FA")
            cell.set_text_props(fontweight="bold")
        elif c == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#F8FAFC")
        elif c == len(columns) - 1:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#F8FAFC")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_category_bar(stats: Dict[str, Dict[str, float]], output_path: str | os.PathLike, model_name: str = "Model") -> None:
    cats = [c for c in CATEGORY_ORDER if c in stats and stats[c]["episodes"] > 0]
    rates = [100 * stats[c]["success_rate"] for c in cats]
    labels = [f"{c}\n{int(stats[c]['successes'])}/{int(stats[c]['episodes'])}" for c in cats]

    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=180)
    bars = ax.bar(labels, rates)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Success rate (%)")
    ax.set_title(f"LIBERO-Plus Success Rate by Category — {model_name}", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, min(rate + 1.5, 103), f"{rate:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_curve(records: List[Dict[str, object]], output_path: str | os.PathLike, model_name: str = "Model") -> None:
    xs: List[int] = []
    ys: List[float] = []
    successes = 0
    for idx, r in enumerate(records, start=1):
        successes += int(bool(r.get("success", False)))
        xs.append(idx)
        ys.append(100 * successes / idx)

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    ax.plot(xs, ys, linewidth=1.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative success rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)
    final = ys[-1] if ys else 0.0
    ax.set_title(f"Cumulative Eval Success Rate — {model_name} ({final:.1f}%)", fontweight="bold")
    if xs:
        ax.text(xs[-1], final, f"  {final:.1f}%", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_eval_report_from_log(
    log_path: str | os.PathLike,
    output_dir: str | os.PathLike = "eval_figures",
    model_name: Optional[str] = None,
) -> Dict[str, str]:
    """Parse a log and save summary images/CSVs. Returns output file paths."""
    log_path = str(log_path)
    records = parse_eval_log(log_path)
    if not records:
        raise ValueError(f"No episode records found in log: {log_path}")

    stats = summarize(records)
    if model_name is None:
        model_name = Path(log_path).stem

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "summary_table": str(output_dir / "eval_summary_table.png"),
        "category_bar": str(output_dir / "eval_category_bar.png"),
        "cumulative_curve": str(output_dir / "eval_cumulative_curve.png"),
        "summary_csv": str(output_dir / "eval_summary.csv"),
        "episodes_csv": str(output_dir / "eval_episodes.csv"),
        "failures_csv": str(output_dir / "eval_failures.csv"),
    }

    plot_summary_table(stats, outputs["summary_table"], model_name=model_name)
    plot_category_bar(stats, outputs["category_bar"], model_name=model_name)
    plot_cumulative_curve(records, outputs["cumulative_curve"], model_name=model_name)
    save_summary_csv(stats, outputs["summary_csv"])
    save_episode_csv(records, outputs["episodes_csv"])
    save_failures_csv(records, outputs["failures_csv"])

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LIBERO / LIBERO-Plus eval log to images and CSV files.")
    parser.add_argument("--log_path", required=True, help="Path to EVAL-*.txt log file")
    parser.add_argument("--output_dir", default="eval_figures", help="Directory for output images/CSVs")
    parser.add_argument("--model_name", default=None, help="Name shown in the summary images")
    args = parser.parse_args()

    outputs = save_eval_report_from_log(args.log_path, args.output_dir, args.model_name)
    print("Saved eval report files:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
