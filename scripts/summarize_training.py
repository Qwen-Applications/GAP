"""Summarize a Gap training run for review.

Usage:
    python scripts/summarize_training.py <output_dir> [--logging-dir DIR] [--out summary]

What it does:
    1. Reads trainer_state.json (latest checkpoint or top-level).
    2. Optionally reads tensorboard event files for extra scalars.
    3. Writes:
        <out>/summary.md         — markdown report with hyperparams, key metrics, anomalies.
        <out>/loss_curves.png    — train/eval loss + learning rate.
        <out>/grad_norm.png      — grad norm over steps (anomaly hunt).
        <out>/per_metric/*.png   — one plot per scalar tag.
        <out>/last_log_tail.txt  — last 200 trainer log entries (raw).

Then zip <out> and send it to your reviewer.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_trainer_state(output_dir: Path) -> Path | None:
    direct = output_dir / "trainer_state.json"
    if direct.is_file():
        return direct
    checkpoints = sorted(
        (p for p in output_dir.glob("checkpoint-*") if (p / "trainer_state.json").is_file()),
        key=lambda p: int(p.name.split("-", 1)[1]) if p.name.split("-", 1)[1].isdigit() else -1,
    )
    if checkpoints:
        return checkpoints[-1] / "trainer_state.json"
    return None


def _load_trainer_state(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _split_log_history(log_history: list[dict]) -> dict[str, list[tuple[float, float]]]:
    """Return {metric_name: [(step, value), ...]}."""
    series: dict[str, list[tuple[float, float]]] = {}
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        for k, v in entry.items():
            if k in {"step", "epoch"} or not isinstance(v, (int, float)):
                continue
            if math.isnan(v) or math.isinf(v):
                continue
            series.setdefault(k, []).append((step, float(v)))
    for k in series:
        series[k].sort()
    return series


def _try_load_tb_events(logging_dir: Path) -> dict[str, list[tuple[float, float]]]:
    """Read tensorboard event files. Optional, returns {} if tensorboard isn't installed."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return {}
    if not logging_dir.is_dir():
        return {}
    ea = EventAccumulator(str(logging_dir), size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception as e:
        print(f"[warn] failed to read tb events: {e}", file=sys.stderr)
        return {}
    out: dict[str, list[tuple[float, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        out[tag] = [(e.step, float(e.value)) for e in events if not math.isnan(e.value)]
    return out


def _merge(*dicts: dict[str, list[tuple[float, float]]]) -> dict[str, list[tuple[float, float]]]:
    merged: dict[str, list[tuple[float, float]]] = {}
    for d in dicts:
        for k, v in d.items():
            merged.setdefault(k, []).extend(v)
    for k, vs in merged.items():
        seen: dict[float, float] = {}
        for step, val in vs:
            seen[step] = val
        merged[k] = sorted(seen.items())
    return merged


def _smooth(values: list[float], window: int = 50) -> list[float]:
    if window <= 1 or len(values) < window:
        return values
    out: list[float] = []
    s = 0.0
    q: list[float] = []
    for v in values:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def _detect_anomalies(series: dict[str, list[tuple[float, float]]]) -> list[str]:
    flags: list[str] = []
    loss = series.get("loss") or series.get("train/loss")
    if loss:
        vals = [v for _, v in loss]
        if any(math.isnan(v) or math.isinf(v) for v in vals):
            flags.append("loss contains NaN/Inf")
        if len(vals) >= 50:
            tail = vals[-50:]
            head = vals[:50]
            if sum(tail) / len(tail) > sum(head) / len(head) * 1.2:
                flags.append("train loss tail is >20% above the head — possible divergence")
        if len(vals) >= 2:
            jumps = [
                (i, vals[i] - vals[i - 1])
                for i in range(1, len(vals))
                if abs(vals[i] - vals[i - 1]) > 5 * (abs(vals[i - 1]) + 1e-6)
            ]
            if jumps:
                flags.append(f"{len(jumps)} sudden train loss jumps (>5× prior value)")

    gn = series.get("grad_norm") or series.get("train/grad_norm")
    if gn:
        vs = [v for _, v in gn]
        if vs and max(vs) > 100:
            flags.append(f"grad_norm peaked at {max(vs):.1f} (> 100, suspicious)")

    eval_loss = series.get("eval_loss") or series.get("eval/loss")
    if eval_loss and len(eval_loss) >= 3:
        vs = [v for _, v in eval_loss]
        if vs[-1] > min(vs) * 1.10:
            flags.append(
                f"eval_loss now {vs[-1]:.4f} but min was {min(vs):.4f} — likely overfitting"
            )
    return flags


def _plot(series_subset: dict[str, list[tuple[float, float]]], path: Path, title: str) -> None:
    if not series_subset:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for tag, points in series_subset.items():
        if not points:
            continue
        steps, values = zip(*points)
        ax.plot(steps, values, label=tag, alpha=0.4, linewidth=0.8)
        if len(values) >= 50:
            ax.plot(steps, _smooth(list(values), 50), label=f"{tag} (ema-50)", linewidth=1.6)
    ax.set_xlabel("step")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _format_summary(state: dict, series: dict, flags: list[str], loss_png: Path | None) -> str:
    lines: list[str] = ["# Training Run Summary", ""]
    lines.append(f"- Source: `{state.get('_source_path', 'unknown')}`")
    lines.append(f"- Global step: **{state.get('global_step', '?')}**")
    lines.append(f"- Epoch: {state.get('epoch', '?')}")
    lines.append(f"- Max steps: {state.get('max_steps', '?')}")
    if state.get("train_batch_size"):
        lines.append(f"- Train batch size (effective): {state['train_batch_size']}")
    if state.get("best_metric") is not None:
        lines.append(f"- Best metric: {state['best_metric']} @ `{state.get('best_model_checkpoint')}`")
    lines.append("")

    lines.append("## Final values per metric")
    lines.append("")
    lines.append("| metric | n_points | first | last | min | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k in sorted(series.keys()):
        pts = series[k]
        if not pts:
            continue
        vs = [v for _, v in pts]
        lines.append(
            f"| `{k}` | {len(pts)} | {vs[0]:.6g} | {vs[-1]:.6g} | {min(vs):.6g} | {max(vs):.6g} |"
        )
    lines.append("")

    if flags:
        lines.append("## ⚠ Anomalies detected")
        for f in flags:
            lines.append(f"- {f}")
        lines.append("")
    else:
        lines.append("## Anomalies")
        lines.append("None detected by automated checks.")
        lines.append("")

    if loss_png:
        lines.append(f"## Curves")
        lines.append(f"![loss curves]({loss_png.name})")
        lines.append("")

    lines.append("## Last log entries (tail)")
    lines.append("```")
    for entry in state.get("log_history", [])[-20:]:
        lines.append(json.dumps({k: v for k, v in entry.items() if not k.startswith("_")}))
    lines.append("```")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", type=Path, help="Trainer output_dir (contains checkpoint-*/)")
    ap.add_argument("--logging-dir", type=Path, default=None, help="TensorBoard logging dir")
    ap.add_argument("--out", type=Path, default=Path("./training_summary"), help="Where to write the report")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    per_metric_dir = args.out / "per_metric"
    per_metric_dir.mkdir(exist_ok=True)

    state_path = _find_trainer_state(args.output_dir)
    if state_path is None:
        print(f"No trainer_state.json found under {args.output_dir}", file=sys.stderr)
        return 1
    state = _load_trainer_state(state_path)
    state["_source_path"] = str(state_path)

    series = _split_log_history(state.get("log_history", []))

    if args.logging_dir:
        tb_series = _try_load_tb_events(args.logging_dir)
        series = _merge(series, tb_series)

    flags = _detect_anomalies(series)

    headline_keys = {
        k: series[k]
        for k in ("loss", "train/loss", "eval_loss", "eval/loss", "learning_rate", "train/learning_rate")
        if k in series
    }
    loss_png = args.out / "loss_curves.png"
    _plot(headline_keys, loss_png, "loss + lr")

    grad_keys = {k: series[k] for k in ("grad_norm", "train/grad_norm") if k in series}
    if grad_keys:
        _plot(grad_keys, args.out / "grad_norm.png", "grad norm")

    for tag, pts in series.items():
        safe = tag.replace("/", "_").replace(" ", "_")
        _plot({tag: pts}, per_metric_dir / f"{safe}.png", tag)

    (args.out / "summary.md").write_text(_format_summary(state, series, flags, loss_png if loss_png.exists() else None))

    tail_path = args.out / "last_log_tail.txt"
    with tail_path.open("w") as f:
        for entry in state.get("log_history", [])[-200:]:
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote summary to {args.out}")
    print(f"Send back: {args.out}.zip  (run: cd {args.out.parent} && zip -r {args.out.name}.zip {args.out.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
