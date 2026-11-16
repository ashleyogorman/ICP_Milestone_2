import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset

from icp.fields import FIELD_CLASSES, assign_fields_from_labels
from icp.assertions import normalize

SUMMARY = "results/T6_profiling_raw/summary.jsonl"
OUT_TABLE = "results/T6_profiling_tables/t6_table.csv"
OUT_BREAKDOWN = "results/T6_profiling_tables/t6_latency_breakdown.csv"
SLO_MS = json.load(open("results/T5_sla_calibration/sla.json"))["SLO_ms"]

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = {1: "o", 4: "s", 8: "^"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def field_exact_match(records, labels):
    correct = scored = 0
    for r in records:
        target = labels[r["card"]]
        fields = assign_fields_from_labels(r["detections"], target)
        for t in target:
            truth = normalize(t["word"])
            if truth:
                scored += 1
                correct += fields[FIELD_CLASSES[t["cls"]]] == truth
    return 100 * correct / scored if scored else np.nan


def build_table():
    splits = json.load(open("results/splits/splits.json"))
    labels = load_dataset(splits["dataset"], split="train", revision=splits["revision"]).select_columns(["ocr"])["ocr"]
    rows = read_jsonl(SUMMARY)
    probes, done = {}, {}
    for row in rows:
        if row.get("done"):
            done[row["point"]] = row
        elif os.path.exists(row["raw"]):
            probes.setdefault(row["point"], []).append(row)

    outputs = {}
    for point, point_probes in probes.items():
        for p in point_probes:
            for r in read_jsonl(p["raw"]):
                if r.get("ok"):
                    key = json.dumps([(d["word"], [round(v) for v in d["box"]]) for d in r["detections"]], ensure_ascii=False)
                    outputs.setdefault(r["variant"], {}).setdefault(r["card"], {}).setdefault(key, r)
    accuracy, identical = {}, {}
    for variant, cards in outputs.items():
        accuracy[variant] = field_exact_match([next(iter(v.values())) for v in cards.values()], labels)
        identical[variant] = 100 * sum(len(v) == 1 for v in cards.values()) / len(cards)

    table = []
    for point, finished in done.items():
        point_probes = probes.get(point, [])
        passing = [p for p in point_probes if p["passed"] and p["rate"] == finished["max_rate_meeting_slo"]]
        shown = passing[-1] if passing else min(point_probes, key=lambda p: p["rate"])
        records = [r for r in read_jsonl(shown["raw"]) if r.get("ok")]
        latency = np.array([r["latency_ms"] for r in records])
        queue = np.array([r["queue_ms"] for r in records])
        inference = np.array([r["inference_ms"] for r in records])
        table.append({
            "stage": "ocr",
            "variant": finished["variant"],
            "batch": finished["max_batch"],
            "replicas": finished["replicas"],
            "cores_per_replica": finished["cores"],
            "cost": finished["replicas"] * finished["cores"],
            "rate_shown_rps": shown["rate"],
            "p50_ms": round(float(np.percentile(latency, 50)), 1),
            "p95_ms": round(float(np.percentile(latency, 95)), 1),
            "p99_ms": round(float(np.percentile(latency, 99)), 1),
            "max_throughput_rps": finished["max_rate_meeting_slo"],
            "accuracy": round(accuracy[finished["variant"]], 2),
            "cards_scored": len(outputs[finished["variant"]]),
            "same OCR output on every setup (%)": round(identical[finished["variant"]], 1),
            "slo_attainment (%)": round(100 * float((latency <= SLO_MS).mean()), 1),
            "queue_ms": round(float(queue.mean()), 1),
            "inference_ms": round(float(inference.mean()), 1),
            "network_ms": round(float((latency - queue - inference).mean()), 1),
        })
    return pd.DataFrame(table).sort_values(["variant", "cost", "replicas", "batch"], ignore_index=True)


def style(ax, title, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel("field exact match under load (%)", color=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def scatter(table, x, xlabel, title, path):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for color, (variant, group) in zip(SERIES, table.groupby("variant", sort=True)):
        for batch, sub in group.groupby("batch"):
            ax.scatter(sub[x], sub["accuracy"], s=46, color=color, marker=MARKERS.get(batch, "o"),
                       edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.scatter([], [], s=46, color=color, label=variant)
    for batch, marker in MARKERS.items():
        ax.scatter([], [], s=46, facecolors="none", edgecolors=MUTED, marker=marker, label=f"batch {batch}")
    if x == "p95_ms":
        ax.axvline(SLO_MS, color=MUTED, linewidth=1, linestyle="--", zorder=1)
        ax.annotate(f"SLO {SLO_MS / 1000:.1f} s", (SLO_MS, ax.get_ylim()[0]), textcoords="offset points", xytext=(4, 4), fontsize=7, color=MUTED)
    style(ax, title, xlabel)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    os.makedirs("plots", exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)


def main():
    table = build_table()
    os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)
    table.to_csv(OUT_TABLE, index=False)
    table[["variant", "batch", "replicas", "cores_per_replica", "rate_shown_rps", "queue_ms", "inference_ms", "network_ms"]].to_csv(OUT_BREAKDOWN, index=False)
    scatter(table, "p95_ms", "p95 latency at the highest rate meeting the SLO (ms)", "Accuracy vs p95 latency", "plots/accuracy_vs_p95.png")
    scatter(table, "max_throughput_rps", "max throughput meeting the SLO (cards/s)", "Accuracy vs max throughput", "plots/accuracy_vs_throughput.png")
    print(table.drop(columns=["stage"]).to_string(index=False))


if __name__ == "__main__":
    main()
