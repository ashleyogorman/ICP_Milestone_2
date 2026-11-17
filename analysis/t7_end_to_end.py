import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

ACCURACY = "results/T7_end_to_end/t7_accuracy_test.csv"
T6_TABLE = "results/T6_profiling_tables/t6_table.csv"
SLA_DIR = "results/T5_sla_calibration"
OUT_TABLE = "results/T7_end_to_end/t7_table.csv"
OUT_PLOT = "plots/pareto.png"
T6_VARIANT_FOR = {"tesseract-runs": None}  # the 44 MB model was never profiled on the VM

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MEETS, MISSES = "#2a78d6", "#a8a7a2"


def target_rate():
    for path in sorted(glob.glob(os.path.join(SLA_DIR, "*.json"))):
        sla = json.load(open(path))
        for key, value in sla.items():
            if "target" in key.lower() and isinstance(value, (int, float)):
                return float(value)
    return None


def cheapest_point(points: pd.DataFrame, rate: float):
    meeting = points[points["max_throughput_rps"] >= rate]
    if meeting.empty:
        return None
    return meeting.sort_values(["cost", "p95_ms"]).iloc[0]


def pareto_front(table: pd.DataFrame) -> pd.Series:
    front = pd.Series(False, index=table.index)
    best = -1.0
    for i, row in table[table["meets_sla"]].sort_values(["cost", "field exact match (%)"], ascending=[True, False]).iterrows():
        if row["field exact match (%)"] > best:
            front[i] = True
            best = row["field exact match (%)"]
    return front


def plot(table: pd.DataFrame, rate: float):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    miss = table[table["profiled"] & ~table["meets_sla"]]
    miss = miss.assign(cost=table["cost"].max() if table["cost"].notna().any() else 0)
    meet = table[table["meets_sla"]]
    ax.scatter(miss["cost"], miss["field exact match (%)"], s=40, facecolors="none", edgecolors=MISSES, linewidths=1.5, label="misses SLA", zorder=2)
    ax.scatter(meet["cost"], meet["field exact match (%)"], s=40, color=MEETS, edgecolors=SURFACE, linewidths=2, label="meets SLA", zorder=3)
    front = table[table["pareto"]].sort_values("cost")
    ax.step(front["cost"], front["field exact match (%)"], where="post", color=MEETS, linewidth=2, zorder=1, label="Pareto front")
    middle = (table["cost"].min() + table["cost"].max()) / 2
    for _, row in front.iterrows():
        right = row["cost"] > middle
        ax.annotate(f"{row['ocr']}\n{row['enhancement']} + {row['channel']}", (row["cost"], row["field exact match (%)"]),
                    textcoords="offset points", xytext=(-8 if right else 8, -14), ha="right" if right else "left", fontsize=7, color=INK)
    ax.margins(x=0.08)
    ax.set_xlabel("cost at target rate (replicas x cores per replica)", color=MUTED)
    ax.set_ylabel("field exact match, test slice (%)", color=MUTED)
    ax.set_title(f"End-to-end configurations at {rate:g} cards/s", color=INK, loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)
    fig.savefig(OUT_PLOT, facecolor=SURFACE)


def main():
    table = pd.read_csv(ACCURACY)
    rate = target_rate()
    if not os.path.exists(T6_TABLE) or rate is None:
        table.to_csv(OUT_TABLE, index=False)
        print(f"No T6 table or SLA target yet; wrote accuracy only to {OUT_TABLE}")
        return

    t6 = pd.read_csv(T6_TABLE)
    chosen = []
    for _, row in table.iterrows():
        t6_variant = T6_VARIANT_FOR.get(row["ocr"], row["ocr"])
        profiled = t6_variant in set(t6["variant"])
        point = cheapest_point(t6[t6["variant"] == t6_variant], rate) if profiled else None
        chosen.append({
            "t6_variant": t6_variant if profiled else None,
            "profiled": profiled,
            "meets_sla": point is not None,
            "cost": None if point is None else point["cost"],
            "replicas": None if point is None else point["replicas"],
            "cores_per_replica": None if point is None else point["cores_per_replica"],
            "batch": None if point is None else point["batch"],
            "p95_ms": None if point is None else point["p95_ms"],
        })
    table = pd.concat([table, pd.DataFrame(chosen)], axis=1)
    table["target_rate_rps"] = rate
    table["pareto"] = pareto_front(table)
    table.to_csv(OUT_TABLE, index=False)
    plot(table, rate)
    print(table[["enhancement", "channel", "ocr", "field exact match (%)", "meets_sla", "cost", "pareto"]].to_string(index=False))


if __name__ == "__main__":
    main()
