import json
import os

import pandas as pd
from datasets import load_dataset

from icp.assertions import check_card
from icp.fields import FIELD_CLASSES, assign_fields, assign_fields_from_labels

OUT_DIR = "results/T3_online_assertions"
CHECKED_FIELDS = ["gender", "year", "month", "day", "id_number"]

splits = json.load(open("results/splits/splits.json"))
dev_ids = set(splits["dev"])
labels = load_dataset("lansinuote/ocr_id_card", split="train", revision=splits["revision"]).select_columns(["ocr"])["ocr"]
os.makedirs(OUT_DIR, exist_ok=True)

truth = pd.DataFrame([check_card({FIELD_CLASSES[t["cls"]]: t["word"] for t in card}) for card in labels])
truth_table = pd.DataFrame({"cards passing": truth.sum(), "cards failing": (~truth).sum(), "pass rate (%)": (100 * truth.mean()).round(2)})
truth_table.to_csv(f"{OUT_DIR}/ground_truth_through_check_card.csv")
print(truth_table, "\n")

rows = []
with open("results/ocr_cache/dev.jsonl", encoding="utf-8") as f:
    for line in f:
        run = json.loads(line)
        if run["card"] not in dev_ids:
            continue
        online = assign_fields(run["result"])
        offline = assign_fields_from_labels(run["result"], labels[run["card"]])
        online_checks, offline_checks = check_card(online), check_card(offline)
        rows.append({
            "enhancement": run["enhancement"], "channel": run["channel"], "ocr": run["ocr"], "card": run["card"],
            **{f"{k}_same_text": online[k] == offline.get(k, "") for k in CHECKED_FIELDS},
            **{f"{a}_online": online_checks[a] for a in online_checks},
            **{f"{a}_offline": offline_checks[a] for a in offline_checks},
            "all_online": all(online_checks.values()),
            "all_offline": all(offline_checks.values()),
        })
cards = pd.DataFrame(rows)

columns = {}
for k in CHECKED_FIELDS:
    columns[f"{k} same text (%)"] = (k + "_same_text", "same")
for a in ["A1", "A2", "A3", "A4", "all"]:
    columns[f"{a} online pass (%)"] = (f"{a}_online", "online")
    columns[f"{a} offline pass (%)"] = (f"{a}_offline", "offline")
    columns[f"{a} agree (%)"] = (a, "agree")


def summarize(group: pd.DataFrame) -> pd.Series:
    out = {"cards": len(group)}
    for name, (col, kind) in columns.items():
        if kind == "agree":
            out[name] = 100 * (group[f"{col}_online"] == group[f"{col}_offline"]).mean()
        else:
            out[name] = 100 * group[col].mean()
    return pd.Series(out)


per_config = cards.groupby(["enhancement", "channel", "ocr"], sort=False).apply(summarize).round(2).reset_index()
per_config.to_csv(f"{OUT_DIR}/dev_online_vs_offline_per_config.csv", index=False)
overall = summarize(cards).round(2)
overall.to_csv(f"{OUT_DIR}/dev_online_vs_offline_overall.csv", header=["value"])

per_config_rates = per_config[["all online pass (%)", "all offline pass (%)"]]
agreement = pd.DataFrame({
    "configurations": [len(per_config)],
    "pearson": [per_config_rates.iloc[:, 0].corr(per_config_rates.iloc[:, 1])],
    "spearman": [per_config_rates.iloc[:, 0].corr(per_config_rates.iloc[:, 1], method="spearman")],
}).round(3)
agreement.to_csv(f"{OUT_DIR}/dev_online_vs_offline_correlation.csv", index=False)
print(overall.to_string(), "\n")
print(agreement.to_string(index=False))
