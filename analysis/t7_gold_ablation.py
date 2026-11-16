import argparse
import json
import os

import pandas as pd
from datasets import load_dataset

from icp.assertions import check_card, normalize
from icp.fields import FIELD_CLASSES, assign_fields, assign_fields_from_labels

OUT = "results/T7_end_to_end/gold_ablation.csv"
CHECKED_FIELDS = ["gender", "year", "month", "day", "id_number"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="easyocr-runs")
    parser.add_argument("--enhancement", default="+.25 contrast")
    parser.add_argument("--channel", default="blue")
    return parser.parse_args()


def score(card_results, labels):
    rows = []
    for card, detections in card_results:
        target = labels[card]
        truth = {FIELD_CLASSES[t["cls"]]: normalize(t["word"]) for t in target}
        offline = assign_fields_from_labels(detections, target)
        online = assign_fields(detections)
        rows.append({
            "fields_correct": sum(offline[k] == truth[k] for k in truth if truth[k]),
            "fields_scored": sum(1 for k in truth if truth[k]),
            "assertions_pass": all(check_card(offline).values()),
            "online_assertions_pass": all(check_card(online).values()),
            "online_fields_correct": sum(online[k] == truth[k] for k in CHECKED_FIELDS),
        })
    df = pd.DataFrame(rows)
    return {
        "cards": len(df),
        "field exact match (%)": round(100 * df["fields_correct"].sum() / df["fields_scored"].sum(), 2),
        "assertion pass rate (%)": round(100 * df["assertions_pass"].mean(), 2),
        "online assertion pass rate (%)": round(100 * df["online_assertions_pass"].mean(), 2),
        "online fields correct, 5 checked fields (%)": round(100 * df["online_fields_correct"].sum() / (5 * len(df)), 2),
    }


def main():
    args = parse_args()
    splits = json.load(open("results/splits/splits.json"))
    test_ids = splits["test"]
    labels = load_dataset(splits["dataset"], split="train", revision=splits["revision"]).select_columns(["ocr"])["ocr"]

    cached = {}
    with open("results/ocr_cache/test.jsonl", encoding="utf-8") as f:
        for line in f:
            run = json.loads(line)
            if run["ocr"] == args.variant and (run["enhancement"], run["channel"]) in {(args.enhancement, args.channel), ("none", "none")}:
                cached[(run["enhancement"], run["channel"], run["card"])] = run["result"]

    gold = [(card, [{"box": t["box"], "word": t["word"], "conf": 1.0} for t in labels[card]]) for card in test_ids]
    pipeline = [(card, cached[(args.enhancement, args.channel, card)]) for card in test_ids]
    rows = [
        {"input to field assignment and assertions": "gold: ground-truth text and boxes", **score(gold, labels)},
        {"input to field assignment and assertions": f"pipeline: {args.enhancement} + {args.channel} + {args.variant}", **score(pipeline, labels)},
    ]
    if all(("none", "none", card) in cached for card in test_ids):
        no_pre = [(card, cached[("none", "none", card)]) for card in test_ids]
        rows.append({"input to field assignment and assertions": f"pipeline without preprocessing: none + none + {args.variant}", **score(no_pre, labels)})

    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
