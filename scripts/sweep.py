"""T6 profiling sweep: for every (variant, resource config, batch size)

Run on VM, with container `icp` running:
    python3 profiling/sweep.py                      # full sweep
    python3 profiling/sweep.py --only easyocr_r1    # just points whose name contains this
    
points already marked done in summary.jsonl are skipped
"""

import argparse
import itertools
import json
import os
import subprocess
from scale_ocr import config_scale
 
SLO_MS = json.load(open("results/T5_sla_calibration/sla.json"))["SLO_ms"]
RAW_DIR = "results/T6_profiling_raw"
SUMMARY = os.path.join(RAW_DIR, "summary.jsonl")
 
PROBE_DURATION_S = 120
SEARCH_STEPS = 3
MAX_DOUBLINGS = 6
BATCH_TIMEOUT_MS = 10
 
# start_rate_per_core is guess of request per sec one core can handle
VARIANTS = {
    "easyocr":        {"start_rate_per_core": 0.1},
    "easyocr-runs":   {"start_rate_per_core": 0.1},
    "tesseract-runs": {"start_rate_per_core": 0.5},
    "tesseract-fast": {"start_rate_per_core": 0.5},
}
 
RESOURCES = {
    "r1_c1": {"replicas": 1, "cores": 1},
    "r2_c1": {"replicas": 2, "cores": 1},
    "r2_c2": {"replicas": 2, "cores": 2},
    "r4_c1": {"replicas": 4, "cores": 1},
    "r1_c4": {"replicas": 1, "cores": 4},
}
 
BATCHES = {
    "b1": 1,
    "b4": 4,
    "b8": 8,
}
 
 
def append_summary(row):
    with open(SUMMARY, "a") as f:
        f.write(json.dumps(row) + "\n")
 
 
def finished_points():
    if not os.path.exists(SUMMARY):
        return set()
    with open(SUMMARY) as f:
        rows = [json.loads(line) for line in f]
    return {row["point"] for row in rows if row.get("done")}
 
 
def run_load(rate):
    # runs the load generator and returns summary
    command = (f"cd /icp-milestone-2 && PYTHONPATH=.:service python3 loadgenerator/load_generator.py "
               f"--rate {rate} --duration {PROBE_DURATION_S}")
    result = subprocess.run(["docker", "exec", "icp", "bash", "-c", command],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])
 
 
def probe(point, info, rate):
    # runs the load test at the rate
    summary = run_load(rate)
    passed = (summary["p95_ms"] is not None and summary["p95_ms"] <= SLO_MS
              and summary["errors"] == 0 and summary["unfinished"] == 0)
    append_summary({
        "point": point, **info, "rate": rate, "achieved_rate": summary["achieved_rate"],
        "recorded": summary["recorded"], "errors": summary["errors"],
        "p50_ms": summary["p50_ms"], "p95_ms": summary["p95_ms"], "p99_ms": summary["p99_ms"],
        "passed": passed, "raw": summary["out"],
    })
    print(f"  {point} at {rate:g} req/s: p95 {summary['p95_ms']} ms, {'pass' if passed else 'fail'}", flush=True)
    return passed
 
 
def find_max_rate(point, info, start_rate):
    # double the rate until it fails then binary search between the last pass and the first fail
    low, high = 0.0, start_rate
    for _ in range(MAX_DOUBLINGS):
        if not probe(point, info, high):
            break
        low, high = high, round(high * 2, 3)
    else:
        return low
 
    for _ in range(SEARCH_STEPS):
        mid = round((low + high) / 2, 3)
        if probe(point, info, mid):
            low = mid
        else:
            high = mid
    return low
 
 
def main():
    parser = argparse.ArgumentParser(description="T6 profiling sweep")
    parser.add_argument("--only", default=None, help="Run only points whose name contains this text")
    args = parser.parse_args()
 
    os.makedirs(RAW_DIR, exist_ok=True)
    done = finished_points()
 
    for variant, resource, batch in itertools.product(VARIANTS, RESOURCES, BATCHES):
        point = f"{variant}_{resource}_{batch}"
        if args.only and args.only not in point:
            continue
        if point in done:
            print(f"skipping {point}, already done")
            continue
 
        replicas = RESOURCES[resource]["replicas"]
        cores = RESOURCES[resource]["cores"]
        max_batch = BATCHES[batch]
        start_rate = round(VARIANTS[variant]["start_rate_per_core"] * replicas * cores, 3)
        info = {"variant": variant, "replicas": replicas, "cores": cores, "max_batch": max_batch}
 
        print(f"{point}: {replicas} replicas x {cores} cores, batch {max_batch}", flush=True)
        config_scale(replicas, cores, variant, max_batch, BATCH_TIMEOUT_MS)
        max_rate = find_max_rate(point, info, start_rate)
        append_summary({"point": point, **info, "done": True,
                        "max_rate_meeting_slo": max_rate, "cost_cores": replicas * cores})
        print(f"{point}: max rate meeting SLO = {max_rate} req/s", flush=True)
 
 
if __name__ == "__main__":
    main()
 