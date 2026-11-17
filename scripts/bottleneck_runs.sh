#!/bin/bash

set -e
cd "$(dirname "$0")/.."

OUT=results/bottleneck_runs
DURATION=300
SCREENSHOT_AT=210   
RUNS=(
    "tesseract-fast-runs 1.6"
    "easyocr-runs 0.1" 
)
mkdir -p "$OUT"

kubectl -n monitoring port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090 >/dev/null 2>&1 &
PORT_FORWARD=$!
trap 'kill $PORT_FORWARD 2>/dev/null' EXIT
sleep 3

for run in "${RUNS[@]}"; do
    read -r variant rate <<< "$run"
    echo "== $variant at $rate cards/s, 1 replica x 1 core, batch 1, ${DURATION}s"
    python3 scripts/scale_ocr.py -v "$variant" -r 1 -c 1 -m 1
    pod=$(kubectl get pod -l app=ocr --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
    echo "   pod $pod"

    (sleep $((10 + SCREENSHOT_AT)); echo; echo ">>> SCREENSHOT NOW: $variant (Grafana, last 5 minutes)"; echo) &
    reminder=$!
    docker exec icp bash -c "cd /icp-milestone-2 && PYTHONPATH=.:service python3 loadgenerator/load_generator.py --rate $rate --duration $DURATION" \
        | tail -1 | tee "$OUT/${variant}_load_summary.json"
    kill $reminder 2>/dev/null || true

    window=$((DURATION + 150))s
    query="1000 * sum by (step) (increase(icp_step_duration_seconds_sum{pod=\"$pod\"}[$window])) / sum by (step) (increase(icp_step_duration_seconds_count{pod=\"$pod\"}[$window]))"
    curl -sG http://localhost:9090/api/v1/query --data-urlencode "query=$query" > "$OUT/${variant}_steps.json"
done

python3 - "$OUT" "${RUNS[@]}" <<'EOF'
import csv, json, sys
out, runs = sys.argv[1], [r.split() for r in sys.argv[2:]]
rows = []
for variant, rate in runs:
    result = json.load(open(f"{out}/{variant}_steps.json"))["data"]["result"]
    steps = {r["metric"]["step"]: float(r["value"][1]) for r in result}
    total = sum(steps.values())
    for step in ("decode", "preprocess", "ocr", "fields_assertions"):
        ms = steps.get(step, float("nan"))
        rows.append({"variant": variant, "rate_rps": rate, "replicas": 1, "cores_per_replica": 1, "batch": 1,
                     "step": step, "mean_ms": round(ms, 2), "share (%)": round(100 * ms / total, 2) if total else ""})
with open(f"{out}/steps.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
for r in rows:
    print(f"{r['variant']:22} {r['step']:18} {r['mean_ms']:>10} ms  {r['share (%)']}%")
EOF
