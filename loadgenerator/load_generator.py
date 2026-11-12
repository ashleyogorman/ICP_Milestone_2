import argparse
import asyncio
import json
import os
import time
import grpc
import numpy as np
from datasets import Image as HFImage, load_dataset
import icp_pb2
import icp_pb2_grpc

SPLIT_PATH = "results/splits/splits.json"
TARGET = "dns://10.43.0.10:53/ocr-headless.default.svc.cluster.local:50051"
POISSON_SEED = 35
OUT_DIR = "results/T5_load_tests"
WARMUP_S = 10 # the first 10s of each run are discarded e.g. round robin connect



def parse_args():
    parser = argparse.ArgumentParser(description="Open-loop load generator for the OCR service")
    parser.add_argument("--rate", type=float, required=True, help="Target rate, requests per second")
    parser.add_argument("--duration", type=float, default=60, help="Seconds of measured load after warm-up")
    parser.add_argument("--routing", choices=["round_robin"], default="round_robin", help="How requests are spread across replicas")
    return parser.parse_args()


def load_cards():
    splits = json.load(open(SPLIT_PATH))
    ds = load_dataset(splits["dataset"], split="train", revision=splits["revision"])
    raw = ds.cast_column("image", HFImage(decode=False))
    pool = splits["load_pool"]
    position = 0
    while True:
        card = pool[position]
        yield card, raw[card]["image"]["bytes"]
        position = (position + 1) % len(pool)


async def send_one(stub, card, png, rate, sent_at, record, results):
    start = time.perf_counter()
    try:
        response = await stub.Recognize(icp_pb2.CardRequest(request_id=str(card), image=png))
        latency_ms = (time.perf_counter() - start) * 1000
        row = {
            "rate_rps": rate, "card": card, "sent_at": round(sent_at, 3),
            "latency_ms": round(latency_ms, 1),
            "queue_ms": round(response.queue_ms, 1),
            "inference_ms": round(response.inference_ms, 1),
            "batch_size": response.batch_size, "variant": response.variant, "ok": True,
            "detections": [{"box": [d.x1, d.y1, d.x2, d.y2], "word": d.text, "conf": d.conf}
                           for d in response.detections],
        }
    except grpc.aio.AioRpcError as e:
        latency_ms = (time.perf_counter() - start) * 1000
        row = {"rate_rps": rate, "card": card, "sent_at": round(sent_at, 3),
               "latency_ms": round(latency_ms, 1), "ok": False, "error": e.code().name}
    if record:
        results.append(row)


async def main():
    args = parse_args()
    run = f"rate{args.rate:g}_dur{args.duration:g}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = os.path.join(OUT_DIR, f"{run}.jsonl")

    cards = load_cards()
    rng = np.random.default_rng(POISSON_SEED)
    results = []

    options = [("grpc.lb_policy_name", "round_robin")]
    async with grpc.aio.insecure_channel(TARGET, options=options) as channel:
        await channel.channel_ready()
        await asyncio.sleep(1)  # let round-robin connect to every replica
        stub = icp_pb2_grpc.OcrServiceStub(channel)

        loop = asyncio.get_running_loop()
        start = loop.time()
        end_warmup = start + WARMUP_S
        end = end_warmup + args.duration
        tasks = []
        while loop.time() < end:
            card, png = next(cards)
            record = loop.time() >= end_warmup
            # send result at rate regardless of previous
            tasks.append(asyncio.create_task(
                send_one(stub, card, png, args.rate, time.time(), record, results)))
            await asyncio.sleep(rng.exponential(1 / args.rate))  # poisson arrivals

        await asyncio.wait(tasks, timeout=120)
        unfinished = sum(not t.done() for t in tasks)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out, "w") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ok = [r["latency_ms"] for r in results if r["ok"]]
    summary = {
        "run": run, "out": out, "target_rate": args.rate, "seed": POISSON_SEED,
        "recorded": len(results), "errors": sum(not r["ok"] for r in results), "unfinished": unfinished,
        "achieved_rate": round(len(results) / args.duration, 3),
        "p50_ms": round(float(np.percentile(ok, 50)), 1) if ok else None,
        "p95_ms": round(float(np.percentile(ok, 95)), 1) if ok else None,
        "p99_ms": round(float(np.percentile(ok, 99)), 1) if ok else None,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    asyncio.run(main())