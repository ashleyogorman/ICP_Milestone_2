"""Send a few test-slice cards to the OCR server at once and print the responses.
 
Run from the repo root (server must be running):
    PYTHONPATH=. python3 service/test_client.py               # 8 cards to localhost:50051
    PYTHONPATH=. python3 service/test_client.py --n 16 --target localhost:50051
"""
import argparse
import asyncio
import json
import time
 
import grpc
from datasets import Image as HFImage, load_dataset
 
import icp_pb2
import icp_pb2_grpc
 
SPLITS = "results/splits/splits.json"
 
 
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="number of cards to send at once")
    parser.add_argument("--target", default="localhost:50051")
    args = parser.parse_args()
 
    splits = json.load(open(SPLITS))
    ds = load_dataset(splits["dataset"], split="train", revision=splits["revision"])
    raw = ds.cast_column("image", HFImage(decode=False))
    cards = splits["test"][: args.n]
 
    async with grpc.aio.insecure_channel(args.target) as channel:
        stub = icp_pb2_grpc.OcrServiceStub(channel)
 
        async def send(card):
            start = time.perf_counter()
            request = icp_pb2.CardRequest(request_id=str(card), image=raw[card]["image"]["bytes"])
            response = await stub.Recognize(request)
            return response, (time.perf_counter() - start) * 1000
 
        # Send all cards at once, so the server has something to batch
        results = await asyncio.gather(*(send(c) for c in cards))
 
    for response, total_ms in results:
        texts = [d.text for d in response.detections]
        network_ms = total_ms - response.queue_ms - response.inference_ms
        print(f"card {response.request_id:>5} | {response.variant} | batch {response.batch_size} | "
              f"queue {response.queue_ms:7.1f} ms | inference {response.inference_ms:7.1f} ms | "
              f"other {network_ms:5.1f} ms | {len(texts)} detections: {texts[:4]}")
 
 
if __name__ == "__main__":
    asyncio.run(main())
 