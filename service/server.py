"""
Supports a configurable batch size (requests wait in a queue up to a maximum batch size or a timeout)
expose a Prometheus /metrics endpoint with at least: request count, latency histogram, queue length,
current batch size, and an assertion pass/fail counter


Env vars used:
    VARIANT            from load_variant
    MAX_BATCH          max requests per batch, default 1
    BATCH_TIMEOUT_MS   max wait for a batch to fill, default 10
    CORES              CPU cores this replica may use (thread count), default 1

"""
import os

# set pool before imports 
CORES = int(os.environ.get("CORES", "1"))
for var in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, str(CORES))
 
import asyncio
import io
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import grpc
import numpy as np
from PIL import Image
import icp_pb2
import icp_pb2_grpc
from icp.preprocess import preprocess, unrotate_boxes
from icp.fields import assign_fields
from icp.assertions import check_card
from load_variant import load_variant
import logging
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("icp")

MAX_BATCH = int(os.environ.get("MAX_BATCH", "1"))
BATCH_TIMEOUT_MS = float(os.environ.get("BATCH_TIMEOUT_MS", "10"))
PORT = 50051
 
cv2.setNumThreads(CORES) 

# request count, latency histogram, queue length, current batch size, and an assertion pass/fail counter.
REQUEST_COUNT = Counter('icp_requests_total', 'Total ICP Requests')
LAT_HIST = Histogram(
    'icp_request_duration_seconds', 
    'Time spent processing request',
    buckets=(0.5, 1, 2, 4, 8, 16, 32, 64)
)
QUEUE_LENGTH = Gauge("icp_queue_length", "Requests waiting in the queue")
CURR_BATCH_SIZE = Gauge("icp_current_batch_size", "Current size of batch")
ASSERTIONS = Counter("icp_assertions_total", "Cards checked by assertions", ["result"])
ASSERTION_CHECKS = Counter("icp_assertion_checks_total", "Individual assertion results", ["assertion", "result"])
# ASSERTION_CHECKS.labels(assertion="A2", result="fail").inc()

# running the pipeline function
def run_card(variant, image_bytes): # varient from load variant and 
    image = np.array(Image.open(io.BytesIO(image_bytes)))
    x, rotation = preprocess(image, variant["enhancement"], variant["channel"])
    detections = variant["ocr"](x)
    if rotation is not None:
        detections = unrotate_boxes(detections, rotation)
    checks = check_card(assign_fields(detections))
    for name, passed in checks.items():
        ASSERTION_CHECKS.labels(assertion=name, result="pass" if passed else "fail").inc()
    ASSERTIONS.labels(result="pass" if all(checks.values()) else "fail").inc()
    return detections

# function for calling run_card on batches
def run_batch(variant, batch): # batch is built in Recognize, is queued reqs
    results = []
    for item in batch:
        try: # try catch just in case of typo etc and to not kill the batch request 
            results.append(run_card(variant, item["request"].image))
        except Exception as e:
            log.exception("card %s failed", item["request"].request_id)
            results.append(e)
    return results
 
async def batch_worker(queue, variant, executor):
    log.info("batch worker started")
    loop = asyncio.get_running_loop()
    while True:
        # fill until max batch or timeout
        # get request -> check time vs timeout -> check batch size -> if good add request
        batch = [await queue.get()]
        
        deadline = loop.time() + BATCH_TIMEOUT_MS / 1000
        while len(batch) < MAX_BATCH and loop.time() < deadline:
            try:
                batch.append(await asyncio.wait_for(queue.get(), deadline - loop.time()))
            except asyncio.TimeoutError:
                break

        CURR_BATCH_SIZE.set(len(batch))
        
        # track inference time
        start = time.perf_counter()
        results = await loop.run_in_executor(executor, run_batch, variant, batch)
        inference_ms = (time.perf_counter() - start) * 1000

        log.info("batch of %d done in %.1f ms", len(batch), inference_ms) 
        
        # loop through each item in patch and results, catch exceptions if there was one and otherwise give it its result
        for item, result in zip(batch, results):
            if isinstance(result, Exception):
                item["future"].set_exception(result)
            else:
                item["future"].set_result((result, len(batch), (start - item["enqueued"]) * 1000, inference_ms))
 
 
class OcrServicer(icp_pb2_grpc.OcrServiceServicer):
    def __init__(self, queue, variant_name):
        self.queue = queue
        self.variant_name = variant_name
 
    async def Recognize(self, request, context):
        future = asyncio.get_running_loop().create_future()
        enqueued = time.perf_counter()
        await self.queue.put({"request": request, "future": future, "enqueued": enqueued})
        try:
            detections, batch_size, queue_ms, inference_ms = await future
        except Exception as e:
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

        LAT_HIST.observe(time.perf_counter() - enqueued)
        REQUEST_COUNT.inc()

        
        # put into CardResponse from .proto
        return icp_pb2.CardResponse(
            request_id=request.request_id,
            detections=[
                icp_pb2.Detection(x1=d["box"][0], y1=d["box"][1], x2=d["box"][2], y2=d["box"][3],
                                  text=d["word"], conf=d["conf"])
                for d in detections
            ],
            variant=self.variant_name,
            batch_size=batch_size,
            queue_ms=queue_ms,
            inference_ms=inference_ms,
        )
 
 
async def main():
    variant = load_variant()
    queue = asyncio.Queue()
    QUEUE_LENGTH.set_function(queue.qsize)

    # one worker thread so can keep a queue
    executor = ThreadPoolExecutor(max_workers=1)
 
    server = grpc.aio.server()
    icp_pb2_grpc.add_OcrServiceServicer_to_server(OcrServicer(queue, variant["name"]), server)
    server.add_insecure_port(f"[::]:{PORT}")
    start_http_server(8000)
    await server.start()
    asyncio.create_task(batch_worker(queue, variant, executor))
    log.info(f"serving {variant['name']} (max_batch={MAX_BATCH}, timeout={BATCH_TIMEOUT_MS}ms, cores={CORES})")
    await server.wait_for_termination()
 
 
if __name__ == "__main__":
    asyncio.run(main())
