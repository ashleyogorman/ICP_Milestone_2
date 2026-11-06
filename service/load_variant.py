"""Load the OCR variant this service should serve from the registry

env vars:

    VARIANT           variant name in the registry, e.g. easyocr-runs   (required)
    VARIANT_VERSION   version folder, default "1"
    ENHANCEMENT       optional override
    CHANNEL           optional override

Registry layout:
    icp-ocr-variants/<name>/<version>/<model files> + metadata.json
"""
import json
import os
from minio import Minio
from icp.ocr import build_ocr
 
MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
REGISTRY_BUCKET = "icp-ocr-variants"
VARIANT_VERSION = "1"
MODEL_DIR = "/models"
 
 
def load_variant():
    name = os.environ["VARIANT"]
    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
 
    # download files
    prefix = f"{name}/{VARIANT_VERSION}/"
    os.makedirs(MODEL_DIR, exist_ok=True)
    downloaded = []
    for obj in client.list_objects(REGISTRY_BUCKET, prefix=prefix, recursive=True):
        filename = obj.object_name[len(prefix):]
        client.fget_object(REGISTRY_BUCKET, obj.object_name, os.path.join(MODEL_DIR, filename))
        downloaded.append(filename)
    if "metadata.json" not in downloaded:
        raise RuntimeError(f"variant {name!r} not found in bucket {REGISTRY_BUCKET!r}")
 
    # get variant metadata
    with open(os.path.join(MODEL_DIR, "metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
 
    # get the enhancment and channel and check for overrides 
    default_enhancement, default_channel = metadata["preprocessing"].rsplit(" + ", 1)
    enhancement = os.environ.get("ENHANCEMENT", default_enhancement)
    channel = os.environ.get("CHANNEL", default_channel)
 
    # ocr that the files downloaded
    ocr = build_ocr(metadata["engine"], metadata["config"], MODEL_DIR)
 
    print(f"Loaded variant {name} ({metadata['engine']}), "
          f"preprocessing: {enhancement} + {channel}, files: {sorted(downloaded)}", flush=True)
    return {
        "name": name,
        "ocr": ocr,
        "enhancement": enhancement,
        "channel": channel,
        "metadata": metadata,
    }
 
 
if __name__ == "__main__":
    load_variant()