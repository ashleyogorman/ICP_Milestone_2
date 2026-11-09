"""Fill the MinIO model registry with the OCR variants in variant_registry.csv
 
this is a standalone version of notebook cells 69 - 72
 
Run from root inside container:
    MINIO_ENDPOINT=minio:9000 python3 setup/upload_variants.py
    python3 setup/upload_variants.py --force
 
Registry layout (what service/load_variant.py expects):
    icp-ocr-variants/<name>/<version>/<model files> + metadata.json
"""
import argparse
import glob
import io
import json
import os
import urllib.request
import pandas as pd
from minio import Minio
from minio.error import S3Error
 
REGISTRY_CSV = "registry/variant_registry.csv"
BUCKET = os.environ.get("REGISTRY_BUCKET", "icp-ocr-variants")
MODEL_DIR = "models"
 
TESSDATA_SOURCES = {
    "fast": "https://github.com/tesseract-ocr/tessdata_fast/raw/main/chi_sim.traineddata",
    "best": "https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_sim.traineddata",
}
EASYOCR_MODEL_DIR = os.path.expanduser("~/.EasyOCR/model")
 
 
def tessdata_dir(kind: str) -> str:
    """Folder holding chi_sim.traineddata for 'system', 'fast' or 'best' (downloads if needed)."""
    if kind == "system":
        folder = os.environ.get("TESSDATA_PREFIX") or next(iter(glob.glob("/usr/share/tesseract-ocr/*/tessdata")), "")
        if not os.path.exists(os.path.join(folder, "chi_sim.traineddata")):
            raise FileNotFoundError("system chi_sim.traineddata not found; is tesseract-ocr-chi-sim installed?")
        return folder
    folder = os.path.join(MODEL_DIR, f"tessdata_{kind}")
    path = os.path.join(folder, "chi_sim.traineddata")
    if not os.path.exists(path):
        os.makedirs(folder, exist_ok=True)
        print(f"  downloading tessdata_{kind} ...", flush=True)
        urllib.request.urlretrieve(TESSDATA_SOURCES[kind], path)
    return folder

def easyocr_dir() -> str:
    """Folder holding the EasyOCR weights (downloads them if needed)."""
    needed = ("craft_mlt_25k.pth", "zh_sim_g2.pth")
    if not all(os.path.exists(os.path.join(EASYOCR_MODEL_DIR, f)) for f in needed):
        import easyocr
        print("  downloading EasyOCR weights ...", flush=True)
        easyocr.Reader(["ch_sim"], gpu=False, model_storage_directory=EASYOCR_MODEL_DIR)
    return EASYOCR_MODEL_DIR
 
 
def local_files(engine: str, config: dict, filenames: list[str]) -> list[str]:
    """Map the file names stored in the registry CSV to paths on this machine."""
    folder = easyocr_dir() if engine == "easyocr" else tessdata_dir(config["tessdata"])
    paths = [os.path.join(folder, f) for f in filenames]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    return paths
 
 
def exists(client: Minio, object_name: str) -> bool:
    try:
        client.stat_object(BUCKET, object_name)
        return True
    except S3Error:
        return False
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-upload variants that already exist")
    args = parser.parse_args()
 
    client = Minio(
        os.environ.get("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )
    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)
 
    registry = pd.read_csv(REGISTRY_CSV, dtype={"version": str})
    for row in registry.to_dict(orient="records"):
        name, version = row["name"], row["version"]
        prefix = f"{name}/{version}"
        if exists(client, f"{prefix}/metadata.json") and not args.force:
            print(f"{prefix}: already in registry, skipping")
            continue
 
        config = json.loads(row["config"])
        for path in local_files(row["engine"], config, json.loads(row["files"])):
            client.fput_object(BUCKET, f"{prefix}/{os.path.basename(path)}", path)
 
        metadata = json.dumps({**row, "config": config}, ensure_ascii=False).encode("utf-8")
        client.put_object(BUCKET, f"{prefix}/metadata.json", io.BytesIO(metadata), len(metadata),
                          content_type="application/json")
        print(f"{prefix}: uploaded")
 
    print(sorted(o.object_name for o in client.list_objects(BUCKET, recursive=True)))
 
 
if __name__ == "__main__":
    main()