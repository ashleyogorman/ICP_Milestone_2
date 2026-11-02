import json, numpy as np
from datasets import load_dataset
from icp.preprocess import preprocess, unrotate_boxes
from icp.ocr import build_ocr

ds = load_dataset("lansinuote/ocr_id_card", split="train")
image = np.array(ds[5642]["image"])

x, rotation = preprocess(image, "+.25 contrast", "blue")
ocr = build_ocr("easyocr", {"quantize": True}, model_dir="/root/.EasyOCR/model")
result = ocr(x)
if rotation is not None:
    result = unrotate_boxes(result, rotation)

cached = next(json.loads(l) for l in open("results/ocr_cache/test.jsonl")
              if '"card": 5642' in l and '"+.25 contrast"' in l and '"blue"' in l and '"ocr": "easyocr"' in l)
print([r["word"] for r in result] == [r["word"] for r in cached["result"]])
print(result == cached["result"])