"""
Step 3: OCR Step
Contract: ocr_fn(image: np.ndarray) -> list[dict]:

The final step is to actually perform actual ocr. For this step, we pass the processed numpy array of the input image to the ocr models. This uses the following OCR models:

EasyOcr

Tesseract
"""
import easyocr
import re
import unicodedata
import cv2
import numpy as np
import pytesseract
    
""" Tesseract """

# Alex
TESSERACT_LANG = "chi_sim"
TESSERACT_PSM = 11

_CJK_SPACE = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")

def ocr_tesseract(image: np.ndarray, psm: int = TESSERACT_PSM, tessdata_dir: str = None) -> list[dict]:
    data = pytesseract.image_to_data(
        image,
        lang=TESSERACT_LANG,
        config=f"--psm {psm}" + (f" --tessdata-dir {tessdata_dir}" if tessdata_dir else ""),
        output_type=pytesseract.Output.DICT,
    )
    lines = {}
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x1, y1 = data["left"][i], data["top"][i]
        x2, y2 = x1 + data["width"][i], y1 + data["height"][i]
        if key not in lines:
            lines[key] = {"words": [], "box": [x1, y1, x2, y2], "confs": []}
        line = lines[key]
        line["words"].append(word)
        line["box"] = [
            min(line["box"][0], x1),
            min(line["box"][1], y1),
            max(line["box"][2], x2),
            max(line["box"][3], y2),
        ]
        line["confs"].append(float(data["conf"][i]))

    results = []
    for line in lines.values():
        results.append({
            "box": line["box"],
            "word": _CJK_SPACE.sub("", " ".join(line["words"])),
            "conf": round(sum(line["confs"]) / len(line["confs"]) / 100, 2),
        })
    return results


""" Easy OCR """

def easyocr_with(reader, **options):
    def ocr(image: np.ndarray) -> list[dict]:
        results = []
        for bbox, text, conf in reader.readtext(image, **options):
            xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
            results.append({"box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))], "word": text, "conf": round(float(conf), 2)})
        return results
    return ocr

def resized(ocr_fn, scale: float):
    def ocr(image: np.ndarray) -> list[dict]:
        scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return [{**r, "box": [round(v / scale, 1) for v in r["box"]]} for r in ocr_fn(scaled)]
    return ocr


def split_runs(ocr_fn):
    def ocr(image: np.ndarray) -> list[dict]:
        pieces = []
        for r in ocr_fn(image):
            tokens = re.findall(r"\d[\dXx]*|\S", r["word"])
            if not tokens:
                continue
            widths = [sum(0.5 if unicodedata.east_asian_width(ch) in ("Na", "H") else 1.0 for ch in t) for t in tokens]
            x1, y1, x2, y2 = r["box"]
            step = (x2 - x1) / sum(widths)
            for token, width in zip(tokens, widths):
                pieces.append({"box": [round(x1, 1), y1, round(x1 + width * step, 1), y2], "word": token, "conf": r["conf"]})
                x1 += width * step
        return pieces
    return ocr


def build_ocr(engine: str, config: dict, model_dir: str):
    # base OCR function for the engine
    if engine == "easyocr":
        reader = easyocr.Reader(
            ["ch_sim"], gpu=False,
            quantize=config.get("quantize", True),
            model_storage_directory=model_dir,
            download_enabled=False,
        )
        options = {"width_ths": config["width_ths"]} if "width_ths" in config else {}
        ocr = easyocr_with(reader, **options)
    elif engine == "tesseract":
        psm = config.get("psm", 11)
        ocr = lambda image: ocr_tesseract(image, psm=psm, tessdata_dir=model_dir)
    else:
        raise ValueError(f"unknown engine {engine!r}")

    # apply wrappers from notebook
    if "input_scale" in config:
        ocr = resized(ocr, config["input_scale"])
    if "split" in config:
        ocr = split_runs(ocr)
    return ocr