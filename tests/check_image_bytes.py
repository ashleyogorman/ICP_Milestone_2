import argparse
import io
from collections import Counter
 
import numpy as np
from datasets import Image as HFImage, load_dataset
from PIL import Image
 
DATASET_NAME = "lansinuote/ocr_id_card"
 
 
def file_format(b: bytes) -> str:
    if b.startswith(b"\x89PNG"):
        return "PNG"
    if b.startswith(b"\xff\xd8"):
        return "JPEG"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "WEBP"
    return f"unknown ({b[:4]!r})"
 
 
def decode(b: bytes, convert_rgb: bool = False) -> np.ndarray:
    img = Image.open(io.BytesIO(b))
    return np.array(img.convert("RGB") if convert_rgb else img)
 
 
def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="check the first N cards")
    parser.add_argument("--cards", type=int, nargs="*", help="check these card indices instead")
    parser.add_argument("--revision", default=None, help="dataset revision hash")
    args = parser.parse_args()
 
    ds = load_dataset(DATASET_NAME, split="train", revision=args.revision)
    raw = ds.cast_column("image", HFImage(decode=False))
    cards = args.cards if args.cards else list(range(args.n))
 
    formats, modes = Counter(), Counter()
    ok = Counter()
    for i in cards:
        reference = np.array(ds[i]["image"])          # what the offline evaluation used
        original = raw[i]["image"]["bytes"]
        formats[file_format(original)] += 1
        modes[Image.open(io.BytesIO(original)).mode] += 1
 
        ok["original bytes, plain decode"] += np.array_equal(decode(original), reference)
        ok["original bytes, .convert('RGB')"] += np.array_equal(decode(original, convert_rgb=True), reference)
        ok["re-encoded PNG, plain decode"] += np.array_equal(decode(png_bytes(reference)), reference)
 
    print(f"Checked {len(cards)} cards")
    print(f"Stored format: {dict(formats)}")
    print(f"Image mode:    {dict(modes)}  (offline arrays have shape like {np.array(ds[cards[0]]['image']).shape})")
    print()
    for name, count in ok.items():
        status = "OK " if count == len(cards) else "BAD"
        print(f"  [{status}] {name}: {count}/{len(cards)} match the offline pixels")
    print()
    print("Use any option marked OK. Prefer 'original bytes, plain decode' (no re-encoding).")
 
 
if __name__ == "__main__":
    main()