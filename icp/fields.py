import re

from icp.assertions import normalize

FIELD_CLASSES = ["name", "gender", "ethnicity", "year", "month", "day", "address", "id_number"]


def reading_order(detections: list[dict]) -> list[dict]:
    lines = []
    for d in sorted(detections, key=lambda d: (d["box"][1] + d["box"][3]) / 2):
        center = (d["box"][1] + d["box"][3]) / 2
        if lines and abs(center - lines[-1][0]) <= (d["box"][3] - d["box"][1]) / 2:
            lines[-1][1].append(d)
        else:
            lines.append((center, [d]))
    return [d for _, line in lines for d in sorted(line, key=lambda d: d["box"][0])]


def box_overlap(a, b) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return width * height / smaller if smaller > 0 else 0.0


def assign_fields_from_labels(detections: list[dict], labels: list[dict]) -> dict:
    assigned = {t["cls"]: [] for t in labels}
    for d in detections:
        best_overlap, best_cls = max((box_overlap(d["box"], t["box"]), t["cls"]) for t in labels)
        if best_overlap >= 0.5:
            assigned[best_cls].append(d)
    return {FIELD_CLASSES[cls]: normalize("".join(d["word"] for d in reading_order(ds))) for cls, ds in assigned.items()}


def assign_fields(detections: list[dict]) -> dict:
    tokens = [t for d in reading_order(detections) for t in re.findall(r"\d[\dX]*|\S", normalize(d["word"]))]
    numbers = [i for i, t in enumerate(tokens) if t[0].isdigit()]
    id_index = max((i for i in numbers if len(tokens[i]) > 4), key=lambda i: len(tokens[i]), default=None)
    date_parts = [tokens[i] for i in numbers if i != id_index and len(tokens[i]) <= 4]
    year_at = next((i for i, t in enumerate(date_parts) if len(t) == 4), None)
    year, month, day = (date_parts[year_at:year_at + 3] + ["", "", ""])[:3] if year_at is not None else ("", "", "")
    return {
        "gender": next((t for t in tokens if t in ("男", "女")), ""),
        "year": year,
        "month": month,
        "day": day,
        "id_number": tokens[id_index] if id_index is not None else "",
    }
