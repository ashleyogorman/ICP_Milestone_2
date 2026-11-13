import datetime
import re
import unicodedata

ASSERTION_NAMES = {
    "A1": "id_format",
    "A2": "id_matches_date",
    "A3": "valid_date",
    "A4": "valid_gender",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).replace("x", "X")


def check_card(fields: dict) -> dict:
    id_number, year, month, day, gender = (normalize(fields.get(k, "")) for k in ("id_number", "year", "month", "day", "gender"))
    id_format = re.fullmatch(r"\d{17}[\dX]", id_number) is not None
    date_digits = all(re.fullmatch(r"\d+", v) for v in (year, month, day))
    try:
        valid_date = date_digits and datetime.date(1900, 1, 1) <= datetime.date(int(year), int(month), int(day)) <= datetime.date(2030, 12, 31)
    except (ValueError, OverflowError):
        valid_date = False
    return {
        "A1": id_format,
        "A2": id_format and date_digits and id_number[6:14] == year + month.zfill(2) + day.zfill(2),
        "A3": valid_date,
        "A4": gender in ("男", "女"),
    }
