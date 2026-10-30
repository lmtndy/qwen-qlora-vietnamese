"""
SYNTHETIC DATA GENERATION — tự sinh dataset phân loại cảm xúc tiếng Việt.
(Ăn luôn gap "synthetic data generation" của JD.)

Sinh từ template + slot-filling -> đa dạng câu, cân bằng 2 nhãn.
Chạy:  python -m src.gen_data
"""
import json
import random

from src import config

random.seed(42)

POS_ADJ = ["tuyệt vời", "rất tốt", "xuất sắc", "ưng ý", "chất lượng cao",
           "vượt mong đợi", "đáng tiền", "hài lòng", "quá ổn", "đẹp và bền"]
NEG_ADJ = ["tệ", "kém", "thất vọng", "dở", "không đáng tiền", "quá tệ",
           "lừa đảo", "chán", "kém chất lượng", "phí tiền"]

POS_TEMPLATES = [
    "Sản phẩm {a}, tôi rất hài lòng.",
    "Giao hàng nhanh, đóng gói cẩn thận, {a}.",
    "Chất lượng {a}, sẽ ủng hộ shop dài dài.",
    "Mình rất thích, {a}, recommend cho mọi người.",
    "Dùng {a}, đúng như mô tả, 5 sao.",
    "Shop tư vấn nhiệt tình, sản phẩm {a}.",
    "Mua lần 2 rồi, vẫn {a} như lần đầu.",
    "Giá hợp lý mà chất lượng {a}, quá ok.",
]
NEG_TEMPLATES = [
    "Sản phẩm {a}, tôi rất thất vọng.",
    "Giao hàng chậm, hàng lại {a}, không mua nữa.",
    "Chất lượng {a}, khác xa quảng cáo.",
    "Mình không thích, {a}, phí tiền.",
    "Dùng được vài hôm đã hỏng, {a}.",
    "Shop trả lời cộc lốc, sản phẩm {a}.",
    "Đặt một đằng giao một nẻo, {a}.",
    "Tiền mất tật mang, hàng {a}.",
]


def make_rows(n_per_class: int):
    rows = []
    for _ in range(n_per_class):
        rows.append({"text": random.choice(POS_TEMPLATES).format(a=random.choice(POS_ADJ)),
                     "label": "tích cực"})
        rows.append({"text": random.choice(NEG_TEMPLATES).format(a=random.choice(NEG_ADJ)),
                     "label": "tiêu cực"})
    random.shuffle(rows)
    # Khử trùng lặp để dataset "thật" hơn
    seen, uniq = set(), []
    for r in rows:
        key = (r["text"], r["label"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = make_rows(120)                 # ~240 câu (trước khi khử trùng)
    n_val = max(10, len(rows) // 10)
    val, train = rows[:n_val], rows[n_val:]
    write_jsonl(config.DATA_DIR / "train.jsonl", train)
    write_jsonl(config.DATA_DIR / "val.jsonl", val)
    print(f"✅ Đã sinh: train={len(train)} câu, val={len(val)} câu -> {config.DATA_DIR}")
    print("   Ví dụ:", train[0])


if __name__ == "__main__":
    main()
