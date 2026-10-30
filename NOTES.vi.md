# Module 05 — Fine-tune LLM với QLoRA (Phase 2 lộ trình JD)

Fine-tune một LLM nhỏ (Qwen2.5-1.5B) bằng **QLoRA (4-bit)** cho task **phân loại cảm xúc tiếng Việt**.
Cùng cơ chế này → đổi nhãn + dữ liệu là thành **moderation** (độc hại/bình thường).

> Lấp gap JD: **fine-tune LLM** (J1.3, J2.3) · **moderation** (J1.2) · **synthetic data** (J1.5, J2.2).

## Ý tưởng
- **LoRA**: đóng băng model gốc, chỉ "cắm" ma trận nhỏ (low-rank) vào lớp attention → train ~1% tham số, lưu adapter vài MB.
- **QLoRA**: + nén model gốc xuống **4-bit (NF4)** → fine-tune model 1–7B vừa GPU 11GB (RTX 2080 Ti).

## Cấu trúc
```
module-02-finetune-llm/
├── src/
│   ├── config.py      # model, nhãn, tham số LoRA & train
│   ├── gen_data.py    # SINH dataset tổng hợp (synthetic) tiếng Việt
│   ├── train.py       # QLoRA fine-tune -> lưu adapter/
│   └── infer.py       # load model gốc + adapter -> phân loại thử
├── data/              # train.jsonl / val.jsonl (do gen_data sinh)
└── adapter/           # LoRA adapter sau khi train (vài MB)
```

## Chạy (trên remote, venv-finetune, CẦN GPU)
```bash
ssh <user>@<gpu-server>
cd <DATA_DIR>/finetune
source <DATA_DIR>/venv-finetune/bin/activate
python -m src.gen_data   # 1) sinh dataset
python -m src.train      # 2) fine-tune (QLoRA) -> adapter/
python -m src.infer      # 3) test phân loại
```

## Môi trường
- venv riêng: `<DATA_DIR>/venv-finetune` (CUDA torch + peft + trl + bitsandbytes).
- Tách khỏi venv RAG (`venv-chatbot`) để không đụng setup đang chạy.
- Model + cache: ổ DATA (`HF_HOME`).

## Bước tiếp (nâng cấp)
1. Đổi task sang **moderation** (nhãn: `độc hại` / `bình thường`) + dữ liệu thật.
2. Đánh giá **precision/recall** trên tập val (Phase 4 — Evaluation).
3. Merge adapter vào model & serve (hoặc gắn vào pipeline RAG/agent).
