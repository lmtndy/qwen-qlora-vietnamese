"""
Cấu hình cho Module 05 — Fine-tune LLM (QLoRA).
Task demo: phân loại cảm xúc tiếng Việt (tích cực / tiêu cực).
Cùng cơ chế này đổi nhãn + dữ liệu là thành MODERATION (độc hại / bình thường).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ADAPTER_DIR = ROOT / "adapter"      # nơi lưu LoRA adapter sau khi train (vài MB)

# --- Model gốc (nhỏ, tiếng Việt tốt, vừa GPU 11GB khi 4-bit) ---
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# --- Task ---
LABELS = ["tích cực", "tiêu cực"]
INSTRUCTION = (
    "Bạn là bộ phân loại cảm xúc. Phân loại câu đánh giá sau, "
    "CHỈ trả lời đúng một trong hai nhãn: 'tích cực' hoặc 'tiêu cực'."
)

# --- QLoRA / LoRA ---
LORA_R = 16            # rank của ma trận low-rank (càng cao càng nhiều tham số train)
LORA_ALPHA = 32        # hệ số scale
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]  # cắm adapter vào các lớp attention

# --- Training ---
EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-4
MAX_SEQ_LEN = 256
