"""
Fine-tune LLM bằng QLoRA (4-bit) cho task phân loại cảm xúc tiếng Việt.
Chạy:  python -m src.train   (CẦN GPU)

Luồng: tải model 4-bit -> cắm LoRA adapter -> train trên dataset -> lưu adapter (vài MB).
"""
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from src import config


def build_text(tokenizer, example):
    """Định dạng mỗi mẫu thành hội thoại instruction -> label (dùng chat template của model)."""
    messages = [
        {"role": "user", "content": f"{config.INSTRUCTION}\nCâu: {example['text']}"},
        {"role": "assistant", "content": example["label"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def main():
    print(f"[1/5] Tải tokenizer + model 4-bit (NF4): {config.MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(config.MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,                      # nén model gốc xuống 4-bit -> tiết kiệm VRAM
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,   # Turing (2080 Ti) không có bf16 -> dùng fp16
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, quantization_config=bnb, device_map="auto",
        dtype=torch.float16,        # ép fp16 toàn bộ (tránh bfloat16 làm lỗi GradScaler trên Turing)
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)   # chuẩn bị model 4-bit để train (cast norm, bật grad...)

    print("[2/5] Nạp dataset + định dạng theo chat template")
    ds = load_dataset("json", data_files={
        "train": str(config.DATA_DIR / "train.jsonl"),
        "val": str(config.DATA_DIR / "val.jsonl"),
    })
    ds = ds.map(lambda e: build_text(tok, e))

    print("[3/5] Cấu hình LoRA (chỉ train ~1% tham số)")
    lora = LoraConfig(
        r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGETS, task_type="CAUSAL_LM", bias="none",
    )

    print("[4/5] Train (QLoRA)")
    args = SFTConfig(
        output_dir=str(config.ADAPTER_DIR),
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        learning_rate=config.LR,
        logging_steps=5,
        save_strategy="no",
        report_to="none",                        # không bật fp16/bf16 -> tránh GradScaler lỗi với bfloat16
        dataset_text_field="text",
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds["train"],
        peft_config=lora, processing_class=tok,
    )
    trainer.train()

    print("[5/5] Lưu LoRA adapter")
    trainer.save_model(str(config.ADAPTER_DIR))
    tok.save_pretrained(str(config.ADAPTER_DIR))
    print(f"✅ Xong! Adapter lưu tại {config.ADAPTER_DIR}  (test: python -m src.infer)")


if __name__ == "__main__":
    main()
