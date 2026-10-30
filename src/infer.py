"""
Test model đã fine-tune: load model gốc (4-bit) + LoRA adapter rồi phân loại vài câu.
Chạy:  python -m src.infer
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src import config

TESTS = [
    "Hàng đẹp, giao nhanh, mình rất ưng!",
    "Sản phẩm dở tệ, phí tiền, không mua nữa.",
    "Shop tư vấn nhiệt tình, chất lượng ổn áp.",
    "Đặt một đằng giao một nẻo, quá thất vọng.",
    "Giao đúng hẹn, đóng gói kỹ, sẽ ủng hộ tiếp.",
]


def main():
    tok = AutoTokenizer.from_pretrained(str(config.ADAPTER_DIR))
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    print("Nạp model gốc (4-bit) + adapter đã fine-tune ...")
    base = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, quantization_config=bnb, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(config.ADAPTER_DIR))
    model.eval()

    print("\n=== KẾT QUẢ PHÂN LOẠI (sau fine-tune) ===")
    for text in TESTS:
        messages = [{"role": "user", "content": f"{config.INSTRUCTION}\nCâu: {text}"}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"  '{text}'\n      -> {ans}\n")


if __name__ == "__main__":
    main()
