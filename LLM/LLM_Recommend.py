import os
from typing import Any, Dict, List

from tqdm import tqdm

from config import (
    LLM_MAX_NEW_TOKENS,
    LLM_REPETITION_PENALTY,
    LLM_TEMPERATURE,
    LLM_TOP_K,
    LLM_TOP_P,
)

def load_model(model_path: str, cuda_device: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    print(f"[LLM] Loading tokenizer from '{model_path}' …")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"[LLM] Loading model from '{model_path}' …")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    return tokenizer, model


def run_inference(
    prompts_data: List[Dict[str, Any]],
    tokenizer,
    model,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []

    for entry in tqdm(prompts_data, desc="[LLM] Inference"):
        messages = [{"role": "user", "content": entry["prompt"]}]
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            top_k=LLM_TOP_K,
            min_length=0,
            repetition_penalty=LLM_REPETITION_PENALTY,
        )

        new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
        completion = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        outputs.append({
            "patient_index":  entry["patient_index"],
            "visit_index":    entry["visit_index"],
            "recommendation": completion,
            "real_med_ids":   entry["real_med_ids"],
            "real_med_names": entry["real_med_names"],
        })

    return outputs