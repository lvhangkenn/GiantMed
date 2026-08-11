import os
import json
from typing import Any, Dict, List

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "../Qwen3-8B"
CUDA_DEVICE = "0"
PROMPTS_JSON_PATH = "../LLM-data/MedAlign_prompts.json"
OUTPUT_JSON_PATH = "MedAlign_output.json"
MAX_SAMPLES = None  # Set to None for the full dataset, or use a list such as [100, 200, 300...] for small-sample testing.

LLM_MAX_NEW_TOKENS = 4096
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.8
LLM_TOP_K = 20
LLM_REPETITION_PENALTY = 1.0

def load_model(model_path: str, cuda_device: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    print(f"[LLM] Loading tokenizer from '{model_path}' ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"[LLM] Loading model from '{model_path}' ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    print("[LLM] Model loaded successfully.")

    return tokenizer, model

def run_inference(
    prompts_data: List[Dict[str, Any]],
    tokenizer,
    model,
) -> List[Dict[str, Any]]:

    outputs: List[Dict[str, Any]] = []

    for entry in tqdm(
        prompts_data,
        total=len(prompts_data),
        desc="[LLM] Inference",
    ):
        prompt = entry["prompt"]

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = tokenizer(
            chat_text,
            return_tensors="pt",
        ).to(model.device)

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

        input_length = inputs.input_ids.shape[1]

        new_tokens = generated_ids[:, input_length:]

        completion = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )[0].strip()

        outputs.append(
            {
                "patient_index": entry["patient_index"],
                "visit_index": entry["visit_index"],
                "recommendation": completion,
                "real_med_ids": entry["real_med_ids"],
                "real_med_names": entry["real_med_names"],
            }
        )

    return outputs

def main():

    print(f"[LLM] Loading prompts from '{PROMPTS_JSON_PATH}' ...")

    with open(
        PROMPTS_JSON_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        prompts_data: List[Dict[str, Any]] = json.load(f)

    print(
        f"[LLM] Loaded {len(prompts_data)} prompts."
    )

    if MAX_SAMPLES is not None:

        prompts_data = prompts_data[:MAX_SAMPLES]

        print(
            f"[LLM] Using first {len(prompts_data)} prompts "
            f"for inference."
        )

    tokenizer, model = load_model(
        model_path=MODEL_PATH,
        cuda_device=CUDA_DEVICE,
    )

    llm_outputs = run_inference(
        prompts_data=prompts_data,
        tokenizer=tokenizer,
        model=model,
    )

    print(
        f"[LLM] Saving recommendations to "
        f"'{OUTPUT_JSON_PATH}' ..."
    )

    with open(
        OUTPUT_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as fout:

        json.dump(
            llm_outputs,
            fout,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"[LLM] Saved {len(llm_outputs)} "
        f"patient visit recommendations to "
        f"'{OUTPUT_JSON_PATH}'."
    )

if __name__ == "__main__":
    main()