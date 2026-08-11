import argparse
import json
import os
import sys
import time

from config import LLM_CUDA_DEVICE, LLM_MODEL_PATH, MODEL_CONFIGS, SUPPORTED_MODELS
from LLM_Recommend import load_model, run_inference
from Prompt_for_Refinement import build_prompts

def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def _sep(label: str = "") -> None:
    width = 60
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * pad}\n")
    else:
        print(f"\n{'─' * width}\n")

def _stage1_build(model_name: str, cfg, debug: bool) -> list:
    _sep("Stage 1 · Build Prompts")
    t0 = time.time()

    prompts = build_prompts(model_name)

    with open(cfg.prompts_json, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {len(prompts)} prompts saved → {cfg.prompts_json}  ({_fmt_time(time.time() - t0)})")

    if debug and prompts:
        _sep("First Prompt (debug)")
        print(prompts[0]["prompt"])

    return prompts


def _stage1_load(prompts_json_path: str) -> list:
    _sep("Stage 1 · Load Existing Prompts")

    if not os.path.exists(prompts_json_path):
        print(f"[ERROR] Prompts file not found: '{prompts_json_path}'")
        print("        Remove --skip-prompts to generate it first.")
        sys.exit(1)

    with open(prompts_json_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    print(f"✓ Loaded {len(prompts)} prompts from {prompts_json_path}")
    return prompts


def _stage2_infer(cfg, prompts: list, llm_path: str, cuda_device: str, debug: bool) -> None:
    _sep("Stage 2 · LLM Inference")
    t0 = time.time()

    if debug and prompts:
        _sep("First Prompt (debug)")
        print(prompts[0]["prompt"])

    tokenizer, model = load_model(llm_path, cuda_device)
    results = run_inference(prompts, tokenizer, model)

    with open(cfg.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {len(results)} recommendations saved → {cfg.output_json}  ({_fmt_time(time.time() - t0)})")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end drug-recommendation refinement pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=SUPPORTED_MODELS,
        help="Base drug-recommendation model to process.",
    )
    parser.add_argument(
        "--llm-path",
        default=LLM_MODEL_PATH,
        metavar="PATH",
        help=f"Path to the LLM checkpoint (default: {LLM_MODEL_PATH}).",
    )
    parser.add_argument(
        "--cuda-device",
        default=LLM_CUDA_DEVICE,
        metavar="N",
        help=f"CUDA_VISIBLE_DEVICES value (default: {LLM_CUDA_DEVICE}).",
    )
    parser.add_argument(
        "--skip-prompts",
        action="store_true",
        help="Skip Stage 1; load an existing prompts JSON instead.",
    )
    parser.add_argument(
        "--prompts-only",
        action="store_true",
        help="Run Stage 1 only; skip LLM inference.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the first prompt before each stage.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg  = MODEL_CONFIGS[args.model]

    _sep("Pipeline Start")
    print(f"  model        : {args.model}")
    print(f"  prompts file : {cfg.prompts_json}")
    if not args.prompts_only:
        print(f"  output file  : {cfg.output_json}")
        print(f"  LLM path     : {args.llm_path}")
        print(f"  CUDA device  : {args.cuda_device}")

    total_start = time.time()

    # Stage 1
    if args.skip_prompts:
        prompts = _stage1_load(cfg.prompts_json)
    else:
        prompts = _stage1_build(args.model, cfg, debug=args.debug)

    # Stage 2
    if not args.prompts_only:
        _stage2_infer(cfg, prompts, args.llm_path, args.cuda_device, debug=args.debug)

    _sep("Done")
    print(f"  Total time : {_fmt_time(time.time() - total_start)}")
    if not args.prompts_only:
        print(f"  Results    : {cfg.output_json}")


if __name__ == "__main__":
    main()