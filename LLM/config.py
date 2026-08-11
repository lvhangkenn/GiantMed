from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SUPPORTED_MODELS = ["SafeDrug", "MoleRec", "DEPOT", "MedAlign"]

LLM_ROOT = Path(__file__).resolve().parent
REPO_ROOT = LLM_ROOT.parents[1]
LLM_DATA_ROOT = LLM_ROOT / "LLM-data"
DATA_ROOT = REPO_ROOT / "data" / "ready"


@dataclass
class ModelConfig:
    predictions_json: str
    mapping_json: str = str(LLM_DATA_ROOT / "code_mapping.json")
    records_pkl: str = str(LLM_DATA_ROOT / "records_valid.pkl")
    similarity_json: str = str(LLM_DATA_ROOT / "similar_EHRs.json")
    ddi_txt: str = str(LLM_DATA_ROOT / "ddi_report.txt")
    prompts_json: str = ""
    output_json: str = ""


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "SafeDrug": ModelConfig(
        predictions_json=str(LLM_DATA_ROOT / "SafeDrug_predictions.json"),
        prompts_json=str(LLM_ROOT / "SafeDrug_prompts.json"),
        output_json=str(LLM_ROOT / "SafeDrug_output.json"),
    ),
    "MoleRec": ModelConfig(
        predictions_json=str(LLM_DATA_ROOT / "MoleRec_predictions.json"),
        prompts_json=str(LLM_ROOT / "MoleRec_prompts.json"),
        output_json=str(LLM_ROOT / "MoleRec_output.json"),
    ),
    "DEPOT": ModelConfig(
        predictions_json=str(LLM_DATA_ROOT / "DEPOT_predictions.json"),
        prompts_json=str(LLM_ROOT / "DEPOT_prompts.json"),
        output_json=str(LLM_ROOT / "DEPOT_output.json"),
    ),
    "MedAlign": ModelConfig(
        predictions_json=str(LLM_DATA_ROOT / "MedAlign_predictions.json"),
        prompts_json=str(LLM_ROOT / "MedAlign_prompts.json"),
        output_json=str(LLM_ROOT / "MedAlign_output.json"),
    ),
}


# LLM settings
LLM_MODEL_PATH = "/LLM/Qwen3-8B"
LLM_CUDA_DEVICE = "0"
LLM_MAX_NEW_TOKENS = 4096
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.8
LLM_TOP_K = 20
LLM_REPETITION_PENALTY = 1.0

# Probability interval used to select boundary medications.
PROB_MIN = 0.3
PROB_MAX = 0.7

# None means: infer the complete medication vocabulary from voc_final.pkl.
MED_MAX_IDX: Optional[int] = None

VOC_PKL_PATH = str(DATA_ROOT / "voc_final.pkl")
