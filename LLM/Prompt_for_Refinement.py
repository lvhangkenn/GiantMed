import json
import pickle
from typing import Any, Dict, List

from tqdm import tqdm

from config import (
    MED_MAX_IDX,
    MODEL_CONFIGS,
    PROB_MAX,
    PROB_MIN,
    VOC_PKL_PATH,
)
from utils import (
    build_patient_info,
    build_similar_visits_section,
    get_med_names_for_indices,
    get_meds_in_prob_range,
    id_to_code,
    ids_to_names_with_probs,
    load_ddi_id_pairs,
    load_entity_mapping,
    load_visit_prob_data,
    load_voc,
)

PROMPT_TEMPLATE = """/no_think
You are now playing the role of a clinical pharmacy expert. Your task is to revise the drug recommendation probabilities between 0.3 and 0.7 predicted by a deep learning model.

Task Rules:
- Drugs with a final probability ≥ 0.5 will be recommended to the patient.
- You must revise the drug recommendation probabilities (0.3–0.7) based on Task Steps below and clinical reasoning.

Task Steps:
1. First, read the Electronic health record of the patient's current condition carefully.
2. Second, examine the model's predicted probabilities for candidate drugs.
3. Third, use the top-3 similar visits for cross-reference, focusing on diagnoses, procedures and drugs.
4. Fourth, check for Drug-Drug Interactions (DDIs) among the candidate drugs.

Your goals:
- Raise the probability of clinically necessary drugs to the [0.70–0.90] range.
- Lower the probability of drugs that lack evidence, have low necessity, or raise safety concerns.
- Do not retain drugs solely due to moderate model score without justification.
- Avoid keeping drugs in the ambiguous [0.3–0.7] range.
- Eliminate drugs with serious drug-drug interactions (DDI) or functional redundancy.
- You need to change every drug probability according to the input knowledge.

Input data

Electronic health record (EHR): The patient's electronic health record contains the patient's diagnosis and procedure information for this visit. You need to recommend appropriate drugs for the patient's visit.
The patient's electronic health record is as follows:
{ehr_text}

Candidate drugs: The following are drugs with an initial recommendation probability between 0.3 and 0.7. Your task is to revise the drug recommendation probabilities below:
{meds_to_correct_str}

Top-3 similar visits: Based on the patient's current condition, the following are the top-3 most similar visits.
- 'Overlapping Diagnoses' and 'Overlapping Procedures' that appear in both the current patient and the similar visit.
- 'Overlapping Drugs' are those both in the candidate drug and in the similar visit.
- If a candidate drug was prescribed in a similar visit with clinical overlap, interpret this as a sign that the drug may be relevant in similar clinical scenarios. Carefully assess whether the same rationale applies to the current patient before making any adjustment.
{similar_visits_text}

Drug-Drug Interactions (DDI): Based on a DDI database, the following potential interactions were found *among the candidate drugs above*. Avoid recommending drugs that have serious interactions with each other.
The format is: ("Drug A", "Drug B"), ("Drug C", "Drug D"), ...
Potential interactions among candidate drugs:
{ddi_str}

Output format:
("drug name 1", "probability 1"), ("drug name 2", "probability 2"),...

Note: Strictly follow the required output format, one by one output each drug in the modified drug list and its corresponding probability, do not output the analysis process. You need to change every drug probability according to the input knowledge.
/no_think"""

def build_prompts(model_name: str) -> List[Dict[str, Any]]:
    cfg = MODEL_CONFIGS[model_name]

    voc_data        = load_voc(VOC_PKL_PATH)
    entity_mapping  = load_entity_mapping(cfg.mapping_json)
    all_medications = get_med_names_for_indices(cfg.mapping_json, voc_data, max_idx=MED_MAX_IDX)
    all_ddi_pairs   = load_ddi_id_pairs(cfg.ddi_txt)
    visit_data_map  = load_visit_prob_data(cfg.predictions_json)
    med_map         = entity_mapping.get("med", {})

    with open(cfg.records_pkl, "rb") as f:
        test_patients = pickle.load(f)

    with open(cfg.similarity_json, "r", encoding="utf-8") as f:
        similarity_data = json.load(f)

    patients_prompts: List[Dict[str, Any]] = []

    for test_idx, patient in tqdm(
        list(enumerate(test_patients)),
        desc=f"[{model_name}] Building prompts",
        total=len(test_patients),
    ):
        for visit_idx in range(len(patient)):
            ehr_text, _ = build_patient_info(
                patient, entity_mapping, all_medications, voc_data,
                predict_visit_idx=visit_idx,
            )

            visit_data_list = visit_data_map.get(test_idx, [])
            predicted = (
                visit_data_list[visit_idx].get("predicted", [])
                if visit_idx < len(visit_data_list) else []
            )

            meds_to_correct = get_meds_in_prob_range(predicted, PROB_MIN, PROB_MAX)
            meds_to_correct_str = (
                ", ".join(ids_to_names_with_probs(meds_to_correct, entity_mapping, voc_data))
                if meds_to_correct else "None"
            )

            # DDI section
            candidate_drug_ids = {med[0] for med in meds_to_correct}
            potential_ddis = all_ddi_pairs.get(test_idx, {}).get(visit_idx, [])
            relevant_ddis = [
                (a, b) for a, b in potential_ddis
                if a in candidate_drug_ids and b in candidate_drug_ids
            ]
            ddi_parts = []
            for d_a, d_b in relevant_ddis:
                try:
                    name_a = med_map.get(id_to_code(voc_data, "med", d_a))
                    name_b = med_map.get(id_to_code(voc_data, "med", d_b))
                    if name_a and name_b:
                        ddi_parts.append(f'("{name_a}", "{name_b}")')
                except KeyError:
                    continue
            ddi_str = ", ".join(ddi_parts) if ddi_parts else "None."

            # Similar visits section
            top_similar = (
                similarity_data
                .get(f"patient_{test_idx}", {})
                .get(f"visit_{visit_idx}", [])[:3]
            )
            candidate_drug_names = []
            for med_id, _ in meds_to_correct:
                try:
                    name = med_map.get(id_to_code(voc_data, "med", med_id))
                    if name:
                        candidate_drug_names.append(name)
                except KeyError:
                    continue
            similar_visits_text = build_similar_visits_section(
                top_similar, entity_mapping, voc_data, candidate_drug_names,
            )

            prompt = PROMPT_TEMPLATE.format(
                ehr_text=ehr_text,
                meds_to_correct_str=meds_to_correct_str,
                similar_visits_text=similar_visits_text,
                ddi_str=ddi_str,
            )

            real_med_ids = patient[visit_idx][2]
            real_med_names = [
                med_map.get(id_to_code(voc_data, "med", m), str(m))
                for m in real_med_ids
            ]

            patients_prompts.append({
                "patient_index":  test_idx,
                "visit_index":    visit_idx,
                "prompt":         prompt,
                "real_med_ids":   real_med_ids,
                "real_med_names": real_med_names,
            })

    return patients_prompts