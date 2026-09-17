#!/usr/bin/env python3
"""Run LLM probability refinement, update predictions, and report metrics."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import dill
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM refinement pipeline")
    parser.add_argument("--model-name", default="Qwen3-8B")
    parser.add_argument("--cuda-visible-devices", default="8")
    parser.add_argument("--mapping-path", default=Path("../data/LLM-data/MIMIC-III/code_mapping.json"))
    parser.add_argument("--test-pkl-path", default=Path("../data/LLM-data/MIMIC-III/records_valid.pkl"))
    parser.add_argument("--similarity-json-path", default=Path("../data/LLM-data/MIMIC-III/similar_EHRs.json"))
    parser.add_argument("--ddi-file-path", default=Path("../data/LLM-data/MIMIC-III/ddi_report.txt"))
    parser.add_argument("--voc-path", default=Path("../data/ready/MIMIC-III/voc_final.pkl"))
    parser.add_argument("--ddi-matrix-path", default=Path("../data/ready/MIMIC-III/ddi_A_final.pkl"))
    parser.add_argument("--prob-json-path", default=Path("../data/LLM-data/MIMIC-III/MedAlign_predictions.json"))
    parser.add_argument("--boundary-low", type=float, default=0.3)
    parser.add_argument("--boundary-high", type=float, default=0.7)
    parser.add_argument("--refinement-output", default=Path("../data/LLM-data/MIMIC-III/MedAlign_refine.json"))
    parser.add_argument("--updated-output", default=Path("../data/LLM-data/MIMIC-III/MedAlign_refine-update.json"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k-sampling", type=int, default=20)
    parser.add_argument("--print-prompts", action="store_true")
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_vocabulary(path: str) -> Any:
    with open(path, "rb") as file:
        return dill.load(file)


def code_from_id(vocabulary: Any, entity_type: str, index: int) -> str:
    key = {"diag": "diag_voc", "proc": "pro_voc", "med": "med_voc"}[entity_type]
    code = vocabulary[key].idx2word.get(int(index))
    if code is None:
        raise KeyError(f"Missing {entity_type} index: {index}")
    if entity_type in {"diag", "proc"} and isinstance(code, int) and code < 100:
        return str(code).zfill(4)
    return str(code)


def name_from_id(vocabulary: Any, mapping: Dict[str, Dict[str, str]], entity_type: str, index: int) -> str:
    fallback = f"{entity_type.upper()}_ID_{index}"
    try:
        return mapping.get(entity_type if entity_type != "proc" else "proc", {}).get(
            code_from_id(vocabulary, entity_type, index), fallback
        )
    except KeyError:
        return fallback


def build_history(patient: Sequence[Sequence[Any]], visit_index: int, vocabulary: Any, mapping: Dict[str, Dict[str, str]]) -> str:
    if visit_index == 0:
        return "none (no historical medications before this visit)"
    lines = []
    for prior_index in range(visit_index):
        names = [f'"{name_from_id(vocabulary, mapping, "med", med_id)}"' for med_id in patient[prior_index][2]]
        lines.append(f"  visit_{prior_index}: {', '.join(names) or 'none'}")
    return "\n".join(lines)


def build_ehr(patient: Sequence[Sequence[Any]], visit_index: int, vocabulary: Any, mapping: Dict[str, Dict[str, str]]) -> str:
    diagnoses, procedures = patient[visit_index][0], patient[visit_index][1]
    diagnosis_text = ", ".join(f'"{name_from_id(vocabulary, mapping, "diag", item)}"' for item in diagnoses) or "none"
    procedure_text = ", ".join(f'"{name_from_id(vocabulary, mapping, "proc", item)}"' for item in procedures) or "none"
    return f"Information for current visit:\n  diagnosis: {diagnosis_text}\n  procedure: {procedure_text}"


def load_ddi_pairs(path: str) -> Dict[int, Dict[int, List[Tuple[int, int]]]]:
    result: Dict[int, Dict[int, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    patient_id = visit_id = None
    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            patient_match = re.match(r"Patient\s+(\d+):", line)
            visit_match = re.match(r"Visit\s+(\d+):", line)
            if patient_match:
                patient_id = int(patient_match.group(1))
            elif visit_match:
                visit_id = int(visit_match.group(1))
            elif line.startswith("(") and patient_id is not None and visit_id is not None:
                result[patient_id][visit_id].extend((int(a), int(b)) for a, b in re.findall(r"\((\d+),\s*(\d+)\)", line))
    return result


def build_similar_visits(similar_visits: List[Dict[str, Any]], candidate_ids: set[int], vocabulary: Any, mapping: Dict[str, Dict[str, str]]) -> str:
    if not similar_visits:
        return "No similar patient visits found."
    lines: List[str] = []
    for position, item in enumerate(similar_visits, 1):
        diagnosis_names = [f'"{name_from_id(vocabulary, mapping, "diag", int(x))}"' for x in item.get("Overlapping Diagnoses", [])]
        procedure_names = [f'"{name_from_id(vocabulary, mapping, "proc", int(x))}"' for x in item.get("Overlapping Procedures", [])]
        medication_names = [f'"{name_from_id(vocabulary, mapping, "med", int(x))}"' for x in item.get("Medications (from train visit)", []) if int(x) in candidate_ids]
        lines.extend([f"Similar Visit {position}:", f"  - Overlapping Diagnoses: {', '.join(diagnosis_names) or 'none'}", f"  - Overlapping Procedures: {', '.join(procedure_names) or 'none'}", f"  - Overlapping Drugs: {', '.join(sorted(medication_names)) or 'none'}", ""])
    return "\n".join(lines).strip()


def build_prompt(ehr: str, history: str, candidates: str, similar_visits: str, ddi: str) -> str:
    return f'''
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

Consider the following evidence in order:
1. Current EHR: determine whether the diagnoses and procedures provide a clinical indication for the drug.
2. Deep-model probability: treat the original probability as a prior signal rather than the final decision.
3. Similar visits: use overlapping diagnoses, procedures, and medications only as supporting evidence.
4. DDI evidence: check whether recommending the drug together with other candidate drugs introduces safety concerns.
5. Necessity and redundancy: determine whether the drug is clinically necessary or functionally redundant.

Input data

Electronic health record (EHR): The patient's electronic health record contains the patient's diagnosis and procedure information for this visit. You need to recommend appropriate drugs for the patient's visit.
The patient's electronic health record is as follows:
{ehr}

Historical medications: The following are medications from the patient's previous visits. Only visits before the current visit are included:
{history}

Candidate drugs: The following are drugs with an initial recommendation probability between 0.3 and 0.7. Your task is to revise the drug recommendation probabilities below:
{candidates}

Top-3 similar visits: Based on the patient's current condition, the following are the top-3 most similar visits.
- 'Overlapping Diagnoses' and 'Overlapping Procedures' that appear in both the current patient and the similar visit.
- 'Overlapping Drugs' are those both in the candidate drug and in the similar visit.
- If a candidate drug was prescribed in a similar visit with clinical overlap, interpret this as a sign that the drug may be relevant in similar clinical scenarios. Carefully assess whether the same rationale applies to the current patient before making any adjustment.
{similar_visits}

Drug-Drug Interactions (DDI): Based on a DDI database, the following potential interactions were found *among the candidate drugs above*. Avoid recommending drugs that have serious interactions with each other.
The format is: ("Drug A", "Drug B"), ("Drug C", "Drug D"), ...
Potential interactions among candidate drugs:
{ddi}

Before producing the final answer, reason carefully about each candidate drug using the patient's EHR, the model probability, similar visits, and DDI evidence.
For each candidate drug, internally determine whether the evidence supports increasing or decreasing its probability.
Output only the final revised drug probabilities in the required format. Do not include your reasoning or analysis in the final answer.

Output format:
("drug name 1", "probability 1"), ("drug name 2", "probability 2"),...

Note: Strictly follow the required output format, one by one output each drug in the modified drug list and its corresponding probability, do not output the analysis process.
'''


def parse_recommendation(text: str) -> List[Tuple[str, float]]:
    results = []
    for name, probability in re.findall(r'\(\s*"([^"]+)"\s*,\s*"?([0-9]*\.?[0-9]+)"?\s*\)', text):
        value = float(probability)
        if 0.0 <= value <= 1.0:
            results.append((name, value))
    return results


def run_refinement(args: argparse.Namespace, vocabulary: Any, mapping: Dict[str, Dict[str, str]]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto", device_map="auto")
    patients = pickle.load(open(args.test_pkl_path, "rb"))
    probabilities = {item["patient_id"]: item["visits"] for item in load_json(args.prob_json_path)}
    similar_data = load_json(args.similarity_json_path)
    ddi_pairs = load_ddi_pairs(args.ddi_file_path)
    outputs = []
    for patient_index, patient in tqdm(list(enumerate(patients)), desc="Refining visits"):
        visits = probabilities.get(patient_index, [])
        for visit_index in range(len(patient)):
            predicted = visits[visit_index].get("predicted", []) if visit_index < len(visits) else []
            candidates = [(int(med_id), float(probability)) for med_id, probability in predicted if args.boundary_low <= float(probability) < args.boundary_high]
            candidate_ids = {med_id for med_id, _ in candidates}
            candidate_text = ", ".join(f'("{name_from_id(vocabulary, mapping, "med", med_id)}", "{probability:.2f}")' for med_id, probability in candidates) or "None"
            ddi_text = ", ".join(f'("{name_from_id(vocabulary, mapping, "med", left)}", "{name_from_id(vocabulary, mapping, "med", right)}")' for left, right in ddi_pairs.get(patient_index, {}).get(visit_index, []) if left in candidate_ids and right in candidate_ids) or "None."
            similar_visits = similar_data.get(f"patient_{patient_index}", {}).get(f"visit_{visit_index}", [])[:args.top_k]
            prompt = build_prompt(build_ehr(patient, visit_index, vocabulary, mapping), build_history(patient, visit_index, vocabulary, mapping), candidate_text, build_similar_visits(similar_visits, candidate_ids, vocabulary, mapping), ddi_text)
            if args.print_prompts:
                print(prompt)
            messages = [{"role": "user", "content": prompt}]
            chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = tokenizer(chat, return_tensors="pt").to(model.device)
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k_sampling, repetition_penalty=1.0)
            response = tokenizer.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            actual = patient[visit_index][2]
            outputs.append({"patient_index": patient_index, "visit_index": visit_index, "prompt": prompt, "recommendation": response, "real_med_ids": actual, "real_med_names": [name_from_id(vocabulary, mapping, "med", med_id) for med_id in actual]})
    save_json(outputs, args.refinement_output)
    print(f"Saved LLM refinements to {args.refinement_output}")


def update_predictions(args: argparse.Namespace, vocabulary: Any, mapping: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    name_to_id: Dict[str, int] = {}
    for med_id, code in vocabulary["med_voc"].idx2word.items():
        name = mapping.get("med", {}).get(str(code))
        if name is not None:
            name_to_id.setdefault(name, int(med_id))
    updates: Dict[Tuple[int, int], Dict[int, float]] = {}
    for item in load_json(args.refinement_output):
        values = {name_to_id[name]: probability for name, probability in parse_recommendation(item["recommendation"]) if name in name_to_id}
        if values:
            updates[(int(item["patient_index"]), int(item["visit_index"]))] = values
    data = load_json(args.prob_json_path)
    for patient in data:
        patient_id = int(patient["patient_id"])
        for visit in patient.get("visits", []):
            values = updates.get((patient_id, int(visit["visit_id"])), {})
            visit["predicted"] = [[med_id, values.get(int(med_id), probability)] for med_id, probability in visit.get("predicted", [])]
    save_json(data, args.updated_output)
    print(f"Saved updated probabilities to {args.updated_output}")
    return data


def calculate_metrics(data: List[Dict[str, Any]], ddi_matrix_path: str) -> None:
    all_ids = {int(med_id) for patient in data for visit in patient.get("visits", []) for med_id in visit.get("actual", [])}
    all_ids.update(int(med_id) for patient in data for visit in patient.get("visits", []) for med_id, _ in visit.get("predicted", []))
    size = max(all_ids) + 1 if all_ids else 0
    patient_scores = defaultdict(list)
    medication_counts = []
    all_combinations = ddi_combinations = 0
    ddi_matrix = dill.load(open(ddi_matrix_path, "rb"))
    for patient in data:
        jac, ap, precision, recall, f1, p1, p3, p5, ece, counts = [], [], [], [], [], [], [], [], [], []
        for visit in patient.get("visits", []):
            target = set(map(int, visit.get("actual", [])))
            ranked = [(int(med_id), float(probability)) for med_id, probability in visit.get("predicted", [])]
            positives = {med_id for med_id, probability in ranked if probability >= 0.5}
            probabilities = np.zeros(size)
            for med_id, probability in ranked:
                probabilities[med_id] = probability
            intersection, union = target & positives, target | positives
            visit_precision = len(intersection) / len(positives) if positives else 0.0
            visit_recall = len(intersection) / len(target) if target else 0.0
            jac.append(len(intersection) / len(union) if union else 0.0)
            precision.append(visit_precision); recall.append(visit_recall)
            f1.append(2 * visit_precision * visit_recall / (visit_precision + visit_recall) if visit_precision + visit_recall else 0.0)
            labels = np.zeros(size, dtype=int); labels[list(target)] = 1
            try: ap.append(average_precision_score(labels, probabilities, average="macro"))
            except ValueError: ap.append(0.0)
            ranked_ids = [med_id for med_id, _ in ranked]
            p1.append(sum(item in target for item in ranked_ids[:1]) / 1)
            p3.append(sum(item in target for item in ranked_ids[:3]) / 3)
            p5.append(sum(item in target for item in ranked_ids[:5]) / 5)
            mask = (probabilities >= 0.3) & (probabilities <= 0.7)
            if np.any(mask):
                bins = np.linspace(0.3, 0.7, 11); local_ece = 0.0
                for low, high in zip(bins[:-1], bins[1:]):
                    in_bin = mask & (probabilities > low) & (probabilities <= high)
                    if np.any(in_bin): local_ece += np.mean(in_bin[mask]) * abs(np.mean(labels[in_bin]) - np.mean(probabilities[in_bin]))
                ece.append(local_ece)
            else: ece.append(0.0)
            counts.append(len(positives))
            positive_list = sorted(positives)
            for offset, left in enumerate(positive_list):
                for right in positive_list[offset + 1:]:
                    all_combinations += 1
                    ddi_combinations += int(ddi_matrix[left, right] == 1 or ddi_matrix[right, left] == 1)
        for name, values in {"jaccard": jac, "prauc": ap, "avg_prc": precision, "avg_recall": recall, "avg_f1": f1, "p@1": p1, "p@3": p3, "p@5": p5, "ECE": ece}.items():
            patient_scores[name].append(float(np.mean(values)))
        medication_counts.append(float(np.mean(counts)))
    print("\nAveraged Metrics (Patient-Batch):")
    for name in ("jaccard", "prauc", "avg_prc", "avg_recall", "avg_f1", "p@1", "p@3", "p@5", "ECE"):
        print(f"{name}: {np.mean(patient_scores[name]):.4f}")
    print(f"Average number: {np.mean(medication_counts):.4f}")
    print(f"Average DDI Rate: {ddi_combinations / all_combinations if all_combinations else 0.0:.4f}")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.boundary_low < args.boundary_high <= 1.0:
        raise ValueError("Boundary thresholds must satisfy 0 <= low < high <= 1.")
    vocabulary, mapping = load_vocabulary(args.voc_path), load_json(args.mapping_path)
    run_refinement(args, vocabulary, mapping)
    updated_data = update_predictions(args, vocabulary, mapping)
    calculate_metrics(updated_data, args.ddi_matrix_path)


if __name__ == "__main__":
    main()
