from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

try:
    import dill
except ImportError:  # Standard pickle is sufficient for plain-list record files.
    import pickle as dill


def load_voc(pkl_path: str) -> Dict[str, Any]:
    try:
        with open(pkl_path, "rb") as f:
            voc_data = dill.load(f)
        print(f"[utils] Loaded vocabulary from '{pkl_path}'.")
        return voc_data
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] Vocabulary file not found: '{pkl_path}'")
    except Exception as exc:
        raise SystemExit(f"[ERROR] Failed to load '{pkl_path}': {exc}")


def id_to_code(voc_data: Dict[str, Any], entity_type: str, idx: int) -> str:
    idx = int(idx)
    voc_key = {"diag": "diag_voc", "proc": "pro_voc", "med": "med_voc"}.get(entity_type)
    if voc_key is None:
        raise ValueError(f"entity_type must be 'diag', 'proc', or 'med'; got '{entity_type}'")

    voc = voc_data.get(voc_key)
    if voc is None:
        raise KeyError(f"Key '{voc_key}' not found in voc_data.")

    code = voc.idx2word.get(idx)
    if code is None:
        raise KeyError(f"No code for idx={idx} in {voc_key}.")

    if entity_type in ("proc", "diag"):
        return str(code).zfill(4) if isinstance(code, int) and code < 100 else str(code)
    return str(code)

def load_entity_mapping(mapping_path: str) -> Dict[str, Dict[str, str]]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_med_names_for_indices(
    mapping_path: str,
    voc_data: Dict[str, Any],
    max_idx: int | None = None,
) -> List[str]:
    full_mapping = load_entity_mapping(mapping_path)
    med_mapping = full_mapping.get("med", {})
    med_names: List[str] = []
    upper_bound = max_idx if max_idx is not None else len(voc_data["med_voc"].idx2word) - 1
    for idx in range(upper_bound + 1):
        try:
            code = id_to_code(voc_data, "med", idx)
        except KeyError:
            continue
        name = med_mapping.get(code)
        if name:
            med_names.append(name)
    return med_names


def translate_codes_to_names(codes: List[str], code_map: Dict[str, str]) -> List[str]:
    return [f'"{code_map[c]}"' for c in codes if code_map.get(c)]


def ids_to_names_with_probs(
    med_probs: List[List[float]],
    entity_mapping: Dict[str, Dict[str, str]],
    voc_data: Dict[str, Any],
) -> List[str]:
    med_map = entity_mapping.get("med", {})
    results = []
    for med_id, prob in med_probs:
        try:
            name = med_map.get(id_to_code(voc_data, "med", med_id), str(med_id))
        except KeyError:
            name = str(med_id)
        results.append(f'("{name}", "{prob:.2f}")')
    return results


def get_meds_in_prob_range(
    predicted: List[List[float]],
    min_prob: float = 0.3,
    max_prob: float = 0.7,
) -> List[List[float]]:
    return [med for med in predicted if min_prob <= med[1] < max_prob]

def build_ehr_section(visit_named: Dict[str, List[str]]) -> str:
    diag_str = ", ".join(visit_named.get("diagnosis", [])) or "none"
    proc_str = ", ".join(visit_named.get("procedure", [])) or "none"
    return (
        "Information for current visit:\n"
        f"  diagnosis: {diag_str}\n"
        f"  procedure: {proc_str}"
    )


def build_drug_list_section(all_medications: List[str]) -> str:
    return " ".join(f'{i}. "{name}"' for i, name in enumerate(all_medications, start=1))


def build_patient_info(
    patient: List[List[int]],
    entity_mapping: Dict[str, Dict[str, str]],
    all_medications: List[str],
    voc_data: Dict[str, Any],
    predict_visit_idx: int,
) -> Tuple[str, str]:
    diag_map = entity_mapping.get("diag", {})
    proc_map = entity_mapping.get("proc", {})

    diag_ids, proc_ids, *_ = patient[predict_visit_idx]
    diag_names = [
        f'"{diag_map.get(id_to_code(voc_data, "diag", d), f"DIAG_ID_{d}")}"'
        for d in diag_ids
    ]
    proc_names = [
        f'"{proc_map.get(id_to_code(voc_data, "proc", p), f"PROC_ID_{p}")}"'
        for p in proc_ids
    ]
    ehr_text = build_ehr_section({"diagnosis": diag_names, "procedure": proc_names})
    drug_list_text = build_drug_list_section(all_medications)
    return ehr_text, drug_list_text


def build_similar_visits_section(
    similar_visits: List[Dict],
    entity_mapping: Dict[str, Dict[str, str]],
    voc_data: Dict[str, Any],
    candidate_drug_names: List[str],
) -> str:
    if not similar_visits:
        return "No similar patient visits found."

    diag_map = entity_mapping.get("diag", {})
    proc_map = entity_mapping.get("proc", {})
    med_map  = entity_mapping.get("med", {})

    lines: List[str] = []
    for i, visit_info in enumerate(similar_visits, 1):
        lines.append(f"Similar Visit {i}:")

        def _resolve_ids(id_list: List[str], etype: str) -> List[str]:
            codes = []
            for sid in id_list:
                try:
                    codes.append(id_to_code(voc_data, etype, int(sid)))
                except KeyError as exc:
                    print(f"  [Warning] Skipping {etype} ID {sid}: {exc}")
            return codes

        # Diagnoses
        diag_codes = _resolve_ids(visit_info.get("Overlapping Diagnoses", []), "diag")
        lines.append(f"  - Overlapping Diagnoses: {', '.join(translate_codes_to_names(diag_codes, diag_map)) or 'none'}")

        # Procedures
        proc_codes = _resolve_ids(visit_info.get("Overlapping Procedures", []), "proc")
        lines.append(f"  - Overlapping Procedures: {', '.join(translate_codes_to_names(proc_codes, proc_map)) or 'none'}")

        # Drugs (filtered to candidate set)
        drug_codes = _resolve_ids(visit_info.get("Medications (from train visit)", []), "med")
        drug_names = sorted(
            med_map[c] for c in drug_codes if c in med_map and med_map[c] in candidate_drug_names
        )
        lines.append(f"  - Overlapping Drugs: {', '.join(f'{chr(34)}{n}{chr(34)}' for n in drug_names) or 'none'}")
        lines.append("")

    return "\n".join(lines).strip()


def load_ddi_id_pairs(ddi_file_path: str) -> Dict[int, Dict[int, List[Tuple[int, int]]]]:
    ddi_data: Dict[int, Dict[int, List[Tuple[int, int]]]] = {}
    current_patient: int | None = None
    current_visit:   int | None = None

    with open(ddi_file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("Patient"):
                current_patient = int(re.search(r"Patient (\d+):", line).group(1))
                ddi_data[current_patient] = {}
            elif line.startswith("Visit") and current_patient is not None:
                current_visit = int(re.search(r"Visit (\d+):", line).group(1))
                ddi_data[current_patient][current_visit] = []
            elif line.startswith("(") and current_patient is not None and current_visit is not None:
                pairs = re.findall(r"\((\d+),\s*(\d+)\)", line)
                ddi_data[current_patient][current_visit].extend(
                    (int(a), int(b)) for a, b in pairs
                )
    return ddi_data

def load_visit_prob_data(json_path: str) -> Dict[int, List[Dict[str, Any]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {patient["patient_id"]: patient["visits"] for patient in data}