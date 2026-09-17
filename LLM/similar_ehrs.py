from __future__ import annotations

import argparse
import ast
import csv
import heapq
import json
import pickle
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import dill
except ImportError:
    dill = None
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: Any):
        return iterable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrieve similar training visits."
    )
    p.add_argument("--predictions", type=Path,
                   default=Path("../data/LLM-data/MIMIC-III/MedAlign_predictions.json"))
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--boundary-low", type=float, default=0.3)
    p.add_argument("--boundary-high", type=float, default=0.7)
    p.add_argument("--train-records", type=Path,
                   default=Path("../data/LLM-data/MIMIC-III/records_train.pkl"))
    p.add_argument("--valid-records", type=Path,
                   default=Path("../data/LLM-data/MIMIC-III/records_valid.pkl"))
    p.add_argument("--medi-c", type=Path, default=Path("../data/LLM-data/MIMIC-III/MEDI.csv"))
    p.add_argument("--ndc-atc", type=Path,
                   default=Path("../data/raw/MIMIC-III/ndc2atc_level4.csv"))
    p.add_argument("--ndc-rxcui", type=Path,
                   default=Path("../data/raw/MIMIC-III/ndc2rxnorm_mapping.txt"))
    p.add_argument("--prescriptions", type=Path,
                   default=Path("../data/raw/MIMIC-III/PRESCRIPTIONS.csv"))
    p.add_argument("--diagnoses-icd", type=Path,
                   default=Path("../data/raw/MIMIC-III/DIAGNOSES_ICD.csv"))
    p.add_argument("--code-mapping", type=Path,
                   default=Path("../data/LLM-data/MIMIC-III/code_mapping.json"))
    p.add_argument("--clinical-ingredient-cache", type=Path,
                   default=Path("../data/raw/MIMIC-III/clinical_rxcui_to_atc3.json"))
    p.add_argument("--output", type=Path,
                   default=Path("../data/LLM-data/MIMIC-III/similar_EHRs.json"))
    p.add_argument("--medi-profile", choices=("all_medi", "medi1", "hps"),
                   default="all_medi")
    p.add_argument("--diagnosis-match", choices=("exact", "medi_ancestor"),
                   default="exact")
    p.add_argument("--no-history-rescue", action="store_true",
                   help="Disable retention based on medications used in the previous visit.")
    args = p.parse_args()
    if args.top_k <= 0:
        p.error("--top-k must be greater than zero")
    if not 0 <= args.boundary_low <= args.boundary_high <= 1:
        p.error("Boundary values must satisfy 0 <= low <= high <= 1")
    return args


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_records(path: Path) -> Any:
    with path.open("rb") as f:
        if dill is not None:
            try:
                return dill.load(f)
            except Exception:
                f.seek(0)
        return pickle.load(f)


def normal_records(records: Any) -> list:
    out = []
    for patient in records:
        visits = []
        for visit in patient:
            if len(visit) < 3:
                raise ValueError(f"Visit is missing one of the first three fields: {visit}")
            visits.append([[str(x) for x in visit[0]], [str(x) for x in visit[1]],
                           [str(x) for x in visit[2]], *visit[3:]])
        out.append(visits)
    return out


def sort_codes(codes: set[str]) -> list[str]:
    def key(x: str) -> tuple[int, Any]:
        try:
            return 0, int(x)
        except ValueError:
            return 1, x
    return sorted(codes, key=key)


def build_train_db(records: list) -> list[dict[str, Any]]:
    db = []
    for pid, patient in enumerate(records):
        for vid, visit in enumerate(patient):
            db.append({"pid": pid, "vid": vid, "diagnoses": set(visit[0]),
                       "procedures": set(visit[1]), "medications": visit[2]})
    return db


def retrieve(valid_visit: list, db: list[dict[str, Any]], top_k: int) -> list[dict]:
    diagnoses, procedures = set(valid_visit[0]), set(valid_visit[1])
    heap: list[tuple] = []
    for row in db:
        union = diagnoses | row["diagnoses"]
        sim = len(diagnoses & row["diagnoses"]) / len(union) if union else 0.0
        result = {
            "train_patient_id": f"patient_{row['pid']}",
            "train_visit_index": row["vid"],
            "total_similarity": float(sim),
            "Overlapping Diagnoses": sort_codes(diagnoses & row["diagnoses"]),
            "Overlapping Procedures": sort_codes(procedures & row["procedures"]),
            "Medications (from train visit)": list(row["medications"]),
        }
        item = (sim, -row["pid"], -row["vid"], result)
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        elif item[:3] > heap[0][:3]:
            heapq.heapreplace(heap, item)
    return [x[3] for x in sorted(heap, key=lambda x: (-x[0], -x[1], -x[2]))]


def retrieve_all(valid: list, db: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for pid, patient in enumerate(tqdm(valid, desc="Retrieving similar EHRs", unit="patient")):
        output[f"patient_{pid}"] = {
            f"visit_{vid}": retrieve(visit, db, top_k)
            for vid, visit in enumerate(patient)
        }
    return output


def norm_icd(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def record_visit(records: list, pid: int, vid: int, split: str) -> list:
    if pid < 0 or pid >= len(records) or vid < 0 or vid >= len(records[pid]):
        raise IndexError(f"{split} index out of range: patient={pid}, visit={vid}")
    return records[pid][vid]


def hadm_id(records: list, pid: int, vid: int, split: str) -> int:
    visit = record_visit(records, pid, vid, split)
    if len(visit) <= 4 or not visit[4]:
        raise ValueError(f"{split} record has no HADM_ID: patient={pid}, visit={vid}")
    return int(visit[4][0])


def parse_predictions(payload: list[dict], valid: list) -> dict[tuple[int, int], dict[int, float]]:
    result: dict[tuple[int, int], dict[int, float]] = {}
    for patient in payload:
        pid = int(str(patient["patient_id"]).replace("patient_", ""))
        if not 0 <= pid < len(valid):
            continue
        for visit in patient["visits"]:
            vid = int(str(visit["visit_id"]).replace("visit_", ""))
            record_visit(valid, pid, vid, "VALID")
            key = (pid, vid)
            if key in result:
                raise ValueError(f"Duplicate prediction: {key}")
            result[key] = {int(m): float(p) for m, p in visit["predicted"]}
    expected = {(p, v) for p, x in enumerate(valid) for v in range(len(x))}
    missing = expected - set(result)
    if missing:
        raise KeyError(f"Prediction file is missing {len(missing)} visits; examples: {sorted(missing)[:10]}")
    return result


def load_medi(path: Path) -> dict[str, dict[str, dict[str, bool]]]:
    index: dict[str, dict[str, dict[str, bool]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("SAB", "")).upper().strip() != "ICD9CM":
                continue
            rxcui, code = str(row.get("RXCUI", "")).strip(), norm_icd(row.get("CODE", ""))
            if not rxcui or len(code) < 3:
                continue
            flags = index[rxcui].setdefault(code, {"medi1": False, "medi1_hps": False,
                                                     "medi2": False, "medi2_hps": False})
            for name, column in (("medi1", "MEDI1"), ("medi1_hps", "MEDI1_HPS"),
                                 ("medi2", "MEDI2"), ("medi2_hps", "MEDI2_HPS")):
                flags[name] |= str(row.get(column, "")).strip().upper() == "TRUE"
    return index


def load_identity(ndc_rxcui: Path, ndc_atc: Path) -> tuple[dict[str, str], dict[str, str]]:
    ndc_to_rxcui = {str(k).strip(): str(v).strip() for k, v in ast.literal_eval(
        ndc_rxcui.read_text(encoding="utf-8-sig")).items() if str(k).strip() and str(v).strip()}
    clinical_to_atc = {}
    with ndc_atc.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rxcui, atc4 = str(row.get("RXCUI", "")).strip(), str(row.get("ATC4", "")).strip()
            if rxcui and atc4 and rxcui not in clinical_to_atc:
                clinical_to_atc[rxcui] = atc4[:4]
    return ndc_to_rxcui, clinical_to_atc


def collect_diagnoses(path: Path, needed: set[int]) -> dict[int, list[str]]:
    result: defaultdict[int, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            try: h = int(float(row.get("HADM_ID", "")))
            except (TypeError, ValueError): continue
            if h in needed:
                code = norm_icd(row.get("ICD9_CODE", ""))
                if len(code) >= 3: result[h].add(code)
    return {h: sorted(codes) for h, codes in result.items()}


def collect_drugs(path: Path, needed: set[tuple[int, str]], ndc_to_rxcui: dict[str, str],
                  clinical_to_atc: dict[str, str]) -> dict[tuple[int, str], set[str]]:
    result: defaultdict[tuple[int, str], set[str]] = defaultdict(set)
    needed_hadm = {h for h, _ in needed}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            try: h = int(float(row.get("HADM_ID", "")))
            except (TypeError, ValueError): continue
            if h not in needed_hadm: continue
            clinical = ndc_to_rxcui.get(str(row.get("NDC", "")).strip())
            atc = clinical_to_atc.get(clinical) if clinical else None
            if clinical and atc and (h, atc) in needed: result[(h, atc)].add(clinical)
    return result


def supported(clinicals: set[str], cache: dict, current_codes: list[str], medi: dict,
              profile: str, diagnosis_match: str) -> bool:
    for clinical in clinicals:
        for ingredient in cache.get(clinical, {}).get("ingredients", []):
            for medi_code, flags in medi.get(str(ingredient.get("rxcui", "")), {}).items():
                profile_ok = profile == "all_medi" or (profile == "medi1" and flags["medi1"]) or (
                    profile == "hps" and (flags["medi1_hps"] or flags["medi2_hps"]))
                diagnosis_ok = any(code == medi_code or (diagnosis_match == "medi_ancestor" and
                                    len(medi_code) >= 3 and code.startswith(medi_code)) for code in current_codes)
                if profile_ok and diagnosis_ok: return True
    return False


def refine(similar: dict, train: list, valid: list, probabilities: dict, medication_codes: list[str],
           diagnoses: dict[int, list[str]], visit_hadm: dict[tuple[int, int], int],
           drugs: dict[tuple[int, str], set[str]], cache: dict, medi: dict, args: argparse.Namespace) -> dict:
    result = deepcopy(similar)
    for patient_key, visits in tqdm(similar.items(), desc="Filtering boundary medications", unit="patient"):
        pid = int(patient_key.removeprefix("patient_"))
        for visit_key, neighbours in visits.items():
            vid = int(visit_key.removeprefix("visit_"))
            boundary = {m for m, p in probabilities[(pid, vid)].items()
                        if args.boundary_low <= p <= args.boundary_high}
            history = set(map(int, valid[pid][vid - 1][2])) if vid and not args.no_history_rescue else set()
            current_codes = diagnoses.get(visit_hadm[(pid, vid)], [])
            for row in result[patient_key][visit_key]:
                tpid = int(str(row["train_patient_id"]).removeprefix("patient_"))
                tvid = int(str(row["train_visit_index"]).removeprefix("visit_"))
                train_hadm = hadm_id(train, tpid, tvid, "TRAIN")
                rejected = set()
                for med_id in boundary & set(map(int, row["Medications (from train visit)"])):
                    if not 0 <= med_id < len(medication_codes):
                        raise IndexError(f"Medication ID out of range: {med_id}")
                    if med_id not in history and not supported(drugs.get((train_hadm, medication_codes[med_id]), set()),
                                                               cache, current_codes, medi, args.medi_profile,
                                                               args.diagnosis_match):
                        rejected.add(med_id)
                row["Medications (from train visit)"] = [m for m in row["Medications (from train visit)"]
                                                          if int(m) not in rejected]
    return result


def main() -> None:
    args = parse_args()
    required = (args.predictions, args.train_records, args.valid_records, args.medi_c, args.ndc_atc,
                args.ndc_rxcui, args.prescriptions, args.diagnoses_icd, args.code_mapping,
                args.clinical_ingredient_cache)
    missing = [str(p) for p in required if not p.is_file()]
    if missing: raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))
    print("Loading training records, validation records, and predictions...")
    train, valid = normal_records(load_records(args.train_records)), normal_records(load_records(args.valid_records))
    probabilities = parse_predictions(load_json(args.predictions), valid)
    mapping = load_json(args.code_mapping)
    medication_codes = list(mapping["med"])
    print(f"Training patients: {len(train)}; validation patients: {len(valid)} (all processed)")
    similar = retrieve_all(valid, build_train_db(train), args.top_k)
    visit_hadm = {(pid, vid): hadm_id(valid, pid, vid, "VALID") for pid, patient in enumerate(valid)
                  for vid in range(len(patient))}
    needed_pairs = set()
    for patient_key, visits in similar.items():
        pid = int(patient_key.removeprefix("patient_"))
        for visit_key, rows in visits.items():
            vid = int(visit_key.removeprefix("visit_"))
            boundary = {m for m, p in probabilities[(pid, vid)].items() if args.boundary_low <= p <= args.boundary_high}
            for row in rows:
                tpid = int(str(row["train_patient_id"]).removeprefix("patient_")); tvid = int(str(row["train_visit_index"]).removeprefix("visit_"))
                for m in boundary & set(map(int, row["Medications (from train visit)"])):
                    if not 0 <= m < len(medication_codes): raise IndexError(f"Medication ID out of range: {m}")
                    needed_pairs.add((hadm_id(train, tpid, tvid, "TRAIN"), medication_codes[m]))
    print("Loading MEDI, diagnosis, and prescription evidence...")
    ndc_to_rxcui, clinical_to_atc = load_identity(args.ndc_rxcui, args.ndc_atc)
    cache_payload = load_json(args.clinical_ingredient_cache)
    cache = cache_payload.get("clinical_rxcuis", cache_payload)
    refined = refine(similar, train, valid, probabilities, medication_codes,
                     collect_diagnoses(args.diagnoses_icd, set(visit_hadm.values())), visit_hadm,
                     collect_drugs(args.prescriptions, needed_pairs, ndc_to_rxcui, clinical_to_atc), cache,
                     load_medi(args.medi_c), args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f: json.dump(refined, f, ensure_ascii=False, indent=2)
    print(f"Completed: {args.output}")


if __name__ == "__main__":
    main()
