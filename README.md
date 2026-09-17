<div align="center">
<img src="GiantMed_Logo.png" alt="GiantMed Logo" width="240">
<h2>Awaken the Giant: Activating LLMs via Deep Model Guidance for Boundary-Aware Medication Recommendation</h2>

<p><strong>Accepted at KDD 2026</strong></p>

<p><strong>Hang Lv<sup>1</sup>, Zixuan Guo<sup>1</sup>, Yanchao Tan<sup>1,*</sup>, Wanzi Shao<sup>1</sup>, Hengyu Zhang<sup>2</sup>, Carl Yang<sup>3</sup></strong></p>

<p>
<sup>1</sup>Fuzhou University &nbsp;&nbsp;
<sup>2</sup>Macquarie University &nbsp;&nbsp;
<sup>3</sup>Emory University
</p>

<p><sup>*</sup>Corresponding author</p>

[![Paper](https://img.shields.io/badge/Paper-ACM%20Digital%20Library-CB3837?style=flat-square)](https://dl.acm.org/doi/abs/10.1145/3770854.3780297)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/lvhangkenn/GiantMed)
![Conference](https://img.shields.io/badge/Conference-KDD%202026-F4B400?style=flat-square)
![Task](https://img.shields.io/badge/Task-Medication%20Recommendation-2E8B57?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
</div>

## Abstract

Accurate and safe medication recommendations from Electronic Health Records (EHRs) are essential for clinical decision support. While Large Language Models (LLMs) have shown strong semantic reasoning capabilities in healthcare, they tend to make coarse binary predictions, overlooking medications near the decision boundary and leading to overprescription. In contrast, deep models offer fine-grained probability outputs but lack the contextual reasoning needed for complex boundary cases. We propose GiantMed, a boundary-aware medication recommendation framework that activates LLM reasoning under deep-model guidance. GiantMed uses a deep model to identify boundary medications and directs the LLM to focus on these clinically ambiguous yet informative cases. It further augments boundary-medication evidence by retrieving relevant historical EHRs and incorporating Drug–Drug Interaction (DDI) constraints. The final recommendation combines LLM-refined boundary medications with confident deep-model predictions. Experiments on two real-world EHR datasets show that GiantMed achieves state-of-the-art accuracy while reducing DDI rates.

## Framework

<p align="center">
  <img src="GiantMed_Framework.png" alt="Overview of the GiantMed framework" width="95%">
</p>

## Full Pipeline

### Step 1 — Install Dependencies

The main package versions used in our experiments are:

```text
pandas==1.3.0
dill==0.3.4
torch==1.8.0
rdkit==2021.03.4
scikit-learn==0.24.2
numpy==1.21.1
transformers==4.51.0
```

MoleRec additionally requires PyTorch Geometric and OGB. MedAlign additionally requires the Python Optimal Transport package (`POT`). Install versions compatible with your PyTorch and CUDA environment.

### Step 2 — Preprocess the EHR Data

Place the authorized raw MIMIC-III files and medication-mapping files under `data/`, then run:

```bash
cd data
python process.py
```

<details>
<summary><strong>Expected raw-data structure</strong></summary>

```text
data/
└── raw/MIMIC-III/
    ├── DIAGNOSES_ICD.csv
    ├── PRESCRIPTIONS.csv
    ├── PROCEDURES_ICD.csv
    ├── drug-atc.csv
    ├── drug-DDI.csv
    ├── idx2SMILES.pkl
    ├── ndc2atc_level4.csv
    └── ndc2RXCUI.txt
```
</details>

<details>
<summary><strong>Expected processed-data structure</strong></summary>

```text
data/
└── ready/MIMIC-III/
    ├── atc3toSMILES.pkl
    ├── ddi_A_final.pkl
    ├── ddi_mask_H.pkl
    ├── drug_smile.pkl
    ├── drug_text_embs.pkl
    ├── ehr_adj_final.pkl
    ├── records_final.pkl
    ├── smile_sub_b.pkl
    ├── smile_sub_degree_b.pkl
    ├── smile_sub_recency_b.pkl
    ├── smile_sub_voc_b.pkl
    └── voc_final.pkl
```
</details>

### Step 3 — Generate Deep-Model Probabilities

Released checkpoints and inference code for SafeDrug, MoleRec, DEPOT, and MedAlign are provided in `Deep_Model/`. Each `main_*.py` script loads its corresponding checkpoint, performs visit-level inference, and generates the probability file used by the LLM refinement stage.

```text
GiantMed/
├── data/
├── Deep_Model/
│   ├── model/
│   ├── modules/
│   ├── ckpt_DEPOT.model
│   ├── ckpt_MedAlign.model
│   ├── ckpt_MoleRec.model
│   ├── ckpt_SafeDrug.model
│   ├── main_DEPOT.py
│   ├── main_MedAlign.py
│   ├── main_MoleRec.py
│   └── main_SafeDrug.py
├── LLM/
└── README.md
```

Run the desired deep model from `Deep_Model/`:

```bash
cd GiantMed/Deep_Model

# SafeDrug
python main_SafeDrug.py

# MoleRec
python main_MoleRec.py

# DEPOT
python main_DEPOT.py

# MedAlign
python main_MedAlign.py
```

The prediction files are saved to `data/LLM-data/MIMIC-III/`:

```text
data/LLM-data/MIMIC-III/
├── SafeDrug_predictions.json
├── MoleRec_predictions.json
├── DEPOT_predictions.json
└── MedAlign_predictions.json
```

For example, `python main_MedAlign.py` generates `../data/LLM-data/MIMIC-III/MedAlign_predictions.json`.

>Running MedAlign also generates `../data/LLM-data/MIMIC-III/records_train.pkl` and `../data/LLM-data/MIMIC-III/records_valid.pkl`. These files are used to retrieve similar visits and to run the LLM refinement pipeline on the validation set.

The prediction files have the following format:

```json
[
  {
    "patient_id": 0,
    "visits": [
      {
        "visit_id": 0,
        "actual": [1, 5, 10],
        "predicted": [[10, 0.9273410439491272], [5, 0.8912528157234192], [35, 0.7821558713912964]]
      }
    ]
  }
]
```

`actual` stores the ground-truth medication IDs for evaluation. `predicted` stores `[medication_id, probability]` pairs in descending order of probability.

> Ensure that each checkpoint matches the vocabulary and dataset configuration used for inference.

### Step 4 — Generate the DDI Report

Generate a visit-level report containing all 131 medication IDs and their DDI relationships for the validation set. Ensure that `records_valid.pkl` and `ddi_A_final.pkl` are available in `../data/LLM-data/MIMIC-III/` and `../data/ready/MIMIC-III/` respectively, then run:

```bash
cd GiantMed/LLM
python generate_ddi_report.py
```

The report is saved to `../data/LLM-data/MIMIC-III/ddi_report.txt`.

### Step 5 — Generate similar visit and Run LLM Refinement

The required DDI evidence is provided in `data/LLM-data/MIMIC-III/`. First generate refined similar visits, then run the LLM refinement pipeline. The following commands use MedAlign as an example; replace `MedAlign` with `SafeDrug`, `MoleRec`, or `DEPOT` as needed.

```bash
cd GiantMed/LLM

python similar_ehrs.py \
  --predictions ../data/LLM-data/MIMIC-III/MedAlign_predictions.json \
  --top-k 3 \
  --boundary-low 0.3 \
  --boundary-high 0.7

python LLM_refine.py \
  --prob-json-path ../data/LLM-data/MIMIC-III/MedAlign_predictions.json \
  --boundary-low 0.3 \
  --boundary-high 0.7
```

## Citation

If you find GiantMed useful in your research, please cite our paper:

```bibtex
@inproceedings{lv2026awaken,
  title={Awaken the Giant: Activating LLMs via Deep Model Guidance for Boundary-aware Medication Recommendation},
  author={Lv, Hang and Guo, Zixuan and Tan, Yanchao and Shao, Wanzi and Zhang, Hengyu and Yang, Carl},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1},
  pages={1030--1041},
  year={2026}
}
```

## Acknowledgements

We thank the authors of [DEPOT](https://github.com/xmed-lab/DrugRec) for making their preprocessing implementation publicly available.
