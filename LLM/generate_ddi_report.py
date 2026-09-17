from pathlib import Path
import argparse
import pickle


def load_pickle(path: Path):
    try:
        import dill
        with path.open("rb") as f:
            return dill.load(f)
    except ImportError:
        with path.open("rb") as f:
            return pickle.load(f)


def get_ddi_pairs(ddi_matrix, num_medications: int):
    if len(ddi_matrix) < num_medications:
        raise ValueError(
            f"DDI matrix has {len(ddi_matrix)} rows, but {num_medications} medications were requested."
        )

    pairs = []
    for i in range(num_medications):
        for j in range(i + 1, num_medications):
            if ddi_matrix[i][j] != 0 or ddi_matrix[j][i] != 0:
                pairs.append((i, j))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Write the complete medication set and its DDI pairs for every validation visit."
    )
    parser.add_argument("--records", default=Path("../data/LLM-data/MIMIC-III/records_valid.pkl"))
    parser.add_argument("--ddi", default=Path("../data/ready/MIMIC-III/ddi_A_final.pkl"))
    parser.add_argument("--output", default=Path("../data/LLM-data/MIMIC-III/ddi_report.txt"))
    parser.add_argument("--num-medications", type=int, default=131)
    args = parser.parse_args()

    records = load_pickle(Path(args.records))
    ddi_matrix = load_pickle(Path(args.ddi))
    medications = list(range(args.num_medications))
    ddi_pairs = get_ddi_pairs(ddi_matrix, args.num_medications)

    visit_count = 0
    with Path(args.output).open("w", encoding="utf-8") as f:
        for patient_idx, patient in enumerate(records):
            f.write(f"Patient {patient_idx}:\n")
            for visit_idx in range(len(patient)):
                f.write(f"  Visit {visit_idx}: Medications (All) {medications}\n")
                f.write("    DDI Pairs:\n")
                for med_i, med_j in ddi_pairs:
                    f.write(f"      ({med_i}, {med_j})\n")
                visit_count += 1

    print(f"Saved {visit_count} visits and {len(ddi_pairs)} DDI pairs per visit to: {args.output}")


if __name__ == "__main__":
    main()
