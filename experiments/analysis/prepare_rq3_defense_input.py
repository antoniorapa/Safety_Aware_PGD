import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("experiments/outputs/realvisxl_pgd/realvisxl_adv_prompts.parquet")
DEFAULT_OUTPUT = Path("experiments/outputs/analysis/rq3_realvisxl_prompts_for_defense.csv")


def read_table(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError("Formato non supportato. Usa .parquet o .csv")


def main(input_file, output_file):
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df = read_table(input_file)

    if "adv_prompt" not in df.columns:
        raise ValueError("Il file deve contenere la colonna adv_prompt.")

    if "category" not in df.columns:
        df["category"] = "unknown"

    out = pd.DataFrame()

    out["prompt"] = df["adv_prompt"]
    out["adv_prompt"] = df["adv_prompt"]
    out["category"] = df["category"]
    out["method"] = "RealVisXL_PGD_SUDO"
    out["condition"] = "RealVisXL_FullPipeline"

    optional_cols = [
        "base_prompt",
        "unsafe_prompt",
        "revised_prompt",
        "harm",
        "unsafe_replacements",
        "safe_replacements",
        "unsafe_words_gradient",
        "safe_words_gradient",
    ]

    for col in optional_cols:
        if col in df.columns:
            out[col] = df[col]

    out.to_csv(output_file, index=False)
    out.to_parquet(output_file.with_suffix(".parquet"), index=False)

    print(f"[OK] RQ3 defense input CSV: {output_file}")
    print(f"[OK] RQ3 defense input Parquet: {output_file.with_suffix('.parquet')}")
    print()
    print("Questo file va usato come input per gli script originali di difesa/ablation:")
    print("  notebooks/MAIN/ablation/4_score_classifiers.py")
    print("  notebooks/MAIN/ablation/5_score_llms.py")
    print("  notebooks/MAIN/ablation/6_analysis.py")
    print()
    print("Se quegli script hanno path hardcoded, creare una copia *_realvisxl.py")
    print("e impostare questo file come input.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare RealVisXL PGD prompts as input for RQ3 defense evaluation."
    )

    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT),
        help="File realvisxl_adv_prompts.parquet/csv.",
    )

    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_OUTPUT),
        help="Output CSV per gli script di difesa.",
    )

    args = parser.parse_args()

    main(
        input_file=args.input_file,
        output_file=args.output_file,
    )