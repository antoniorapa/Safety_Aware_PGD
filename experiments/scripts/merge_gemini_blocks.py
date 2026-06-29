import argparse
from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple Gemini/Nano Banana generation blocks."
    )
    parser.add_argument("--blocks-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--expected-total", type=int, default=525)

    args = parser.parse_args()

    blocks_root = Path(args.blocks_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(blocks_root.glob("**/gemini_nano_banana_results.csv"))

    if not files:
        raise FileNotFoundError(f"No block CSV files found under {blocks_root}")

    dfs = []

    for file in files:
        print("Reading:", file)
        block_df = read_table(file)
        block_df["source_block"] = str(file.parent.name)
        dfs.append(block_df)

    merged = pd.concat(dfs, ignore_index=True)

    if "original_index" in merged.columns:
        merged = merged.sort_values("original_index").drop_duplicates(
            subset=["original_index"], keep="last"
        )

    merged = merged.reset_index(drop=True)

    output_csv = output_dir / "gemini_nano_banana_results_merged.csv"
    output_parquet = output_dir / "gemini_nano_banana_results_merged.parquet"

    merged.to_csv(output_csv, index=False)
    merged.to_parquet(output_parquet, index=False)

    print("Merged rows:", len(merged))
    print("Expected rows:", args.expected_total)
    print("Saved CSV:", output_csv)
    print("Saved Parquet:", output_parquet)


if __name__ == "__main__":
    main()
