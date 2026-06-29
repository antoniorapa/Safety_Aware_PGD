import argparse
from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Unsupported format. Use .csv or .parquet")


def save_table(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("Unsupported output format. Use .csv or .parquet")


def main():
    parser = argparse.ArgumentParser(
        description="Select the top-3 adversarial prompts for each original prompt."
    )
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--output-parquet", type=str, required=True)
    parser.add_argument("--group-col", type=str, default="record_id")
    parser.add_argument("--clip-col", type=str, default="clip_score")
    parser.add_argument("--harm-col", type=str, default="harm_score")
    parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    df = read_table(Path(args.input_file)).reset_index(drop=True)

    if "original_index" not in df.columns:
        df.insert(0, "original_index", range(len(df)))

    if args.group_col not in df.columns:
        raise ValueError(f"Missing group column: {args.group_col}")

    sort_cols = [args.group_col]
    ascending = [True]

    if args.clip_col in df.columns:
        sort_cols.append(args.clip_col)
        ascending.append(False)

    if args.harm_col in df.columns:
        sort_cols.append(args.harm_col)
        ascending.append(False)

    selected = (
        df.sort_values(sort_cols, ascending=ascending)
        .groupby(args.group_col, group_keys=False)
        .head(args.top_k)
        .reset_index(drop=True)
    )

    save_table(selected, Path(args.output_csv))
    save_table(selected, Path(args.output_parquet))

    print("Input prompts:", len(df))
    print("Selected prompts:", len(selected))
    print("Groups:", selected[args.group_col].nunique())


if __name__ == "__main__":
    main()
