import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Create a balanced subset of harmful prompts across categories."
    )
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--samples-per-category", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    if "category" not in df.columns:
        raise ValueError("Missing column: category")

    balanced = (
        df.groupby("category", group_keys=False)
        .apply(lambda x: x.sample(
            n=min(args.samples_per_category, len(x)),
            random_state=args.seed
        ))
        .reset_index(drop=True)
    )

    records = balanced.to_dict(orient="records")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print("Input records:", len(df))
    print("Output records:", len(records))
    print("Categories:")
    print(balanced["category"].value_counts())
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
