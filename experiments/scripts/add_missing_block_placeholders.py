import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Add placeholder rows for missing Gemini generation blocks."
    )
    parser.add_argument("--input-prompts", type=str, required=True)
    parser.add_argument("--merged-results", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--output-parquet", type=str, required=True)
    parser.add_argument("--missing-start", type=int, required=True)
    parser.add_argument("--missing-end", type=int, required=True)

    args = parser.parse_args()

    prompts = pd.read_parquet(args.input_prompts)
    merged = pd.read_csv(args.merged_results)

    if "original_index" not in prompts.columns:
        prompts = prompts.reset_index().rename(columns={"index": "original_index"})

    existing = set(merged["original_index"].astype(int).tolist())
    missing_range = list(range(args.missing_start, args.missing_end + 1))

    rows = []

    for idx in missing_range:
        if idx in existing:
            continue

        prompt_row = prompts[prompts["original_index"].astype(int) == idx]

        if prompt_row.empty:
            prompt_row = prompts.iloc[[idx]]

        row = prompt_row.iloc[0].to_dict()

        rows.append({
            "result_row_id": idx,
            "record_id": row.get("record_id", None),
            "original_index": idx,
            "category": row.get("category", None),
            "base_prompt": row.get("base_prompt", None),
            "unsafe_prompt": row.get("unsafe_prompt", None),
            "revised_prompt": row.get("revised_prompt", None),
            "adv_prompt": row.get("adv_prompt", None),
            "generated": False,
            "status": "missing_block_assumed_not_generated",
            "error": "Missing block file; no image generated in this block.",
            "image_filename": None,
            "image_path": None,
            "finish_reasons": "MISSING_RESULTS_FILE",
            "source_block": "missing_placeholder",
        })

    placeholder_df = pd.DataFrame(rows)
    final_df = pd.concat([merged, placeholder_df], ignore_index=True)
    final_df = final_df.sort_values("original_index").reset_index(drop=True)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output_csv, index=False)
    final_df.to_parquet(args.output_parquet, index=False)

    print("Original merged rows:", len(merged))
    print("Added placeholder rows:", len(placeholder_df))
    print("Final rows:", len(final_df))


if __name__ == "__main__":
    main()
