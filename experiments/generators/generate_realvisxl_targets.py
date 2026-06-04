import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.utils.image_generator import RealVisXL


DEFAULT_INPUT_JSON = Path("notebooks/MAIN/adversarial_test.json")
DEFAULT_OUTPUT_DIR = Path("data/images/realvisxl/reference")


def main(input_json, output_dir, limit=10, num_inference_steps=30):
    input_json = Path(input_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data if limit is None else data[:limit]

    print(f"Input JSON: {input_json}")
    print(f"Output dir: {output_dir}")
    print(f"Records da processare: {len(records)}")
    print("Caricamento RealVisXL...")

    generator = RealVisXL()
    rows = []

    for i, item in enumerate(records):
        unsafe_prompt = item["unsafe_prompt"]
        revised_prompt = item["revised_prompt"]
        category = item["category"]
        prompt_folder = item["prompt_folder"]
        image_file = item["image_file"]

        target_dir = output_dir / category / prompt_folder
        target_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"{image_file}.png"
        out_path = target_dir / out_name

        try:
            images = generator.generate_image(
                prompt=unsafe_prompt,
                num_images=1,
                num_inferences=num_inference_steps
            )

            image = images[0]
            image.save(out_path)

            rows.append({
                "record_id": i,
                "category": category,
                "base_prompt": item.get("base_prompt"),
                "unsafe_prompt": unsafe_prompt,
                "revised_prompt": revised_prompt,
                "prompt_folder": prompt_folder,
                "image_file": image_file,
                "generator_model": "SG161222/RealVisXL_V5.0",
                "target_image_path": str(out_path),
                "status": "generated",
                "error": None
            })

            print(f"[OK] {out_path}")

        except Exception as e:
            rows.append({
                "record_id": i,
                "category": category,
                "base_prompt": item.get("base_prompt"),
                "unsafe_prompt": unsafe_prompt,
                "revised_prompt": revised_prompt,
                "prompt_folder": prompt_folder,
                "image_file": image_file,
                "generator_model": "SG161222/RealVisXL_V5.0",
                "target_image_path": None,
                "status": "error",
                "error": str(e)
            })

            print(f"[ERR] record {i}: {e}")

    metadata_path = output_dir / "realvisxl_targets_metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    print(f"\nMetadata salvati in: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate RealVisXL target images from unsafe_prompt in adversarial_test.json"
    )

    parser.add_argument(
        "--input-json",
        default=str(DEFAULT_INPUT_JSON),
        help="Path al file adversarial_test.json"
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Cartella dove salvare le immagini RealVisXL"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Numero di prompt da processare. Usa -1 per processarli tutti."
    )

    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        help="Numero di inference steps per RealVisXL"
    )

    args = parser.parse_args()

    limit = None if args.limit == -1 else args.limit

    main(
        input_json=args.input_json,
        output_dir=args.output_dir,
        limit=limit,
        num_inference_steps=args.num_inference_steps
    )