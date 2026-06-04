import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.moderated_prompter import ModeratedPrompter, process_batches


DEFAULT_INPUT_JSON = Path("notebooks/MAIN/adversarial_test.json")
DEFAULT_REFERENCE_IMAGES_ROOT = Path("data/images/realvisxl/reference")
DEFAULT_OUTPUT_DIR = Path("experiments/outputs/realvisxl_pgd")
DEFAULT_CATEGORY_CONFIG = Path("src/configs/category_block_dict.json")


def parse_list_columns(df):
    """
    Converte eventuali colonne salvate come stringhe in vere liste Python.
    Nel JSON originale di solito sono già liste, ma questa funzione rende lo script più robusto.
    """
    list_cols = [
        "unsafe_replacements",
        "safe_replacements",
        "positions",
        "unsafe_words_gradient",
        "safe_words_gradient",
        "flatten_positions",
    ]

    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

    return df


def check_required_columns(df):
    required_cols = [
        "base_prompt",
        "unsafe_prompt",
        "revised_prompt",
        "category",
        "prompt_folder",
        "image_file",
        "flatten_positions",
        "unsafe_words_gradient",
        "safe_replacements",
        "unsafe_replacements",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            "Mancano colonne richieste nel dataframe: "
            + ", ".join(missing)
        )


def check_realvisxl_images(df, reference_images_root):
    """
    Controlla che per ogni record esista l'immagine RealVisXL nel path atteso:

    data/images/realvisxl/reference/<category>/<prompt_folder>/<image_file>.png
    """
    missing_images = []

    for _, row in df.iterrows():
        img_path = (
            reference_images_root
            / row["category"]
            / row["prompt_folder"]
            / f"{row['image_file']}.png"
        )

        if not img_path.exists():
            missing_images.append(str(img_path))

    return missing_images


def main(
    input_json,
    reference_images_root,
    output_dir,
    checkpoint_path,
    api_key,
    category_config,
    limit=10,
    batch_size=2,
    n_iterations=200,
    n_s=10,
    device="cuda",
):
    input_json = Path(input_json)
    reference_images_root = Path(reference_images_root)
    output_dir = Path(output_dir)
    checkpoint_path = Path(checkpoint_path)
    category_config = Path(category_config)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON non trovato: {input_json}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")

    if not category_config.exists():
        raise FileNotFoundError(f"Category config non trovato: {category_config}")

    with open(category_config, "r", encoding="utf-8") as f:
        category_block_dict = json.load(f)

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df = parse_list_columns(df)
    check_required_columns(df)

    if limit is not None:
        df = df.head(limit).copy()

    print(f"Input JSON: {input_json}")
    print(f"Reference images root: {reference_images_root}")
    print(f"Output dir: {output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Category config: {category_config}")
    print(f"Rows PGD: {len(df)}")
    print(f"Device richiesto: {device}")

    missing_images = check_realvisxl_images(df, reference_images_root)

    if missing_images:
        print("\n[ERRORE] Mancano alcune immagini RealVisXL.")
        print("Esempi di immagini mancanti:")

        for path in missing_images[:10]:
            print(path)

        print("\nPrima devi eseguire generate_realvisxl_targets.py sugli stessi record.")
        print("Esempio:")
        print(
            "python experiments\\generators\\generate_realvisxl_targets.py "
            f"--limit {len(df)} --num-inference-steps 30"
        )
        return

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Hai richiesto device='cuda', ma CUDA non è disponibile. "
            "Questo script va eseguito sul PC del collega con GPU NVIDIA, "
            "oppure usa --device cpu solo per test molto piccoli."
        )

    print("\nCaricamento ModeratedPrompter...")

    prompter = ModeratedPrompter(
        api_key=api_key,
        checkpoint_path=str(checkpoint_path),
        device=device,
    )

    # Questa è la modifica chiave:
    # process_batches continuerà a funzionare come nell'originale,
    # ma userà le immagini RealVisXL invece di quelle SD originali.
    prompter.reference_images_root = str(reference_images_root)

    final_parts = []

    for category in sorted(df["category"].unique()):
        df_cat = df[df["category"] == category].copy()

        if df_cat.empty:
            continue

        if category not in category_block_dict:
            print(f"[WARN] Categoria non trovata in category_block_dict: {category}. Skip.")
            continue

        all_unsafe_words = list(set().union(*df_cat["unsafe_words_gradient"]))

        print(f"\n[PGD] Categoria: {category}")
        print(f"Rows categoria: {len(df_cat)}")
        print(f"Unsafe words globali categoria: {len(all_unsafe_words)}")

        df_cat_adv = process_batches(
            df=df_cat,
            prompter=prompter,
            CAT=category,
            all_unsafe_words=all_unsafe_words,
            batch_size=batch_size,
            n_iterations=n_iterations,
            n_s=n_s,
            enable_focus=False,
            global_mod=0.5,
            tau_prime=0.4,
            category_dict=category_block_dict,
        )

        final_parts.append(df_cat_adv)

    if not final_parts:
        raise RuntimeError(
            "Nessun risultato PGD prodotto. "
            "Controlla categorie, input JSON e category_block_dict."
        )

    df_adv = pd.concat(final_parts, ignore_index=True)

    out_csv = output_dir / "realvisxl_adv_prompts.csv"
    out_parquet = output_dir / "realvisxl_adv_prompts.parquet"

    df_adv.to_csv(out_csv, index=False)
    df_adv.to_parquet(out_parquet, index=False)

    print(f"\n[OK] Salvato CSV: {out_csv}")
    print(f"[OK] Salvato Parquet: {out_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PGD using RealVisXL-generated target images."
    )

    parser.add_argument(
        "--input-json",
        default=str(DEFAULT_INPUT_JSON),
        help="Path ad adversarial_test.json",
    )

    parser.add_argument(
        "--reference-images-root",
        default=str(DEFAULT_REFERENCE_IMAGES_ROOT),
        help="Root delle immagini target RealVisXL",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Cartella output per i nuovi adv_prompt",
    )

    parser.add_argument(
        "--checkpoint-path",
        default="src/models/mlp_model_selected_layer_1.pth",
        help="Path al checkpoint del moderation classifier usato da ModeratedPrompter",
    )

    parser.add_argument(
        "--category-config",
        default=str(DEFAULT_CATEGORY_CONFIG),
        help="Path al file category_block_dict.json",
    )

    parser.add_argument(
        "--api-key",
        default="",
        help="OpenAI API key. Lascia vuoto se non serve nella fase PGD locale.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Numero di record da processare. Usa -1 per tutti.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size PGD.",
    )

    parser.add_argument(
        "--n-iterations",
        type=int,
        default=200,
        help="Numero iterazioni PGD.",
    )

    parser.add_argument(
        "--n-s",
        type=int,
        default=10,
        help="Numero candidati selezionati per prompt.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device per PGD. Consigliato cuda.",
    )

    args = parser.parse_args()

    limit = None if args.limit == -1 else args.limit

    main(
        input_json=args.input_json,
        reference_images_root=args.reference_images_root,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        api_key=args.api_key,
        category_config=args.category_config,
        limit=limit,
        batch_size=args.batch_size,
        n_iterations=args.n_iterations,
        n_s=args.n_s,
        device=args.device,
    )