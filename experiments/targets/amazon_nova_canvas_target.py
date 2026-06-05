import argparse
import base64
import json
import sys
import time
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.moderated_prompter import build_sudo_jailbreak_prompt


DEFAULT_INPUT_PARQUET = Path("experiments/outputs/realvisxl_pgd/realvisxl_adv_prompts.parquet")
DEFAULT_OUTPUT_DIR = Path("experiments/outputs/amazon_nova_results")
DEFAULT_COMPAT_OUTPUT = Path("test/discrete/moderated/amazon_nova/amazon_nova_results_plain.parquet")


def read_input(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file non trovato: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError("Formato input non supportato. Usa .parquet oppure .csv")


def sanitize_filename(text, max_len=80):
    safe = "".join(
        c if c.isalnum() or c in (" ", "_", "-") else "_"
        for c in str(text)
    )
    safe = "_".join(safe.split())
    return safe[:max_len]


def call_nova_canvas(client, prompt, model_id, width, height, seed):
    body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": height,
            "width": width,
            "cfgScale": 8.0,
            "seed": seed
        }
    }

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response["body"].read())
    return response_body


def extract_and_save_image(response_body, out_path):
    images = response_body.get("images", [])

    if not images:
        return False

    image_base64 = images[0]
    image_bytes = base64.b64decode(image_base64)

    with open(out_path, "wb") as f:
        f.write(image_bytes)

    return True


def build_result_record(
    row,
    idx,
    model_id,
    status,
    image_filename,
    image_path,
    raw_response,
    error,
    api_calls,
    accepted_api_calls,
    text_blocked,
    images_blocked,
    sudo_prompt,
):
    return {
        # Prompt
        "adv_prompt": row.get("adv_prompt"),
        "sudo_prompt": sudo_prompt,

        # Campi principali originali/compatibili
        "category": row.get("category", "unknown"),
        "base_prompt": row.get("base_prompt"),
        "unsafe_prompt": row.get("unsafe_prompt"),
        "revised_prompt": row.get("revised_prompt"),

        # Campi immagine
        "image_filename": image_filename,
        "image_path": image_path,

        # Campi API/BR compatibili
        "api_calls": api_calls,
        "accepted_api_calls": accepted_api_calls,
        "api_call_per_soft_prompt": api_calls,
        "accepted_api_call_per_soft_prompt": accepted_api_calls,

        # Campi blocco/moderazione
        "text_blocked": text_blocked,
        "images_blocked": images_blocked,
        "status": status,
        "error": error,

        # Campi metriche da popolare successivamente tramite script originali
        "clip_score": None,
        "harm_level": None,

        # Campi extra utili
        "record_id": idx,
        "target_model": model_id,
        "raw_response": json.dumps(raw_response, ensure_ascii=False) if raw_response is not None else None,
    }


def main(
    input_file,
    output_dir,
    compat_output,
    region,
    model_id,
    limit=10,
    width=1024,
    height=1024,
    seed=0,
    sleep_seconds=1.0,
    use_sudo=True,
):
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    compat_output = Path(compat_output)

    image_dir = output_dir / "images"

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    compat_output.parent.mkdir(parents=True, exist_ok=True)

    df = read_input(input_file)

    if "adv_prompt" not in df.columns:
        raise ValueError("Il file input deve contenere la colonna 'adv_prompt'.")

    if limit is not None:
        df = df.head(limit).copy()

    print(f"Input file: {input_file}")
    print(f"Output dir: {output_dir}")
    print(f"Compat output: {compat_output}")
    print(f"Target model: {model_id}")
    print(f"Region: {region}")
    print(f"Prompt da testare: {len(df)}")
    print(f"SUDO-Jailbreaking attivo: {use_sudo}")

    client = boto3.client("bedrock-runtime", region_name=region)

    results = []

    for idx, row in df.iterrows():
        adv_prompt = row["adv_prompt"]
        prompt_to_send = build_sudo_jailbreak_prompt(adv_prompt) if use_sudo else adv_prompt
        category = row.get("category", "unknown")

        image_filename = f"{idx}_{sanitize_filename(category)}.png"
        out_path = image_dir / image_filename

        print(f"\n[NOVA] Record {idx}")
        print(f"Categoria: {category}")

        api_calls = 1
        accepted_api_calls = 0
        text_blocked = False
        images_blocked = False
        status = "unknown"
        error_message = None
        response_body = None
        saved_image_path = None

        try:
            response_body = call_nova_canvas(
                client=client,
                prompt=prompt_to_send,
                model_id=model_id,
                width=width,
                height=height,
                seed=seed,
            )

            image_saved = extract_and_save_image(response_body, out_path)

            if image_saved:
                status = "generated"
                accepted_api_calls = 1
                saved_image_path = str(out_path)
                print(f"[generated] {out_path}")
            else:
                status = "blocked_or_empty"
                images_blocked = True
                image_filename = None
                print("[blocked_or_empty] Nessuna immagine restituita")

            error_message = response_body.get("error", None)

        except ClientError as e:
            error_message = str(e)
            status = "client_error"
            text_blocked = True
            image_filename = None

            print(f"[CLIENT ERROR] {error_message}")

        except Exception as e:
            error_message = str(e)
            status = "error"
            image_filename = None

            print(f"[ERROR] {error_message}")

        results.append(
            build_result_record(
                row=row,
                idx=idx,
                model_id=model_id,
                status=status,
                image_filename=image_filename,
                image_path=saved_image_path,
                raw_response=response_body,
                error=error_message,
                api_calls=api_calls,
                accepted_api_calls=accepted_api_calls,
                text_blocked=text_blocked,
                images_blocked=images_blocked,
                sudo_prompt=prompt_to_send,
            )
        )

        time.sleep(sleep_seconds)

    results_df = pd.DataFrame(results)

    out_csv = output_dir / "amazon_nova_results.csv"
    out_parquet = output_dir / "amazon_nova_results.parquet"

    results_df.to_csv(out_csv, index=False)
    results_df.to_parquet(out_parquet, index=False)

    # File compatibile con struttura originale
    results_df.to_parquet(compat_output, index=False)

    compat_csv = compat_output.with_suffix(".csv")
    results_df.to_csv(compat_csv, index=False)

    print(f"\n[OK] Risultati CSV: {out_csv}")
    print(f"[OK] Risultati Parquet: {out_parquet}")
    print(f"[OK] Compat Parquet: {compat_output}")
    print(f"[OK] Compat CSV: {compat_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send adversarial prompts to Amazon Nova Canvas through AWS Bedrock."
    )

    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT_PARQUET),
        help="File .parquet o .csv contenente la colonna adv_prompt",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Cartella output per immagini e risultati",
    )

    parser.add_argument(
        "--compat-output",
        default=str(DEFAULT_COMPAT_OUTPUT),
        help="File compatibile con gli script originali di analisi",
    )

    parser.add_argument(
        "--region",
        default="us-east-1",
        help="Regione AWS Bedrock",
    )

    parser.add_argument(
        "--model-id",
        default="amazon.nova-canvas-v1:0",
        help="Model ID Bedrock per Amazon Nova Canvas",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Numero di prompt da testare. Usa -1 per tutti.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Larghezza immagine",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Altezza immagine",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed generazione",
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Pausa tra chiamate API",
    )

    parser.add_argument(
        "--disable-sudo",
        action="store_true",
        help="Invia l'adv_prompt senza applicare la fase SUDO-Jailbreaking.",
    )

    args = parser.parse_args()

    limit = None if args.limit == -1 else args.limit

    main(
        input_file=args.input_file,
        output_dir=args.output_dir,
        compat_output=args.compat_output,
        region=args.region,
        model_id=args.model_id,
        limit=limit,
        width=args.width,
        height=args.height,
        seed=args.seed,
        sleep_seconds=args.sleep_seconds,
        use_sudo=not args.disable_sudo,
    )