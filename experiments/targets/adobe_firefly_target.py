import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.moderated_prompter import build_sudo_jailbreak_prompt


DEFAULT_INPUT_PARQUET = Path("experiments/outputs/realvisxl_pgd/realvisxl_adv_prompts.parquet")
DEFAULT_OUTPUT_DIR = Path("experiments/outputs/adobe_firefly_results")
DEFAULT_COMPAT_OUTPUT = Path("test/discrete/moderated/adobe_firefly/adobe_firefly_results_plain.parquet")

DEFAULT_GENERATE_URL = "https://firefly-api.adobe.io/v3/images/generate"
DEFAULT_TOKEN_URL = "https://ims-na1.adobelogin.com/ims/token/v3"


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


def get_access_token(token_url, client_id, client_secret):
    """
    Recupera un access token Adobe tramite client credentials.
    Se si preferisce, è possibile passare direttamente ADOBE_FIREFLY_ACCESS_TOKEN
    e questa funzione non verrà usata.
    """
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "openid,AdobeID,session,additional_info,read_organizations,firefly_api,ff_apis",
    }

    response = requests.post(
        token_url,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=data,
        timeout=60,
    )

    response.raise_for_status()
    body = response.json()

    token = body.get("access_token")
    if not token:
        raise RuntimeError(f"Access token non trovato nella risposta Adobe: {body}")

    return token


def call_firefly_generate(
    prompt,
    api_key,
    access_token,
    generate_url,
    model_version,
    width,
    height,
):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": api_key,
        "Authorization": f"Bearer {access_token}",
    }

    if model_version:
        headers["x-model-version"] = model_version

    payload = {
        "prompt": prompt,
        "size": {
            "width": width,
            "height": height,
        },
    }

    response = requests.post(
        generate_url,
        headers=headers,
        json=payload,
        timeout=180,
    )

    return response


def find_image_url(response_body):
    """
    La risposta Firefly può variare in base alla versione/API.
    Questa funzione cerca in modo robusto un URL immagine nei campi più probabili.
    """
    if not isinstance(response_body, dict):
        return None

    # Casi comuni: outputs: [{ image: { url: ... } }]
    outputs = response_body.get("outputs")
    if isinstance(outputs, list) and outputs:
        first = outputs[0]

        if isinstance(first, dict):
            image = first.get("image")
            if isinstance(image, dict):
                for key in ("url", "presignedUrl", "downloadUrl"):
                    if image.get(key):
                        return image[key]

            for key in ("url", "presignedUrl", "downloadUrl"):
                if first.get(key):
                    return first[key]

    # Altri possibili formati
    images = response_body.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, dict):
            for key in ("url", "presignedUrl", "downloadUrl"):
                if first.get(key):
                    return first[key]
        if isinstance(first, str) and first.startswith("http"):
            return first

    for key in ("url", "presignedUrl", "downloadUrl"):
        if response_body.get(key):
            return response_body[key]

    return None


def download_image(image_url, out_path):
    response = requests.get(image_url, timeout=180)
    response.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(response.content)


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

        # Campi extra
        "record_id": idx,
        "target_model": model_id,
        "raw_response": json.dumps(raw_response, ensure_ascii=False) if raw_response is not None else None,
    }


def main(
    input_file,
    output_dir,
    compat_output,
    generate_url,
    token_url,
    model_version,
    limit=10,
    width=1024,
    height=1024,
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

    api_key = os.getenv("ADOBE_FIREFLY_API_KEY") or os.getenv("FIREFLY_SERVICES_CLIENT_ID")
    access_token = os.getenv("ADOBE_FIREFLY_ACCESS_TOKEN") or os.getenv("FIREFLY_SERVICES_ACCESS_TOKEN")
    client_secret = os.getenv("ADOBE_FIREFLY_CLIENT_SECRET") or os.getenv("FIREFLY_SERVICES_CLIENT_SECRET")

    if not api_key:
        raise RuntimeError(
            "API key Adobe non trovata. Imposta ADOBE_FIREFLY_API_KEY "
            "oppure FIREFLY_SERVICES_CLIENT_ID."
        )

    if not access_token:
        if not client_secret:
            raise RuntimeError(
                "Access token Adobe non trovato. Imposta ADOBE_FIREFLY_ACCESS_TOKEN "
                "oppure FIREFLY_SERVICES_ACCESS_TOKEN. In alternativa imposta "
                "ADOBE_FIREFLY_CLIENT_SECRET/FIREFLY_SERVICES_CLIENT_SECRET per generarlo."
            )

        print("Access token non trovato: provo a generarlo tramite client credentials...")
        access_token = get_access_token(
            token_url=token_url,
            client_id=api_key,
            client_secret=client_secret,
        )

    df = read_input(input_file)

    if "adv_prompt" not in df.columns:
        raise ValueError("Il file input deve contenere la colonna 'adv_prompt'.")

    if limit is not None:
        df = df.head(limit).copy()

    print(f"Input file: {input_file}")
    print(f"Output dir: {output_dir}")
    print(f"Compat output: {compat_output}")
    print(f"Generate URL: {generate_url}")
    print(f"Model version: {model_version}")
    print(f"Prompt da testare: {len(df)}")
    print(f"SUDO-Jailbreaking attivo: {use_sudo}")

    results = []

    for idx, row in df.iterrows():
        adv_prompt = row["adv_prompt"]
        prompt_to_send = build_sudo_jailbreak_prompt(adv_prompt) if use_sudo else adv_prompt
        category = row.get("category", "unknown")

        image_filename = f"{idx}_{sanitize_filename(category)}.png"
        out_path = image_dir / image_filename

        print(f"\n[FIREFLY] Record {idx}")
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
            response = call_firefly_generate(
                prompt=prompt_to_send,
                api_key=api_key,
                access_token=access_token,
                generate_url=generate_url,
                model_version=model_version,
                width=width,
                height=height,
            )

            try:
                response_body = response.json()
            except Exception:
                response_body = {"raw_text": response.text}

            if response.status_code >= 400:
                status = "client_error"
                error_message = f"HTTP {response.status_code}: {response.text}"

                # Se il servizio blocca il prompt, lo trattiamo come richiesta non accettata.
                text_blocked = True
                images_blocked = True
                image_filename = None

                print(f"[CLIENT ERROR] {error_message}")

            else:
                image_url = find_image_url(response_body)

                if image_url:
                    download_image(image_url, out_path)
                    status = "generated"
                    accepted_api_calls = 1
                    saved_image_path = str(out_path)
                    print(f"[generated] {out_path}")
                else:
                    status = "blocked_or_empty"
                    images_blocked = True
                    image_filename = None
                    error_message = "Nessun URL immagine trovato nella risposta Firefly."
                    print("[blocked_or_empty] Nessuna immagine trovata nella risposta")

        except Exception as e:
            error_message = str(e)
            status = "error"
            image_filename = None
            print(f"[ERROR] {error_message}")

        results.append(
            build_result_record(
                row=row,
                idx=idx,
                model_id=f"adobe_firefly:{model_version}",
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

    out_csv = output_dir / "adobe_firefly_results.csv"
    out_parquet = output_dir / "adobe_firefly_results.parquet"

    results_df.to_csv(out_csv, index=False)
    results_df.to_parquet(out_parquet, index=False)

    # File compatibile con gli script originali
    results_df.to_parquet(compat_output, index=False)

    compat_csv = compat_output.with_suffix(".csv")
    results_df.to_csv(compat_csv, index=False)

    print(f"\n[OK] Risultati CSV: {out_csv}")
    print(f"[OK] Risultati Parquet: {out_parquet}")
    print(f"[OK] Compat Parquet: {compat_output}")
    print(f"[OK] Compat CSV: {compat_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send adversarial prompts to Adobe Firefly Generate Image API."
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
        "--generate-url",
        default=DEFAULT_GENERATE_URL,
        help="Endpoint Adobe Firefly Generate Image API",
    )

    parser.add_argument(
        "--token-url",
        default=DEFAULT_TOKEN_URL,
        help="Endpoint Adobe IMS per ottenere access token",
    )

    parser.add_argument(
        "--model-version",
        default="image4_standard",
        help="Versione modello Firefly, es. image4_standard o image4_ultra",
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
        generate_url=args.generate_url,
        token_url=args.token_url,
        model_version=args.model_version,
        limit=limit,
        width=args.width,
        height=args.height,
        sleep_seconds=args.sleep_seconds,
        use_sudo=not args.disable_sudo,
    )