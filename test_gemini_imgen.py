import os
import base64
from google import genai
from google.genai import types

PROJECT_ID = "secret-antonym-494814-q9"
LOCATION = "us-central1"
MODEL = "gemini-2.0-flash-preview-image-generation"

# Vertex AI usa Application Default Credentials (gcloud auth application-default login)
# NON serve una api_key
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

PROMPT = "A photorealistic red apple on a wooden table, soft natural lighting, high detail"

print(f"Model  : {MODEL}")
print(f"Project: {PROJECT_ID} | Location: {LOCATION}")
print(f"Prompt : {PROMPT}")
print("Generating...")

contents = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text=PROMPT)]
    )
]

config = types.GenerateContentConfig(
    temperature=1,
    top_p=0.95,
    max_output_tokens=8192,
    response_modalities=["TEXT", "IMAGE"],
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="OFF"),
    ],
)

try:
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config,
    )

    image_bytes = None
    if response.candidates:
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                raw = part.inline_data.data
                image_bytes = base64.b64decode(raw) if isinstance(raw, str) else raw
                break

    if image_bytes:
        out = "test_gemini_output.png"
        with open(out, "wb") as f:
            f.write(image_bytes)
        print(f"OK — immagine salvata: {out}  ({len(image_bytes):,} bytes)")
    else:
        print("FAIL — nessuna immagine ricevuta")
        if response.candidates:
            print("finish_reason:", response.candidates[0].finish_reason)

except Exception as e:
    print(f"ERRORE: {e}")
