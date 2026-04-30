import os
import base64
from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyDvH1vznhecKK-GZI543eHkULQ5L1vOG0k")

MODEL = "gemini-3.1-flash-image-preview"
PROMPT = "A photorealistic red apple on a wooden table, soft natural lighting, high detail"

print(f"Model : {MODEL}")
print(f"Prompt: {PROMPT}")
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
    max_output_tokens=32768,
    response_modalities=["TEXT", "IMAGE"],
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="OFF"),
    ],
    image_config=types.ImageConfig(
        aspect_ratio="1:1",
    ),
)

response = client.models.generate_content(
    model=MODEL,
    contents=contents,
    config=config,
)

image_bytes = None
if response.candidates:
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            image_bytes = base64.b64decode(part.inline_data.data)
            break

if image_bytes:
    out = "test_gemini_output.png"
    with open(out, "wb") as f:
        f.write(image_bytes)
    print(f"OK — immagine salvata: {out}  ({len(image_bytes):,} bytes)")
else:
    print("FAIL — nessuna immagine ricevuta (bloccato o errore)")
