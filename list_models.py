import os
from google import genai

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY is not set.")
    raise SystemExit(1)

try:
    client = genai.Client(api_key=api_key)

    print("Models supporting text generation:\n")

    for model in client.models.list():
        if "generateContent" in (model.supported_actions or []):
            print(model.name)

except Exception as exc:
    print(type(exc).__name__)
    print(str(exc))