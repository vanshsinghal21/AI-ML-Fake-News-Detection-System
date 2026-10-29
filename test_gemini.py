
import os
from google import genai

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not api_key:
    print("ERROR: GEMINI_API_KEY is not set.")
else:
    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model_name,
            contents="Reply with exactly: Gemini connection successful."
        )

        print("SUCCESS: Gemini is connected.")
        print(response.text)

    except Exception as exc:
        print("GEMINI CONNECTION FAILED")
        print(type(exc).__name__)
        print(str(exc))
