
import torch
import shap
import numpy as np
import html
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "shivam123143/satyacheck-muril"
CLASS_NAMES = ["FAKE NEWS", "REAL NEWS"]

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()


def predict_proba(texts):
    inputs = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)

    return probabilities.cpu().numpy()


sample_text = "सरकार ने आज नई शिक्षा नीति की घोषणा की।"

print("Creating SHAP explainer...")

masker = shap.maskers.Text(tokenizer)

explainer = shap.Explainer(
    predict_proba,
    masker,
    output_names=CLASS_NAMES
)

print("Generating explanation...")

explanation = explainer(
    [sample_text],
    max_evals=30
)

probabilities = predict_proba([sample_text])[0]
predicted_class = int(np.argmax(probabilities))

print("\nPrediction:", CLASS_NAMES[predicted_class])
print("\nPrediction probabilities:")

for name, probability in zip(CLASS_NAMES, probabilities):
    print(f"{name}: {probability:.4%}")

print("\nSHAP explanation generated successfully.")
print("Tokens:", explanation.data[0])

# Explain the predicted class
class_index = predicted_class
shap_values = explanation[0, :, class_index]

print("\nWord contributions:")
for token, value in zip(shap_values.data, shap_values.values):
    token = str(token).strip()
    if token:
        print(f"{token}: {float(value):+.6f}")

# Generate SHAP interactive HTML
print("\nCreating interactive HTML...")

shap_html = shap.plots.text(
    shap_values,
    display=False
)

if hasattr(shap_html, "data"):
    shap_html = shap_html.data

# Add a readable page layout around the SHAP visualization
page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SatyaCheck - SHAP Explanation</title>
<style>
    body {{
        margin: 0;
        padding: 24px;
        font-family: Arial, "Noto Sans Devanagari", sans-serif;
        color: #222;
        background: #f5f7fb;
    }}
    .container {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 28px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 3px 18px rgba(0,0,0,0.08);
    }}
    h1 {{ margin-top: 0; font-size: 26px; }}
    .prediction {{
        padding: 16px;
        margin: 20px 0;
        background: #f0f4ff;
        border-radius: 8px;
    }}
    .article {{
        line-height: 1.9;
        overflow-wrap: anywhere;
    }}
    .note {{
        color: #555;
        font-size: 14px;
        line-height: 1.6;
    }}
    .visualization {{
        width: 100%;
        overflow-x: auto;
        margin-top: 20px;
    }}
</style>
</head>
<body>
<div class="container">
    <h1>SatyaCheck — SHAP Explanation</h1>

    <div class="prediction">
        <strong>Model prediction:</strong>
        {html.escape(CLASS_NAMES[predicted_class])}<br>
        <strong>Model confidence:</strong>
        {probabilities[predicted_class]:.2%}
    </div>

    <h2>Analyzed news text</h2>
    <p class="article">{html.escape(sample_text)}</p>

    <h2>Word-level explanation</h2>
    <p class="note">
        SHAP shows how words influence the selected model output.
        Positive and negative contributions are relative to the selected
        class and baseline. This explanation does not verify whether the
        news is factually true.
    </p>

    <div class="visualization">
        {shap_html}
    </div>
</div>
</body>
</html>
"""

with open("shap_explanation.html", "w", encoding="utf-8") as file:
    file.write(page_html)

print("\nSHAP test completed.")
print("Saved to shap_explanation.html")

