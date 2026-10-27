
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import re

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

sample_text = (
    "सरकार ने आज नई शिक्षा नीति की घोषणा की। "
    "नई नीति का उद्देश्य शिक्षा की गुणवत्ता में सुधार करना है।"
)

print("Generating LIME explanation...")

def hindi_word_split(text):
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)

explainer = LimeTextExplainer(
    class_names=CLASS_NAMES,
    split_expression=hindi_word_split,
    bow=True
)

explanation = explainer.explain_instance(
    sample_text,
    predict_proba,
    num_features=8,
    num_samples=200
)

probabilities = predict_proba([sample_text])[0]
predicted_class = int(np.argmax(probabilities))

print("\nPrediction:", CLASS_NAMES[predicted_class])
print("Probabilities:")
for name, probability in zip(CLASS_NAMES, probabilities):
    print(f"{name}: {probability:.4f}")

print("\nWords influencing the prediction:")
for word, weight in explanation.as_list():
    print(f"{word}: {weight:.4f}")

explanation.save_to_file("lime_explanation.html")
print("\nSaved explanation to lime_explanation.html")
