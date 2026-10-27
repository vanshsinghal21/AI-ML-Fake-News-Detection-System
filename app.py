
from flask import Flask, render_template, request
from lime.lime_text import LimeTextExplainer
from pathlib import Path
import joblib
import shap

# =====================================================
# SatyaCheck - SVM + TF-IDF + LIME + SHAP
# =====================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

model = joblib.load(MODEL_DIR / "svm_fake_news_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "svm_tfidf_vectorizer.pkl")

# Probability-column order must match model.classes_.
MODEL_CLASSES = list(model.classes_)

CLASS_NAMES = []
for label in MODEL_CLASSES:
    if label == 0:
        CLASS_NAMES.append("FAKE NEWS")
    elif label == 1:
        CLASS_NAMES.append("REAL NEWS")
    else:
        CLASS_NAMES.append(str(label))

# Split on whitespace to avoid splitting Hindi characters.
lime_explainer = LimeTextExplainer(
    class_names=CLASS_NAMES,
    split_expression=r"\s+",
    bow=True
)


# =====================================================
# Shared probability function for LIME and SHAP
# =====================================================

def predict_proba_for_text(texts):
    vectors = vectorizer.transform(list(texts))
    return model.predict_proba(vectors)


# =====================================================
# SHAP explainer
# =====================================================

# Regex tokenizer splits on whitespace and preserves Hindi words.
shap_masker = shap.maskers.Text(r"\s+")

shap_explainer = shap.Explainer(
    predict_proba_for_text,
    shap_masker,
    output_names=CLASS_NAMES
)


# =====================================================
# LIME explanation
# =====================================================

def explain_with_lime(content, class_index):
    explanation = lime_explainer.explain_instance(
        content,
        predict_proba_for_text,
        labels=[class_index],
        num_features=10,
        num_samples=300
    )

    items = []

    for word, weight in explanation.as_list(label=class_index):
        items.append({
            "word": word,
            "weight": round(float(weight), 4),
            "direction": (
                "Supports this prediction"
                if weight > 0
                else "Opposes this prediction"
            )
        })

    return items


# =====================================================
# SHAP explanation
# =====================================================

def explain_with_shap(content, class_index):
    # Limit evaluations to keep the request reasonably bounded.
    result = shap_explainer(
        [content],
        max_evals=50
    )

    class_explanation = result[0, :, class_index]

    items = []

    for token, value in zip(
        class_explanation.data,
        class_explanation.values
    ):
        word = str(token).strip()
        weight = float(value)

        if not word:
            continue

        items.append({
            "word": word,
            "weight": round(weight, 5),
            "direction": (
                "Supports this prediction"
                if weight > 0
                else "Opposes this prediction"
            )
        })

    # Show the largest absolute contributions first.
    items.sort(key=lambda item: abs(item["weight"]), reverse=True)

    return items[:10]


# =====================================================
# Home page and prediction
# =====================================================

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    explanation = None
    shap_explanation = None
    error = None
    title = ""
    text = ""

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        text = request.form.get("text", "").strip()

        if len(title) < 5:
            error = "Please enter a headline of at least 5 characters."

        elif len(text) < 30:
            error = "Please enter at least 30 characters of news text."

        else:
            try:
                content = title + " " + text

                probabilities = predict_proba_for_text([content])[0]

                class_index = int(probabilities.argmax())
                prediction_value = MODEL_CLASSES[class_index]

                prediction = CLASS_NAMES[class_index]
                confidence = round(
                    float(probabilities[class_index]) * 100,
                    2
                )

                # Each explanation explains the same SVM prediction.
                explanation = explain_with_lime(
                    content,
                    class_index
                )

                try:
                    shap_explanation = explain_with_shap(
                        content,
                        class_index
                    )
                except Exception:
                    app.logger.exception("SHAP explanation failed")
                    shap_explanation = None

            except Exception:
                app.logger.exception("Prediction failed")
                error = (
                    "Unable to analyze this news. "
                    "Check the model files and dependencies."
                )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        explanation=explanation,
        shap_explanation=shap_explanation,
        error=error,
        title=title,
        text=text
    )


# =====================================================
# Start Flask application
# =====================================================

if __name__ == "__main__":
    app.run(
        debug=False,
        host="127.0.0.1",
        port=5000
    )

