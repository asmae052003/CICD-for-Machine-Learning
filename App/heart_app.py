import gradio as gr
import skops.io as sio
import pandas as pd

# =====================================
# 1) Chargement du pipeline SKOPS
# =====================================

MODEL_PATH = "./Model/heart_pipeline.skops"

# IMPORTANT : skops >=0.10 exige un argument nommé
trusted_types = sio.get_untrusted_types(file=MODEL_PATH)

pipe = sio.load(MODEL_PATH, trusted=trusted_types)


# =====================================
# 2) Fonction de prédiction
# =====================================

def predict_heart(age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal):

    """
    Prédire la maladie cardiaque à partir des caractéristiques patient.
    """

    df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    pred = pipe.predict(df)[0]

    # Probabilités (si dispo)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(df)[0]
        proba_dict = {
            "No Disease": round(proba[0] * 100, 1),
            "Heart Disease": round(proba[1] * 100, 1),
        }
    else:
        proba_dict = {}

    result = "❤️ Heart Disease Detected" if pred == 1 else "💚 No Heart Disease"

    return result, proba_dict


# =====================================
# 3) Construction de l'Interface Gradio
# =====================================

inputs = [
    gr.Slider(29, 80, step=1, label="Age"),
    gr.Radio(["Female", "Male"], type="index", label="Sex (0=F, 1=M)"),
    gr.Radio(
        ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"],
        type="index",
        label="Chest Pain Type (CP)"
    ),
    gr.Slider(90, 200, step=1, label="Resting Blood Pressure (trestbps)"),
    gr.Slider(100, 600, step=1, label="Cholesterol (chol)"),
    gr.Radio(["False", "True"], type="index", label="Fasting Blood Sugar (fbs)"),
    gr.Radio(
        ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"],
        type="index",
        label="Resting ECG (restecg)"
    ),
    gr.Slider(60, 220, step=1, label="Max Heart Rate (thalach)"),
    gr.Radio(["No", "Yes"], type="index", label="Exercise Induced Angina (exang)"),
    gr.Slider(0, 6.2, step=0.1, label="ST Depression (oldpeak)"),
    gr.Radio(["Upsloping", "Flat", "Downsloping"], type="index", label="Slope"),
    gr.Slider(0, 3, step=1, label="Major Vessels (ca)"),
    gr.Radio(["Normal", "Fixed Defect", "Reversible Defect"], type="index", label="Thal (thal)"),
]

outputs = [
    gr.Label(label="Prediction"),
    gr.Label(label="Model Confidence (%)")
]

examples = [
    [69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0],
    [60, 0, 2, 150, 240, 0, 0, 171, 0, 0.9, 0, 0, 0]
]

title = "❤️ Heart Disease Classification App"
description = """
This app predicts the likelihood of **heart disease** based on clinical features.
Built with **CI/CD – GitHub Actions** and deployed to **Hugging Face Spaces**.
"""
article = """
**Dataset**: UCI Heart Disease  
**Model**: RandomForestClassifier inside a scikit-learn Pipeline  
**Pipeline**: Training → Evaluation → CML Report → Auto-Commit to update branch → Deployment  
"""

demo = gr.Interface(
    fn=predict_heart,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
