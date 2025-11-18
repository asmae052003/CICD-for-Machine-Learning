import gradio as gr
import skops.io as sio

MODEL_PATH = "./Model/heart_pipeline.skops"


# ===============================
# Load pipeline (compatible nouvelles / anciennes versions skops)
# ===============================
def load_pipeline():
    try:
        # 🔹 Nouvelle méthode (skops >= 0.10)
        trusted_types = sio.get_untrusted_types(file=MODEL_PATH)
        return sio.load(MODEL_PATH, trusted=trusted_types)
    except TypeError:
        # 🔹 Anciennes versions de skops (au cas où, en local par exemple)
        try:
            return sio.load(MODEL_PATH, trusted=True)
        except TypeError:
            return sio.load(MODEL_PATH)


pipe = load_pipeline()

# ===============================
# Définitions des choix (on remplace type="index")
# ===============================
SEX = ["Female", "Male"]
CP = ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"]
FBS = ["False", "True"]
RESTECG = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"]
EXANG = ["No", "Yes"]
SLOPE = ["Upsloping", "Flat", "Downsloping"]
THAL = ["Normal", "Fixed Defect", "Reversable Defect"]


def to_index(choice, array):
    """Convert a label (string) to an index (int)."""
    return array.index(choice)


# ===============================
# Fonction de prédiction (même logique que ton code original)
# ===============================
def predict_heart(
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal,
):
    """Predict heart disease based on patient features."""

    features = [
        age,
        to_index(sex, SEX),
        to_index(cp, CP),
        trestbps,
        chol,
        to_index(fbs, FBS),
        to_index(restecg, RESTECG),
        thalach,
        to_index(exang, EXANG),
        oldpeak,
        to_index(slope, SLOPE),
        ca,
        to_index(thal, THAL),
    ]

    prediction = pipe.predict([features])[0]
    label = f"Predicted Condition: {'Heart Disease' if prediction == 1 else 'No Disease'}"
    return label


# ===============================
# Entrées Gradio (sans type="index")
# ===============================
inputs = [
    gr.Slider(29, 80, step=1, label="Age"),
    gr.Radio(SEX, label="Sex"),
    gr.Radio(CP, label="Chest Pain Type (CP)"),
    gr.Slider(90, 200, step=1, label="Resting Blood Pressure (trestbps)"),
    gr.Slider(100, 600, step=1, label="Cholesterol (chol)"),
    gr.Radio(FBS, label="Fasting Blood Sugar > 120 mg/dl (fbs)"),
    gr.Radio(RESTECG, label="Resting ECG (restecg)"),
    gr.Slider(60, 220, step=1, label="Max Heart Rate (thalach)"),
    gr.Radio(EXANG, label="Exercise Induced Angina (exang)"),
    gr.Slider(0, 6.2, step=0.1, label="ST Depression (oldpeak)"),
    gr.Radio(SLOPE, label="Slope of Peak Exercise ST"),
    gr.Slider(0, 3, step=1, label="Number of Major Vessels (ca)"),
    gr.Radio(THAL, label="Thalassemia (thal)"),
]

outputs = [gr.Label(num_top_classes=2)]

# ===============================
# Examples : on met les LABELS (sinon Gradio plante)
# ===============================
examples = [
    [
        69,
        "Male",
        "Typical Angina",
        160,
        234,
        "True",
        "Left ventricular hypertrophy",
        131,
        "No",
        0.1,
        "Flat",
        1,
        "Normal",
    ],
    [
        60,
        "Female",
        "Typical Angina",
        150,
        240,
        "False",
        "Normal",
        171,
        "No",
        0.9,
        "Upsloping",
        0,
        "Normal",
    ],
]

title = "Heart Disease Classification"
description = "Enter patient details to predict the likelihood of heart disease."
article = "This app is part of a CI/CD for ML project."

gr.Interface(
    fn=predict_heart,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch()
