import gradio as gr
import skops.io as sio

MODEL_PATH = "./Model/heart_pipeline.skops"


def load_pipeline():
    """
    Charge le pipeline sauvegardé avec skops, en gérant
    à la fois les versions <0.10 et >=0.10.
    """
    try:
        # skops >= 0.10 : trusted doit être une LISTE de types
        trusted = sio.get_untrusted_types(MODEL_PATH)
        return sio.load(MODEL_PATH, trusted=trusted)
    except TypeError:
        # skops < 0.10 : trusted=True (booléen) fonctionne encore
        return sio.load(MODEL_PATH, trusted=True)


pipe = load_pipeline()

# -----------------------
# Mappings texte -> codes
# -----------------------
SEX_MAP = {"Female": 0, "Male": 1}
CP_MAP = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal": 2,
    "Asymptomatic": 3,
}
FBS_MAP = {"False": 0, "True": 1}
RESTECG_MAP = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left ventricular hypertrophy": 2,
}
EXANG_MAP = {"No": 0, "Yes": 1}
SLOPE_MAP = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
THAL_MAP = {
    "Normal": 0,
    "Fixed Defect": 1,
    "Reversable Defect": 2,
}


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

    # Conversion des textes des Radio en codes numériques
    sex_val = SEX_MAP[sex]
    cp_val = CP_MAP[cp]
    fbs_val = FBS_MAP[fbs]
    restecg_val = RESTECG_MAP[restecg]
    exang_val = EXANG_MAP[exang]
    slope_val = SLOPE_MAP[slope]
    thal_val = THAL_MAP[thal]

    features = [
        age,
        sex_val,
        cp_val,
        trestbps,
        chol,
        fbs_val,
        restecg_val,
        thalach,
        exang_val,
        oldpeak,
        slope_val,
        ca,
        thal_val,
    ]

    prediction = pipe.predict([features])[0]
    label = f"Predicted Condition: {'Heart Disease' if prediction == 1 else 'No Disease'}"
    return label


# -----------------------
# Composants Gradio
# -----------------------
inputs = [
    gr.Slider(29, 80, step=1, label="Age"),
    gr.Radio(["Female", "Male"], label="Sex"),
    gr.Radio(
        ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"],
        label="Chest Pain Type (CP)",
    ),
    gr.Slider(90, 200, step=1, label="Resting Blood Pressure (trestbps)"),
    gr.Slider(100, 600, step=1, label="Cholesterol (chol)"),
    gr.Radio(["False", "True"], label="Fasting Blood Sugar > 120 mg/dl (fbs)"),
    gr.Radio(
        ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"],
        label="Resting ECG (restecg)",
    ),
    gr.Slider(60, 220, step=1, label="Max Heart Rate (thalach)"),
    gr.Radio(["No", "Yes"], label="Exercise Induced Angina (exang)"),
    gr.Slider(0, 6.2, step=0.1, label="ST Depression (oldpeak)"),
    gr.Radio(
        ["Upsloping", "Flat", "Downsloping"],
        label="Slope of Peak Exercise ST",
    ),
    gr.Slider(0, 3, step=1, label="Number of Major Vessels (ca)"),
    gr.Radio(
        ["Normal", "Fixed Defect", "Reversable Defect"],
        label="Thalassemia (thal)",
    ),
]

outputs = [gr.Label(num_top_classes=2)]

# ⚠️ Les examples utilisent maintenant les LABELS des Radio, pas des entiers
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
        "Atypical Angina",
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
