import gradio as gr
import skops.io as sio

# ===============================
# Load the pipeline (new skops)
# ===============================
MODEL_PATH = "./Model/heart_pipeline.skops"
trusted = sio.get_untrusted_types(MODEL_PATH)
pipe = sio.load(MODEL_PATH, trusted=trusted)

# ===============================
# Maps because Gradio removed type="index"
# ===============================
SEX = ["Female", "Male"]
CP = ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"]
FBS = ["False", "True"]
RESTECG = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"]
EXANG = ["No", "Yes"]
SLOPE = ["Upsloping", "Flat", "Downsloping"]
THAL = ["Normal", "Fixed Defect", "Reversable Defect"]

def to_index(choice, array):
    """Convert label → index"""
    return array.index(choice)


# ===============================
# Predict Function
# ===============================
def predict_heart(age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal):

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
    return f"Predicted Condition: {'Heart Disease' if prediction == 1 else 'No Disease'}"


# ===============================
# Gradio Inputs
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

# Convert examples (index → label)
examples = [
    [69, SEX[1], CP[0], 160, 234, FBS[1], RESTECG[2], 131, EXANG[0], 0.1, SLOPE[1], 1, THAL[0]],
    [60, SEX[0], CP[0], 150, 240, FBS[0], RESTECG[0], 171, EXANG[0], 0.9, SLOPE[0], 0, THAL[0]],
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
