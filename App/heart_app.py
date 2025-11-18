import gradio as gr
import skops.io as sio

# Load the pipeline
pipe = sio.load("./Model/heart_pipeline.skops", trusted=True)

def predict_heart(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    """Predict heart disease based on patient features."""
    
    # Create feature list in the exact order expected by the dataframe/pipeline
    features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    
    # Predict
    prediction = pipe.predict([features])[0]
    
    label = f"Predicted Condition: {'Heart Disease' if prediction == 1 else 'No Disease'}"
    return label

# Define inputs matching the dataset columns
inputs = [
    gr.Slider(29, 80, step=1, label="Age"),
    gr.Radio(["Female", "Male"], type="index", label="Sex"),
    gr.Radio(["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"], type="index", label="Chest Pain Type (CP)"),
    gr.Slider(90, 200, step=1, label="Resting Blood Pressure (trestbps)"),
    gr.Slider(100, 600, step=1, label="Cholesterol (chol)"),
    gr.Radio(["False", "True"], type="index", label="Fasting Blood Sugar > 120 mg/dl (fbs)"),
    gr.Radio(["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"], type="index", label="Resting ECG (restecg)"),
    gr.Slider(60, 220, step=1, label="Max Heart Rate (thalach)"),
    gr.Radio(["No", "Yes"], type="index", label="Exercise Induced Angina (exang)"),
    gr.Slider(0, 6.2, step=0.1, label="ST Depression (oldpeak)"),
    gr.Radio(["Upsloping", "Flat", "Downsloping"], type="index", label="Slope of Peak Exercise ST"),
    gr.Slider(0, 3, step=1, label="Number of Major Vessels (ca)"),
    gr.Radio(["Normal", "Fixed Defect", "Reversable Defect"], type="index", label="Thalassemia (thal)"),
]

outputs = [gr.Label(num_top_classes=2)]

# Example data (ensure these match the input types/indices)
examples = [
    [69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0],
    [60, 0, 0, 150, 240, 0, 0, 171, 0, 0.9, 0, 0, 0],
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