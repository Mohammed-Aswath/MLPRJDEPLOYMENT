import gradio as gr
import pickle
import numpy as np

# Load model and encoder
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Prediction function
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction_encoded = model.predict(input_data)
    prediction_label = le.inverse_transform(prediction_encoded)
    return prediction_label[0]

# Gradio interface
iface = gr.Interface(
    fn=predict_crop,
    inputs=[
        gr.Number(label="Nitrogen (N)"),
        gr.Number(label="Phosphorus (P)"),
        gr.Number(label="Potassium (K)"),
        gr.Number(label="Temperature (°C)"),
        gr.Number(label="Humidity (%)"),
        gr.Number(label="pH"),
        gr.Number(label="Rainfall (mm)")
    ],
    outputs=gr.Text(label="Recommended Crop"),
    title="Crop Recommendation using Random Forest",
    description="Enter soil and climate values to predict the best crop to grow."
)

iface.launch()
