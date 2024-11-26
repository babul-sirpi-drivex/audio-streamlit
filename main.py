import streamlit as st
import os
import base64
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
        }} 
        header[data-testid="stHeader"] {{
            display: none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Chicken Disease Detection",  # Page title
    page_icon="🐓",  # Page icon
    layout="wide",  # Wide layout
    initial_sidebar_state="expanded",  # Expanded sidebar
)

current_directory = os.getcwd()
background_img_path = os.path.join(
    current_directory, "discover-savsat-Ag0fAuFtH6I-unsplash.jpg"
)
add_bg_from_local(background_img_path)

# Load the pre-trained model and label encoder
MODEL_PATH = "audio_classification_model.keras"
ENCODER_PATH = "label_encoder.pkl"
model = load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)


# Function to preprocess and predict
def predict_audio(file_path, model, encoder, n_mfcc=40):
    # Load and preprocess the audio file
    audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    features = np.mean(mfccs.T, axis=0)
    features = np.expand_dims(features, axis=0)  # Add batch dimension

    # Normalize features (ensure it matches your training preprocessing)
    features = features / np.max(features)

    # Predict
    predictions = model.predict(features)
    predicted_class_idx = np.argmax(predictions)
    predicted_label = encoder.inverse_transform([predicted_class_idx])[0]
    return predicted_label


# File uploader section
def upload_file():
    st.write("Upload an audio file (e.g., .wav) for disease prediction.")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("temp_audio.wav")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(uploaded_file, format="audio/wav")  # Play uploaded audio file

        # Predict and display result
        if st.button("Detect Disease"):
            with st.spinner("Processing..."):
                predicted_label = predict_audio(temp_file_path, model, encoder)

                # Displaying the result with custom styling
                st.markdown(
                    f"""
                    <div style="text-align: center; margin-top: 20px;">
                        <h1 style="font-size: 36px; font-weight: bold;">
                            Predicted Class: <span style="color: #D6336C;">{predicted_label}</span>
                        </h1>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if os.path.exists(temp_file_path):
                # Delete the temporary audio file after prediction
                os.remove(temp_file_path)
                print(f"File {temp_file_path} deleted successfully.")


# Main function
def main():
    page_style = """
    <style>
        body {
            font-family: "Arial", sans-serif;
        }
        .container {
            max-width: 95%;
            margin: auto;
            padding: 5%;
        }
        .header {
            text-align: center;
            font-size: 24px;
            color: #333;
        }
        .content {
            padding: 10%;
            background-color: #f5f5f5;
            border-radius: 10%;
        }
    </style>
    """
    st.markdown(page_style, unsafe_allow_html=True)
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    st.header("Chicken Disease Detection by audio")
    upload_file()
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
