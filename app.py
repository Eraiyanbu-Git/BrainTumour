import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Set page config
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# Load the trained MobileNetV2 model
@st.cache_resource
def load_trained_model():
    return load_model("mobilenetv2_model.h5")

model = load_trained_model()

# Class labels (update if you use different ones)
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Header
st.title("🧠 Brain Tumor Classification")
st.write("Upload a brain MRI scan to classify the tumor type using a pre-trained MobileNetV2 model.")

# Upload image
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    # Display result
    st.markdown(f"### 🧾 Prediction: `{predicted_class.upper()}`")
    st.markdown(f"### 📊 Confidence: `{confidence:.2f}%`")

    # Show probability distribution
    st.markdown("### 🔍 Class Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"- **{class_names[i]}**: {prob*100:.2f}%")
