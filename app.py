import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Cache model loading to optimize performance
@st.cache_resource
def load_model(model_path):
    """Loads the YOLO model from a specified path."""
    return YOLO(model_path)

# Define paths to the models
models = {
    "Normal Weather": "normal/best.pt",
    "Adverse Weather": "adv/best.pt"
}

# Sidebar for model selection
st.sidebar.title("Vehicle Detection Model")
model_choice = st.sidebar.selectbox("Select the model for detection:", list(models.keys()))
model = load_model(models[model_choice])  # Load the selected model

# Main interface
st.title("Vehicle Detection Interface")
uploaded_image = st.file_uploader("Upload an image for vehicle detection", type=["jpg", "jpeg", "png"])

# Process and display results if an image is uploaded
if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to a compatible format and perform detection
    image_np = np.array(image)
    results = model.predict(source=image_np)
    
    # Display detection results
    st.subheader("Detection Results")
    annotated_img = results[0].plot()  # Generate an annotated image with detected vehicles
    st.image(annotated_img, caption="Detected Vehicles", use_column_width=True)

    # Show the count of detected vehicles
    num_vehicles = len(results[0].boxes)
    st.write(f"**Number of detections:** {num_vehicles}")
