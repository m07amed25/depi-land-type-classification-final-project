import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json

# Set page config
st.set_page_config(
    page_title="EuroSAT Image Classifier",
    page_icon="🛰️",
    layout="centered"
)

# Load model and class indices
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('eurosat_model.keras')
    return model

@st.cache_data
def load_class_indices():
    with open('class_indices.json') as f:
        return json.load(f)

# Class descriptions
class_descriptions = {
    'AnnualCrop': 'Agricultural fields with annual crops that are replanted each year',
    'Forest': 'Dense tree-covered areas, typically natural forests',
    'HerbaceousVegetation': 'Areas dominated by herbaceous plants, grasses, and non-woody vegetation',
    'Highway': 'Major roads with visible lanes and often median separations',
    'Industrial': 'Areas with factories, warehouses, and industrial infrastructure',
    'Pasture': 'Grasslands used for grazing livestock',
    'PermanentCrop': 'Long-term crops like orchards, vineyards, and plantations',
    'Residential': 'Urban or suburban areas with houses and residential buildings',
    'River': 'Natural flowing watercourses',
    'SeaLake': 'Large bodies of water including seas, lakes, and reservoirs'
}

# Preprocess image
def preprocess_image(img, target_size=(128, 128)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def main():
    st.title("🛰️ EuroSAT Satellite Image Classification")
    st.write("Upload a satellite image for land use classification")

    # Load model and class indices
    try:
        model = load_model()
        class_indices = load_class_indices()
        index_to_class = {v: k for k, v in class_indices.items()}
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a satellite image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            # Display image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", width=300)
            
            # Preprocess and predict
            img_array = preprocess_image(img)
            predictions = model.predict(img_array)
            predicted_class = index_to_class[np.argmax(predictions)]
            confidence = np.max(predictions) * 100
            
            # Display results
            st.subheader("Prediction Results")
            st.success(f"**Class:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.write(f"**Description:** {class_descriptions[predicted_class]}")
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    # Model info
    st.markdown("---")
    st.subheader("About the Model")
    st.write("""
    - **Model Type**: Convolutional Neural Network (CNN)
    - **Training Data**: EuroSAT dataset
    - **Classes**: 10 land use/land cover types
    - **Input Size**: 128x128 pixels
    """)

if __name__ == "__main__":
    main()