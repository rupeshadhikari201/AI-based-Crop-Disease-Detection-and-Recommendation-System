import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf 
import pickle
import json

st.title("AI-based Crop Disease Detection and Recommendation System")

uploaded_image = st.file_uploader(label="Upload an Image")

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# load the trained model
try:
    model = tf.keras.models.load_model('plant_disease_prediction_model.h5')
except Exception as e:
    st.error(f"Error Loading Model.", e)

# Load class indices
try:
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
except FileNotFoundError:
    st.error("class_indices.json file not found")


recommendations = {
    'Apple___Apple_scab': 'Apply a fungicide that contains captan, copper, or sulfur. Remove infected leaves and fruit.',
    'Apple___Black_rot': 'Prune and destroy affected branches. Use a copper-based fungicide spray.',
    'Apple___Cedar_apple_rust': 'Remove nearby cedar trees if possible. Apply a fungicide during spring.',
    'Apple___healthy': 'Keep monitoring the crop and ensure optimal conditions for growth.',
    'Blueberry___healthy': 'Maintain good irrigation and nutrient levels.',
    'Cherry_(including_sour)___Powdery_mildew': 'Prune affected parts and apply a sulfur-based fungicide. Ensure proper air circulation around the plants.',
    'Cherry_(including_sour)___healthy': 'Regularly inspect for any early signs of disease and maintain optimal conditions.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Apply fungicides with active ingredients like azoxystrobin or pyraclostrobin. Ensure crop rotation.',
    'Corn_(maize)___Common_rust_': 'Plant resistant varieties and apply fungicides containing mancozeb or chlorothalonil.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Use resistant seed varieties and apply a triazole or strobilurin fungicide.',
    'Corn_(maize)___healthy': 'Maintain good agricultural practices, including adequate spacing and nutrient management.',
    'Grape___Black_rot': 'Remove and destroy infected leaves and fruit. Apply fungicides like mancozeb or myclobutanil.',
    'Grape___Esca_(Black_Measles)': 'Prune affected vines and use appropriate fungicides. Consider improving drainage and limiting stress on the plants.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Prune infected areas and ensure proper airflow. Apply copper-based fungicides as needed.',
    'Grape___healthy': 'Ensure proper care, including balanced nutrition and regular monitoring.',
    'Orange___Haunglongbing_(Citrus_greening)': 'There is no cure; remove affected trees to prevent spreading. Implement vector control for psyllids.',
    'Peach___Bacterial_spot': 'Remove infected leaves and fruit. Apply copper-based bactericides and avoid overhead irrigation.',
    'Peach___healthy': 'Continue monitoring and maintain proper nutrition and watering practices.',
    'Pepper,_bell___Bacterial_spot': 'Remove and destroy infected leaves. Use copper-based bactericides and avoid overhead watering.',
    'Pepper,_bell___healthy': 'Maintain proper spacing and ensure good air circulation around plants.',
    'Potato___Early_blight': 'Apply fungicides like chlorothalonil or mancozeb. Practice crop rotation and use resistant varieties.',
    'Potato___Late_blight': 'Remove affected plants immediately and apply fungicides containing chlorothalonil or mancozeb.',
    'Potato___healthy': 'Keep monitoring and ensure proper soil management and nutrient levels.',
    'Raspberry___healthy': 'Ensure good practices such as regular weeding, proper spacing, and balanced fertilization.',
    'Soybean___healthy': 'Maintain proper soil health and use crop rotation practices to avoid diseases.',
    'Squash___Powdery_mildew': 'Apply sulfur-based fungicides and ensure good air circulation by pruning overcrowded areas.',
    'Strawberry___Leaf_scorch': 'Remove and destroy affected leaves. Apply a copper-based fungicide if necessary.',
    'Strawberry___healthy': 'Maintain proper irrigation practices and inspect regularly for early disease signs.',
    'Tomato___Bacterial_spot': 'Use copper-based sprays and avoid working in the garden when plants are wet to prevent spreading.',
    'Tomato___Early_blight': 'Use a fungicide with mancozeb or chlorothalonil. Remove and destroy affected foliage.',
    'Tomato___Late_blight': 'Remove and destroy affected plants to stop spreading. Apply chlorothalonil-based fungicides.',
    'Tomato___Leaf_Mold': 'Improve airflow around plants and apply fungicides with chlorothalonil or copper.',
    'Tomato___Septoria_leaf_spot': 'Remove affected leaves and use fungicides containing chlorothalonil or copper.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Spray with insecticidal soap or neem oil. Ensure plants are well-hydrated.',
    'Tomato___Target_Spot': 'Apply a fungicide containing azoxystrobin and remove affected leaves.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whitefly populations as they spread the virus. Remove infected plants promptly.',
    'Tomato___Tomato_mosaic_virus': 'Remove infected plants and disinfect tools. Plant virus-resistant varieties.',
    'Tomato___healthy': 'Ensure proper plant spacing and optimal watering practices to prevent diseases.'
}


# Predict the class of the uploaded image
if uploaded_image:
    image_path = uploaded_image
    preprocessed_image = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Get the class name
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
    recommended = recommendations.get(predicted_class_name, "No Recommendation Available!!")
    
    st.write("The Predicted Class is:", predicted_class_name)
    st.title("Recommendation")
    st.info(recommended)
else:
    st.text("Please upload an image.")
