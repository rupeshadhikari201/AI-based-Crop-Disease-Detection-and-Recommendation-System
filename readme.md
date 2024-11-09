# Plant Disease Prediction using Deep Learning
This Colab notebook demonstrates how to build a deep learning model to predict plant diseases using the PlantVillage dataset.

## Dataset

The PlantVillage dataset is used for this project. It contains images of healthy and diseased plants. The dataset is downloaded from Kaggle using the Kaggle API.

## Model

A Convolutional Neural Network (CNN) is used for image classification. The model consists of convolutional layers, max pooling layers, a flatten layer, and dense layers.

## Usage

1. **Install Kaggle:**
2. 2. **Upload Kaggle Credentials:**
   - Create a `kaggle.json` file with your Kaggle username and API key.
   - Upload the `kaggle.json` file to the Colab environment.

3. **Download Dataset:**
4. **Unzip Dataset:**
5. **Data Preprocessing:**
   - Images are resized and normalized.
   - Data is split into training and validation sets.

6. **Model Training:**
   - The model is compiled with the Adam optimizer and categorical cross-entropy loss function.
   - The model is trained on the training data.

7. **Model Evaluation:**
   - The model is evaluated on the validation data.
   - Accuracy and loss metrics are reported.

8. **Prediction:**
   - A function `predict_image_class` is provided to predict the class of a new image.


## Requirements

- Python 3
- TensorFlow
- Keras
- Matplotlib
- Pillow
- Kaggle API


## Note

- Make sure to have a Kaggle account and API key to download the dataset.
- The model is saved as `plant_disease_prediction_model.h5`.
- Class indices are saved as `class_indices.json`.
