import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Attention
# Assuming BahdanauAttention is in 'model.py' (or import from .ipynb notebook if needed)
from model import BahdanauAttention  

# --- Define Custom GetItem Layer ---
class GetItem(Layer):
    """
    Custom layer to handle 'GetItem' operations in the saved model.
    This is a placeholder implementation - you should replace with the original
    implementation if you have it.
    """
    def __init__(self, **kwargs):
        super(GetItem, self).__init__(**kwargs)

    def call(self, inputs):
        # Default implementation - returns first element
        # Replace this with the actual logic if known
        return inputs[0] if isinstance(inputs, (list, tuple)) else inputs

    def get_config(self):
        return super(GetItem, self).get_config()

# --- 1. Load the Saved Model with Custom Objects ---
def load_custom_model(model_path):
    """
    Load model with custom layer support.
    Args:
        model_path (str): Path to the saved model file.
    Returns:
        Loaded Keras model.
    """
    try:
        # Register all custom objects here
        with custom_object_scope({'GetItem': GetItem, 'BahdanauAttention': BahdanauAttention}):
            model = load_model(model_path)
            print("Model loaded successfully with custom layers!")
            return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

model_path = 'image_captioning_model.h5'  # Replace with your model path
model = load_custom_model(model_path)

# --- 2. Load and Preprocess Test Image ---
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess image for model input.
    Args:
        image_path (str): Path to the image.
        target_size (tuple): Resize dimensions (default: (224, 224)).
    Returns:
        np.array: Preprocessed image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    img = cv2.resize(img, target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Normalize for model (e.g., ResNet preprocessing)
    return img

# --- 3. Run Prediction ---
def predict(image_path):
    """
    Run model prediction on an image.
    Args:
        image_path (str): Path to the test image.
    Returns:
        Predicted output (e.g., class probabilities or captions).
    """
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return prediction

# --- 4. Test on a Sample Image ---
if __name__ == "__main__":
    test_image_path = r"Image Captioning System\Image-Captioning-System\data\bike.jpg"
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Test image not found: {test_image_path}")
    
    print(f"Testing on: {test_image_path}")
    try:
        prediction = predict(test_image_path)
        print("Prediction Output:", prediction)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
