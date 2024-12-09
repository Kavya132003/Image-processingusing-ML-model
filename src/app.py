from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
from model import create_model

app = Flask(__name__)

# Define the correct path to your model file
MODEL_PATH = r"C:\Users\Aishwarya G\OneDrive\Desktop\AICTE ML Model\best_model.keras"  # Use your actual path

# Load the trained model with error handling

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise