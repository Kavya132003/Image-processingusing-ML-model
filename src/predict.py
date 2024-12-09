import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

def get_class_names(data_dir):
    return sorted(os.listdir(data_dir))

def predict_image(image_path, model_path='best_model.keras', data_dir=r'C:\Users\Aishwarya G\OneDrive\Desktop\AICTE ML Model\data'):
    # Load the trained model
    model = load_model(model_path)
    
    # Get class names
    class_names = get_class_names(data_dir)
    
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at path: {image_path}")
        
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img)
    
    # Get probabilities for all classes
    class_probabilities = list(zip(class_names, prediction[0]))
    class_probabilities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top prediction
    predicted_class_name = class_probabilities[0][0]
    confidence = class_probabilities[0][1]
    
    return predicted_class_name, confidence, class_probabilities

if __name__ == "__main__":
    IMAGE_PATH = r"C:\Users\Aishwarya G\OneDrive\Desktop\AICTE ML Model\data\cats\cat 1.jpeg"
    
    try:
        class_name, confidence, probabilities = predict_image(IMAGE_PATH)
        print(f"\nImage: {IMAGE_PATH}")
        print(f"Predicted class: {class_name}")
        print(f"Confidence: {confidence:.2f}")
        
        print("\nAll class probabilities:")
        for class_name, prob in probabilities:
            print(f"{class_name}: {prob:.2f}")
    except Exception as e:
        print(f"Error: {str(e)}") 