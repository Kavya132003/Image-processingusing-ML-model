import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_dir, img_size=(224, 224)):
    images = []
    labels = []
    
    # Get all class folders
    class_folders = os.listdir(data_dir)
    
    for class_idx, class_folder in enumerate(class_folders):
        folder_path = os.path.join(data_dir, class_folder)
        if not os.path.isdir(folder_path):
            continue
            
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_file)
            try:
                # Read and preprocess image
                img = cv2.imread(image_path)
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalize pixel values
                
                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")
    
    return np.array(images), np.array(labels)

def prepare_data(data_dir, test_size=0.2):
    # Load and preprocess images
    X, y = load_and_preprocess_data(data_dir)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test 