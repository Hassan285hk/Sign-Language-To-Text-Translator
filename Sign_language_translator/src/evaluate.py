import pickle
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
MODEL_PATH = '../models/sign_language_translator_best.keras'
LB_PATH = '../models/label_binarizer.pkl'
TEST_DATA_DIR = '../data/test'
IMG_SIZE = 64

def load_test_data(data_dir):
    """
    Loads and preprocesses test images from the specified directory.
    NOTE: This uses the EXACT SAME logic as the training preprocessing.
    """
    data = []
    labels = []
    
    categories = os.listdir(data_dir)
    
    for category in categories:
        path = os.path.join(data_dir, category)
        if not os.path.isdir(path): continue

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                # Load as grayscale and resize (MUST match training data processing)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_array is None: continue
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                
                data.append(resized_array)
                labels.append(category)
            except Exception as e:
                print(f"Error loading test image {img_name}: {e}")

    # Normalization and Reshape
    X = np.array(data) / 255.0
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    return X, labels, categories

def evaluate_model():
    # 1. Load Resources
    print("Loading best model and label binarizer...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LB_PATH):
        print("Error: Model or Label Binarizer files not found.")
        return

    model = load_model(MODEL_PATH)
    with open(LB_PATH, 'rb') as f:
        lb = pickle.load(f)
        
    # 2. Load and Prepare Test Data
    print("Loading and preparing test data...")
    X_test, true_labels, categories = load_test_data(TEST_DATA_DIR)
    
    if len(X_test) == 0:
        print("Test data not found or failed to load. Please check data/test.")
        return

    # 3. Predict
    print(f"Running prediction on {len(X_test)} test images...")
    predictions = model.predict(X_test, verbose=0)
    
    # 4. Decode Predictions
    # Convert one-hot prediction arrays back into their letter labels
    predicted_classes_encoded = np.argmax(predictions, axis=1)
    predicted_labels = lb.classes_[predicted_classes_encoded]

    # 5. Generate Metrics
    print("\n" + "="*50)
    print("      FINAL CLASSIFICATION REPORT")
    print("="*50)
    
    # Print a detailed report including precision, recall, and F1-score
    print(classification_report(true_labels, predicted_labels, zero_division=0))
    
    # Optional: Display a confusion matrix (if you have the plotting libraries)
    # This visually shows which signs are being confused with others.
    cm = confusion_matrix(true_labels, predicted_labels, labels=lb.classes_)
    
    print("\n" + "="*50)
    print("      CONFUSION MATRIX (Shows misclassified signs)")
    print("="*50)
    print("NOTE: Requires visualization tools to display the matrix.")
    
    # You can plot the confusion matrix for better visualization:
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=lb.classes_, yticklabels=lb.classes_)
    # plt.title('Confusion Matrix')
    # plt.ylabel('True Label')
    # plt.xlabel('Predicted Label')
    # plt.show()


if __name__ == "__main__":
    evaluate_model()