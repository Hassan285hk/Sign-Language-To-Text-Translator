import cv2
import numpy as np
import tensorflow as tf
import pickle
import os

# --- CONFIGURATION ---
IMG_SIZE = 64
MODEL_PATH = '../models/sign_language_translator_best.keras'
# Path to the new Label Binarizer file (created by the updated train.py)
LB_PATH = '../models/label_binarizer.pkl' 

# --- 1. LOAD MODEL AND LABEL BINARIZER ---

# Load the label binarizer (lb) 
try:
    with open(LB_PATH, 'rb') as f:
        # Load only the Label Binarizer object
        lb = pickle.load(f) 
    print("Label Binarizer loaded successfully.")
except Exception as e:
    print(f"Error loading Label Binarizer from {LB_PATH}: {e}")
    lb = None 

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model from path: {e}")
    model = None

# --- 2. PREPROCESSING FUNCTION ---

def preprocess_image(frame, x, y, w, h):
    """
    Extracts the hand ROI and applies Canny Edge Detection, 
    matching the preprocessing used in training.
    """
    # 1. Extract Hand ROI (Region of Interest)
    hand_roi = frame[y:y+h, x:x+w]
    if hand_roi.size == 0 or hand_roi.shape[0] == 0 or hand_roi.shape[1] == 0:
        return None

    # 2. Convert to Grayscale
    img_gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)

    # 3. Apply Canny Edge Detection (Matching preprocess.py)
    # The parameters (5,5) for Gaussian blur and (50, 150) for Canny thresholds are critical.
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    canny_output = cv2.Canny(blurred, 50, 150)
    
    # Optional: Display the Canny output for visual confirmation
    cv2.imshow("Canny Output", cv2.resize(canny_output, (200, 200))) 

    # 4. Resize and Reshape for Model Input
    resized_canny = cv2.resize(canny_output, (IMG_SIZE, IMG_SIZE))
    
    # Normalize to 0-1 and add batch/channel dimensions (1, 64, 64, 1)
    model_input = resized_canny / 255.0
    model_input = np.expand_dims(model_input, axis=(0, -1))
    
    return model_input

# --- 3. MAIN REAL-TIME LOOP ---

def run_realtime():
    if model is None:
        print("Cannot run real-time inference because the model failed to load.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define the dimensions for the Hand Region Box (Green Box)
    BOX_SIZE = 250
    x_offset, y_offset = 100, 100
    x1, y1 = x_offset, y_offset
    x2, y2 = x_offset + BOX_SIZE, y_offset + BOX_SIZE

    print("Starting Real-Time Translator. Place your hand in the green box.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for natural selfie view
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        prediction_label = "Waiting..."
        prediction_conf = 0.0

        model_input = preprocess_image(frame, x1, y1, BOX_SIZE, BOX_SIZE)

        if model_input is not None:
            # 1. Get predictions 
            predictions = model.predict(model_input, verbose=0)[0]
            predicted_index = np.argmax(predictions)
            prediction_conf = predictions[predicted_index]
            
            # 2. Map index to letter using the loaded Label Binarizer
            if lb is not None:
                prediction_label = lb.classes_[predicted_index]
            else:
                prediction_label = f"INDEX {predicted_index}" 

        # Display the result on the main frame
        text = f"Sign: {prediction_label} ({prediction_conf * 100:.2f}%)"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Sign Language Translator', frame)

        # Exit loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_realtime()