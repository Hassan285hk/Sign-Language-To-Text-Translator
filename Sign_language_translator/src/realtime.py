import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

# --- CONFIGURATION ---
MODEL_PATH = '../models/sign_language_translator_best.keras'
LB_PATH = '../models/label_binarizer.pkl'
IMG_SIZE = 64
CAM_WIDTH, CAM_HEIGHT = 640, 480

# Define the Initial Region of Interest (ROI) box coordinates
# This is the general area where the user should place their hand
BOX_SIZE = 200
X_START, Y_START = 100, 100
X_END, Y_END = X_START + BOX_SIZE, Y_START + BOX_SIZE

# Fallback value for when no hand is clearly detected
DEFAULT_INPUT = np.zeros((1, IMG_SIZE, IMG_SIZE, 1))

def load_resources():
    """Loads the trained model and label binarizer."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LB_PATH):
        raise FileNotFoundError("Model or Label Binarizer not found. Run train.py first.")
        
    model = load_model(MODEL_PATH)
    with open(LB_PATH, 'rb') as f:
        lb = pickle.load(f)
    return model, lb

def process_and_get_model_input(frame_rgb):
    """
    Applies DIP steps: Grayscale -> Thresholding -> Contour Detection -> Bounding Box Crop.
    Returns the processed input for the model AND the visual black/white ROI.
    """
    
    # 1. Initial Grayscale Conversion and ROI Extraction
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    roi = gray[Y_START:Y_END, X_START:X_END]

    # Apply Gaussian Blur (DIP denoising)
    blurred_roi = cv2.GaussianBlur(roi, (7, 7), 0)
    
    # Otsu's Thresholding (DIP segmentation for high-contrast B/W image)
    _, thresh_roi = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Contour Detection (Feature Extraction)
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    model_input = DEFAULT_INPUT.copy()
    visual_crop = thresh_roi.copy() # Default visual is the whole thresholded ROI

    if contours:
        # Find the largest contour (assumed to be the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if the largest contour is large enough to be a hand
        if cv2.contourArea(largest_contour) > 1000: # Threshold for minimum hand size
            
            # Get the tight bounding rectangle (x, y, w, h) around the hand
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            buffer = 5
            
            # --- Boundary Check Fix: Ensure coordinates stay within the ROI (0 to BOX_SIZE) ---
            y_start_safe = max(0, y - buffer)
            y_end_safe = min(BOX_SIZE, y + h + buffer)
            x_start_safe = max(0, x - buffer)
            x_end_safe = min(BOX_SIZE, x + w + buffer)

            # Ensure the crop slice is valid
            if y_end_safe > y_start_safe and x_end_safe > x_start_safe:
                
                # Crop the thresholded image precisely to the hand boundary
                hand_crop = thresh_roi[y_start_safe:y_end_safe, x_start_safe:x_end_safe]
                
                # Update the visual crop and draw the boundary line on the thresholded image
                visual_crop = hand_crop
                cv2.rectangle(thresh_roi, (x, y), (x + w, y + h), (255, 255, 255), 2)
                
                # Prepare the cropped hand image for the model
                resized_input = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
                model_input = resized_input / 255.0 # Normalization
                
                # Reshape (Add batch and channel dimensions)
                model_input = np.expand_dims(model_input, axis=0)
                model_input = np.expand_dims(model_input, axis=-1)
            
    return model_input, visual_crop

def realtime_translator():
    # Load model resources
    model, lb = load_resources()
    
    # Initialize video capture (0 is typically the default webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Check if another app is using it.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    print("Starting Real-Time Translator. Place hand in green box. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) # Flip frame horizontally for mirror view

        # Draw the main ROI box (general guide for the user)
        cv2.rectangle(frame, (X_START, Y_START), (X_END, Y_END), (0, 255, 0), 2)
        
        # Preprocess the frame, get model input and the processed visual
        input_data, visual_crop = process_and_get_model_input(frame)

        # Run Prediction (only if the input is not the default black array)
        if not np.array_equal(input_data, DEFAULT_INPUT):
            # Predict only if a valid hand crop was generated
            prediction = model.predict(input_data, verbose=0)
            predicted_index = np.argmax(prediction)
            predicted_label = lb.classes_[predicted_index]
            confidence = prediction[0][predicted_index] * 100
        else:
            # If hand not detected clearly
            predicted_label = "WAITING..."
            confidence = 0.0

        # Display Result on the main frame
        text = f"Sign: {predicted_label} ({confidence:.2f}%)"
        cv2.putText(frame, text, (X_START, Y_START - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the Processed DIP Image (The isolated hand with contour)
        # Resize the visual crop to maintain a consistent window size for display
        display_visual = cv2.resize(visual_crop, (BOX_SIZE, BOX_SIZE), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Processed Hand Sign (Model Input)', display_visual)
        
        # Display the resulting webcam frame
        cv2.imshow('Webcam Feed (Green Box Guide)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        realtime_translator()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure that '../models/sign_language_translator_best.keras' and '../models/label_binarizer.pkl' exist.")