import cv2
import numpy as np
import tensorflow as tf
import pickle
from collections import deque, Counter

# ===================== CONFIGURATION =====================
IMG_SIZE = 64

MODEL_PATH = '../models/sign_language_translator_best.keras'
LABEL_BINARIZER_PATH = '../models/label_binarizer.pkl'

# Bigger ROI improves accuracy
BOX_SIZE = 350
X_START, Y_START = 100, 100
X_END, Y_END = X_START + BOX_SIZE, Y_START + BOX_SIZE

# Prediction smoothing (IMPORTANT)
PREDICTION_HISTORY_LENGTH = 25
prediction_history = deque(maxlen=PREDICTION_HISTORY_LENGTH)

DEFAULT_INPUT = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.60

# ===================== LOAD MODEL =====================
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully")

    with open(LABEL_BINARIZER_PATH, 'rb') as f:
        lb = pickle.load(f)
    print("[INFO] Label binarizer loaded successfully")

except Exception as e:
    print("[ERROR] Model loading failed:", e)
    exit()

# ===================== DIP PREPROCESSING =====================
def process_and_get_model_input(frame):

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[Y_START:Y_END, X_START:X_END]

    # Gaussian blur (noise removal)
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    # Adaptive threshold (BEST FOR HAND SEGMENTATION)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    model_input = DEFAULT_INPUT.copy()
    visual_crop = thresh.copy()

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) > 2000:
            x, y, w, h = cv2.boundingRect(largest_contour)

            buffer = 10
            x1 = max(0, x - buffer)
            y1 = max(0, y - buffer)
            x2 = min(BOX_SIZE, x + w + buffer)
            y2 = min(BOX_SIZE, y + h + buffer)

            if x2 > x1 and y2 > y1:
                hand_crop = thresh[y1:y2, x1:x2]
                visual_crop = hand_crop

                resized = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
                resized = resized / 255.0

                model_input = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return model_input, visual_crop

# ===================== MAIN FUNCTION =====================
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Webcam not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Draw ROI
        cv2.rectangle(
            frame, (X_START, Y_START), (X_END, Y_END), (0, 255, 0), 2
        )

        model_input, visual_crop = process_and_get_model_input(frame)

        prediction_label = "No Hand Detected"
        confidence = 0.0

        if not np.array_equal(model_input, DEFAULT_INPUT):
            preds = model.predict(model_input, verbose=0)[0]

            idx = np.argmax(preds)
            confidence = preds[idx]
            predicted_char = lb.classes_[idx]

            if confidence > CONFIDENCE_THRESHOLD:
                prediction_history.append(predicted_char)

                most_common = Counter(prediction_history).most_common(1)[0]
                stability = most_common[1] / PREDICTION_HISTORY_LENGTH

                prediction_label = f"Sign: {most_common[0]} | Stability: {stability:.2f}"

        else:
            prediction_history.clear()

        # Color coding
        color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 0, 255)

        cv2.putText(
            frame,
            prediction_label,
            (X_START, Y_START - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.imshow("Sign Language Translator", frame)

        visual_display = cv2.resize(visual_crop, (220, 220))
        cv2.imshow("Processed Hand (Binary)", visual_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===================== RUN =====================
if __name__ == "__main__":
    main()
