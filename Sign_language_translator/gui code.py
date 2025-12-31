import cv2
import numpy as np
import tensorflow as tf
import pickle
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from collections import deque, Counter

# ================= CONFIG =================
IMG_SIZE = 64
MODEL_PATH = '../models/sign_language_translator_best.keras'
LABEL_BINARIZER_PATH = '../models/label_binarizer.pkl'

BOX_SIZE = 350
X_START, Y_START = 100, 100
X_END, Y_END = X_START + BOX_SIZE, Y_START + BOX_SIZE

CONFIDENCE_THRESHOLD = 0.6
PREDICTION_HISTORY_LENGTH = 25

prediction_history = deque(maxlen=PREDICTION_HISTORY_LENGTH)
DEFAULT_INPUT = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_BINARIZER_PATH, 'rb') as f:
    lb = pickle.load(f)

# ================= DIP FUNCTION =================
def process_and_get_model_input(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[Y_START:Y_END, X_START:X_END]

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, 2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, 2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    model_input = DEFAULT_INPUT.copy()
    visual_crop = thresh.copy()

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 2000:
            x, y, w, h = cv2.boundingRect(largest)
            buffer = 10

            x1 = max(0, x - buffer)
            y1 = max(0, y - buffer)
            x2 = min(BOX_SIZE, x + w + buffer)
            y2 = min(BOX_SIZE, y + h + buffer)

            hand = thresh[y1:y2, x1:x2]
            visual_crop = hand

            resized = cv2.resize(hand, (IMG_SIZE, IMG_SIZE))
            resized = resized / 255.0
            model_input = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return model_input, visual_crop

# ================= GUI CLASS =================
class SignLanguageGUI:

    def __init__(self, window):
        self.window = window
        self.window.title("Sign Language to Text Translator")
        self.window.geometry("900x500")
        self.window.configure(bg="#1e1e1e")

        self.video_label = Label(window)
        self.video_label.place(x=20, y=20)

        self.processed_label = Label(window)
        self.processed_label.place(x=650, y=50)

        self.text_label = Label(
            window,
            text="Detected Sign: ",
            font=("Arial", 18, "bold"),
            fg="lime",
            bg="#1e1e1e"
        )
        self.text_label.place(x=300, y=430)

        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        cv2.rectangle(
            frame,
            (X_START, Y_START),
            (X_END, Y_END),
            (0, 255, 0),
            2
        )

        model_input, visual_crop = process_and_get_model_input(frame)

        prediction_text = "No Hand"

        if not np.array_equal(model_input, DEFAULT_INPUT):
            preds = model.predict(model_input, verbose=0)[0]
            idx = np.argmax(preds)
            confidence = preds[idx]

            if confidence > CONFIDENCE_THRESHOLD:
                char = lb.classes_[idx]
                prediction_history.append(char)

                most_common = Counter(prediction_history).most_common(1)[0]
                prediction_text = most_common[0]

        else:
            prediction_history.clear()

        self.text_label.config(text=f"Detected Sign: {prediction_text}")

        # Convert frames for GUI
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_pil = frame_pil.resize((600, 400))
        frame_tk = ImageTk.PhotoImage(frame_pil)

        self.video_label.imgtk = frame_tk
        self.video_label.configure(image=frame_tk)

        processed = cv2.resize(visual_crop, (200, 200))
        processed_pil = Image.fromarray(processed)
        processed_tk = ImageTk.PhotoImage(processed_pil)

        self.processed_label.imgtk = processed_tk
        self.processed_label.configure(image=processed_tk)

        self.window.after(10, self.update_frame)

    def close(self):
        self.cap.release()
        self.window.destroy()

# ================= RUN =================
root = tk.Tk()
app = SignLanguageGUI(root)
root.protocol("WM_DELETE_WINDOW", app.close)
root.mainloop()
