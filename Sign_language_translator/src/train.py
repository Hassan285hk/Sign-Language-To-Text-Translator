import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import create_model  # Import the function from your model.py

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = '../data/processed_data.pickle'
MODEL_SAVE_PATH = '../models/sign_language_translator_best.keras'
BATCH_SIZE = 32
EPOCHS = 50 
VALIDATION_SPLIT = 0.15 # Use 15% of data for validation

def load_data(filepath):
    """Loads the processed image data and labels from the pickle file."""
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}")
        return None, None, None
        
    with open(filepath, 'rb') as f:
        # X: image data, y: one-hot encoded labels, lb: LabelBinarizer
        X, y, lb = pickle.load(f)
    return X, y, lb

def train_model():
    # 1. Load Data
    print("Loading processed data...")
    X, y, lb = load_data(PROCESSED_DATA_PATH)
    
    if X is None:
        return

    # 2. Split Data
    # Separate data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42
    )

    # Determine the number of classes from the one-hot encoded label shape
    num_classes = y.shape[1]
    input_shape = X.shape[1:]

    # 3. Create Model
    print(f"Creating model with {num_classes} classes and input shape {input_shape}...")
    model = create_model(input_shape=input_shape, num_classes=num_classes)
    
    # Optional: Print summary to confirm class count
    # model.summary()

    # 4. Define Callbacks
    # ModelCheckpoint: Saves the model whenever validation accuracy improves
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH, 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )
    # EarlyStopping: Stops training if validation accuracy doesn't improve for 5 epochs
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        mode='min', 
        verbose=1
    )
    callbacks_list = [checkpoint, early_stopping]
    
    # 5. Train Model
    print("Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list
    )

    print("\nTraining complete!")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")
    
    # Save the LabelBinarizer object (crucial for translating predictions back to letters)
    with open('../models/label_binarizer.pkl', 'wb') as f:
        pickle.dump(lb, f)

if __name__ == "__main__":
    train_model()