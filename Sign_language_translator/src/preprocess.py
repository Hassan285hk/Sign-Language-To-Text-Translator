import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer

# --- CONFIGURATION ---
# We define the size here. 64x64 is usually enough for hand signs and fast training.
IMG_SIZE = 64
DATA_DIR = '../data/Train'  # Path to your training data
PROCESSED_DATA_PATH = '../data/processed_data.pickle' # Where to save the result

def get_data(data_dir):
    """
    Loads images from the dataset directory, resizes them,
    and converts them to numpy arrays.
    """
    data = []
    labels = []
    
    # Check if the directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return np.array([]), np.array([])

    # Get the list of classes (folders named 'A', 'B', etc.)
    categories = os.listdir(data_dir)
    print(f"Found {len(categories)} classes: {categories}")

    for category in categories:
        path = os.path.join(data_dir, category)
        
        # Skip if it's not a folder
        if not os.path.isdir(path):
            continue

        print(f"Processing Class: {category}")
        
        for img_name in os.listdir(path):
            try:
                # 1. Read the image
                # We force convert to GRAYSCALE because your images are black & white
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img_array is None:
                    continue

                # 2. Resize
                # CNNs need all input images to be the exact same size
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                # 3. Add to our list
                data.append(resized_array)
                labels.append(category)
                
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")

    # Convert to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # 4. Normalization (DIP Technique)
    # Pixel values are 0-255. We scale them to 0-1 for the Neural Network.
    data = data / 255.0
    
    # 5. Reshape for CNN
    # The model expects shape: (Number_of_Images, 64, 64, 1)
    # The '1' stands for the grayscale channel.
    data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    return data, labels

def save_data(data, labels, filename):
    """
    Saves the processed data to a pickle file so we don't have to 
    process images every time we want to train.
    """
    # Convert text labels (A, B) to binary arrays (0, 1)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    
    with open(filename, 'wb') as f:
        pickle.dump((data, labels, lb), f)
    
    print(f"Data saved successfully to {filename}")

if __name__ == "__main__":
    # This block runs only if you run this file directly
    print("Starting data preprocessing...")
    
    X, y = get_data(DATA_DIR)
    
    if len(X) > 0:
        print(f"Successfully processed {len(X)} images.")
        print(f"Data Shape: {X.shape}")
        save_data(X, y, PROCESSED_DATA_PATH)
    else:
        print("No images found. Please check your DATA_DIR path.")