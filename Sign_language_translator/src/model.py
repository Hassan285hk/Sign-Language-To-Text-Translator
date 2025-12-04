import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the input size based on our preprocessing step
IMG_SIZE = 64

def create_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=26):
    """
    Defines and compiles a Convolutional Neural Network (CNN) model.

    Args:
        input_shape (tuple): The shape of the input image (64, 64, 1).
        num_classes (int): The number of output classes (e.g., 26 for A-Z).

    Returns:
        tf.keras.Model: The compiled CNN model.
    """
    model = Sequential([
        # 1. Feature Extraction Layers (Convolutional Base)
        
        # First Layer: Captures basic features like edges and lines
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Second Layer: Captures more complex patterns
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Third Layer: Captures even richer, abstract features
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Adding Dropout to prevent overfitting, which is common in small datasets
        Dropout(0.25),

        # 2. Classification Layers (Dense Layers)

        # Flatten the 3D output into a 1D vector for the Dense layers
        Flatten(),
        
        # Hidden Dense Layer for learning non-linear combinations
        Dense(512, activation='relu'),
        
        # Final Dropout layer
        Dropout(0.5),

        # Output Layer: One neuron for each class.
        # 'softmax' ensures the outputs are probabilities that sum to 1.
        Dense(num_classes, activation='softmax') 
    ])

    # Compile the model
    # Optimizer: 'adam' is a good default choice for fast learning.
    # Loss: 'categorical_crossentropy' is used for multi-class classification 
    #       when labels are one-hot encoded (which we did in preprocess.py).
    # Metrics: We track accuracy during training.
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    # Test the model creation and print its summary
    print("--- Model Summary ---")
    sign_model = create_model()
    sign_model.summary()
    print("---------------------")