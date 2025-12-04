🤟  *Sign Language (ASL) to Text Translator*

🌟 **Project Overview**

This repository hosts a real-time sign language recognition system built using Convolutional Neural Networks (CNNs) and OpenCV for Digital Image Processing (DIP). The system translates static American Sign Language (ASL) hand gestures (A-Z, plus one digit/control sign) captured via a webcam into text, demonstrating a complete end-to-end Machine Learning pipeline from data preparation to real-time deployment.

**Key Features**

Real-Time Translation: Translates live webcam feed frames into text predictions with confidence scores.

Robust DIP Pipeline: Utilizes Otsu's Thresholding and Contour Detection to accurately segment the hand from complex backgrounds, ensuring high-quality input for the CNN.

High Accuracy: Achieved 100% accuracy on the independent test set after training on over 12,000 processed images.

Modular Architecture: Code is organized into dedicated modules (preprocess.py, model.py, train.py, realtime.py) for clean development and maintenance.


🛠️ **Technology Stack**

**Category**

1-Deep Learning

2-Image Processing

3-Data Handling

4-Development

**Tools/Libraries**  

1.TensorFlow / Keras   

2.OpenCV (cv2)     

3.NumPy, Pandas, Scikit-learn 

4.Python 3.x  


📂 **Repository Structure**

The project follows a standard machine learning structure:

**DIP PROJECT/Sign_language_translator**

    ├── data/

        │   ├── Train/         # Raw images organized into subfolders (A, B, C, ...)

        │   ├── test/          # Images reserved for final model evaluation

        │   └── processed_data.pickle  # NumPy array of normalized data and labels

    ├── models/

        │   ├── sign_language_translator_best.keras # The trained CNN model weights

        │   └── label_binarizer.pkl  # Stores the mapping of index-to-letter (0->A, 1->B, etc.)

        ├── notebooks/         # (Optional) For EDA or initial model exploration

        ├── outputs/           # Placeholder for reports, plots, or history files

    └── src/               # Source code for the application

    ├── preprocess.py  # Data loading, DIP pipeline, normalization
    
    ├── model.py       # CNN architecture definition
    
    ├── train.py       # Script for model training, callbacks, and saving
    
    ├── evaluate.py    # Generates classification reports and metrics
    
    └── realtime.py    # Main script for webcam capture and real-time prediction
