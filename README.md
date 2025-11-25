# 🩺 Multi-Modal Skin Disease Detection System

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?style=for-the-badge&logo=flask)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue?style=for-the-badge&logo=kaggle)

A robust hybrid Deep Learning framework capable of diagnosing **27 distinct skin diseases** by fusing **Computer Vision** (lesion images) with **Clinical Data** (patient symptoms). This project includes a training pipeline, evaluation scripts, and a user-friendly Flask web interface.

🔗 **[View Repository](https://github.com/ImmorTaLRioTZ/Skin_Disease_ML-)**

---

## 🚀 Project Overview

Dermatological diagnosis relies on both visual inspection and patient history. This model mimics that process by using a **Multi-Input Neural Network**:

1.  **Visual Stream:** Analyzes skin lesion morphology using **EfficientNetB4** (Transfer Learning).
2.  **Input Image:***Analyzes images from a dataset of 27 different types of images
3.  **Symptom Stream:** Analyzes a 17-point clinical symptom vector using a custom Dense Network.
4.  **Fusion:** Concatenates features from both streams to predict the specific condition.

## 📂 Dataset

The model is trained on the **Skin Disease Dataset** from Kaggle, which includes dermoscopic images mapped to metadata.

* **Dataset Link:** [Kaggle: Skin Disease Dataset](https://kaggle.com/datasets/bc094de636703778a683389d5a2d60ca848254baed5ad2c7f93700a5cd0840dd)
* **Structure:** Images are organized by class folders. Metadata (symptoms) is mapped via CSV.

## 🧠 Model Architecture

The architecture handles two distinct data types:

```
graph TD
    A[Input: Image (380x380x3)] -->|EfficientNetB4| B[Global Avg Pooling]
    C[Input: Symptoms (Vector of 17)] -->|Dense Layers| D[Feature Extraction]
    B --> E[Concatenation Layer]
    D --> E
    E --> F[Dense Layer 512]
    F --> G[Dense Layer 256]
    G --> H[Output Softmax (27 Classes)]

***Project Structure:***

Skin_Disease_ML/
│
├── app.py                     # Flask Web Application (Inference Interface)
├── Skin_Disease_train.py      # Main training script (Data loading, augmentation, model fit)
├── Skin_Disease_TEST.py       # Script for local testing and debugging predictions
├── Skin_Disease_ML_model.h5   # The trained weights file (Generated after training)
├── Skin_Disease_Metadata.csv  # Symptom mapping file (Required for prediction)
│
└── templates/                 # HTML templates for Flask
    ├── index.html             # User input form (Image upload + Symptom checklist)
    └── result.html            # Prediction results page

```
***Installation Guide:***
Clone the repository:
git clone [https://github.com/ImmorTaLRioTZ/Skin_Disease_ML-](https://github.com/ImmorTaLRioTZ/Skin_Disease_ML-)
cd Skin_Disease_ML-

***Install dependencies:***
pip install tensorflow pandas numpy scikit-learn flask pillow opencv-python matplotlib

***To train the model***
python Skin_Disease_train.py

***Install Weights separately in the same directory(if you want to skip the training overhead)***
https://huggingface.co/ImmoRTaLRioTZ/Skin_Disease_ML_model/tree/main

***Running the model in a clean UI:***
python app.py

***Running the model in CLI for debugging:***
python Skin_Disease_TEST.py
