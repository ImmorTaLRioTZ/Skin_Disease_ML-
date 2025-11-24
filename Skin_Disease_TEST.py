import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras import Input, layers, Model
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Rescaling

# --- CONFIGURATION ---
IMG_SIZE = 380
BATCH_SIZE = 32
# REPLACE THIS WITH THE PATH TO YOUR TEST/VALIDATION FOLDER
TEST_DATA_DIR = "/path/to/your/dataset/test" 

# --- LOAD METADATA ---
df_lookup = pd.read_csv("Skin_Disease_Metadata.csv").sort_values(by="label")
SYMPTOM_COL = df_lookup.columns.tolist()[1:]
NUM_SYMPTOMS = len(SYMPTOM_COL)
NUM_CLASSES = len(df_lookup)
CLASS_NAMES = df_lookup["label"].tolist()

# Create a constant tensor for looking up symptoms by label index
symptom_lookup_table = tf.constant(df_lookup[SYMPTOM_COL].values, dtype=tf.float32)

# --- MODEL BUILDING (Using your working structure) ---
def build_model():
    efficientnet_base = EfficientNetB4(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    efficientnet_base.trainable = False

    aug_and_rescale_block = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomBrightness(factor=0.2),
        layers.RandomContrast(factor=0.2),
        layers.RandomSaturation(factor=0.2),
        layers.RandomHue(factor=0.1),
        layers.RandomRotation(0.1),
        Rescaling(1./255)
    ], name="augmentation_and_rescaling")

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = aug_and_rescale_block(inputs) 
    
    x = efficientnet_base(x)
    x = layers.GlobalAveragePooling2D(name="image_pooling")(x)
    x = layers.Dropout(0.2, name="top_dropout")(x)

    symptom_input = Input(shape=(NUM_SYMPTOMS,), name="symptom_input")
    symptom_branch = layers.Dense(units=128, activation="relu", name="symptom_dense_1")(symptom_input)
    symptom_branch = layers.Dropout(0.2, name="symptom_dropout_1")(symptom_branch)
    symptom_branch = layers.Dense(units=256, activation="relu", name="symptom_dense_2")(symptom_branch)
    symptom_branch = layers.Dropout(0.2, name="symptom_dropout_2")(symptom_branch)
    symptom_branch = layers.Dense(units=512, activation="relu", name="symptom_dense_3")(symptom_branch)
    symptom_branch = layers.Dropout(0.2, name="symptom_dropout_3")(symptom_branch)

    x = layers.Concatenate(name="concatenate")([x, symptom_branch])
    x = layers.Dense(units=512, activation="relu", name="dense")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    x = layers.Dense(units=256, activation="relu", name="dense2")(x)
    x = layers.Dropout(0.2, name="dropout2")(x)
    x = layers.Dense(units=128, activation="relu", name="dense3")(x)
    x = layers.Dropout(0.2, name="dropout3")(x)
    x = layers.Dense(units=NUM_CLASSES, activation="softmax", name="predictions_classes")(x)

    model = Model(inputs=[inputs, symptom_input], outputs=x)
    return model

# --- 1. INITIALIZE & LOAD WEIGHTS ---
model = build_model()
print("Model built.")

try:
    model.load_weights("Skin_Disease_ML_model.h5")
    print("Weights loaded successfully.")
except ValueError:
    print("Standard load failed, trying flexible load...")
    model.load_weights("Skin_Disease_ML_model.h5", skip_mismatch=True, by_name=True)

# --- 2. HELPER TO PREPARE DATASET ---
def format_eval_data(image, label_id):
    symptom_vector = tf.gather(symptom_lookup_table, label_id)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    
    return ((image, symptom_vector), label_id)

# --- 3. EVALUATION FUNCTION ---
def evaluate_model_on_dataset(data_dir):
    print(f"\n--- Loading Test Data from {data_dir} ---")
    
    # Load dataset from folder
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        interpolation='nearest',
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Add the symptom vectors to the input
    test_ds = test_ds.map(format_eval_data, num_parallel_calls=tf.data.AUTOTUNE)

    # Variables to store results
    y_true = []
    y_pred = []

    print("Running predictions on test set...")
    # Iterate over the dataset
    for inputs, labels in test_ds:
        # inputs is a tuple: (images, symptoms)
        predictions = model.predict(inputs, verbose=0)
        
        # Get the index of the highest probability
        predicted_ids = np.argmax(predictions, axis=1)
        
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_ids)

    # --- CALCULATE METRICS ---
    print("\n--- Performance Metrics ---")
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy:  {acc:.4f}")

    # Precision, Recall, F1 (weighted average accounts for class imbalance)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

# --- RUN THE EVALUATION ---
evaluate_model_on_dataset(TEST_DATA_DIR)
# --- SINGLE IMAGE PREDICTION (Your existing helper) ---
def predict_single_image(image_path, symptom_vector):
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    symptom_array = np.array(symptom_vector, dtype=np.float32)
    symptom_array = np.expand_dims(symptom_array, axis=0)

    predictions = model.predict([img_array, symptom_array])
    idx = np.argmax(predictions[0])
    return CLASS_NAMES[idx], predictions[0][idx]

# Example Usage
# image = "Urticaria-191x300.png"
# symptom_matrix = [1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,1,0]
# cls, conf = predict_single_image(image, symptom_matrix)
# print(f"Class: {cls}, Conf: {conf}")