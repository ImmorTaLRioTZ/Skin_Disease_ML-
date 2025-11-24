import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.applications import EfficientNetB4
from PIL import Image
import io

app = Flask(__name__)

# --- 1. CONFIGURATION ---
IMG_SIZE = 380
MODEL_PATH = "Skin_Disease_ML_model.h5"

# The exact order of symptoms the model expects
SYMPTOM_LIST = [
    'symptom_is_itchy', 'symptom_is_painful', 'symptom_is_asymptomatic', 
    'symptom_is_scaly', 'symptom_is_raised', 'symptom_has_blisters', 
    'symptom_is_oozing', 'symptom_is_bleeding', 'symptom_is_chronic', 
    'symptom_is_new_or_changing', 'symptom_loc_face', 'symptom_loc_torso', 
    'symptom_loc_arms_legs', 'symptom_loc_hands_feet', 'symptom_loc_genital_oral', 
    'symptom_loc_widespread', 'symptom_loc_single_lesion'
]

CLASS_NAMES = ['Acne or Rosacea', 'Actinic Keratoses', 'Atopic Dermatitis', 'Basal cell carcinoma', 'Benign keratosis-like lesions', 'Chickenpox', 'Cowpox', 'Dermatofibroma', 'Eczema', 'HFMD', 'Healthy', 'Herpes HPV and other STDs', 'Lupus', 'Lyme Disease', 'Measles', 'Melanocytic nevi', 'Melanoma', 'Monkeypox', 'Psoriasis Lichen Planus and related diseases', 'Scabies', 'Seborrheic Keratoses and other Benign Tumors', 'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 'Vascular Lesions or Vascular Tumors', 'Vitiligo', 'Warts Molluscum and other Viral Infections']

# --- 2. BUILD MODEL FUNCTION (The fix we created) ---
def build_model():
    # Base
    efficientnet_base = EfficientNetB4(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    efficientnet_base.trainable = False

    # Inputs
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    symptom_input = Input(shape=(len(SYMPTOM_LIST),), name="symptom_input")

    # Image Branch
    x = efficientnet_base(inputs)
    x = layers.GlobalAveragePooling2D(name="image_pooling")(x)
    x = layers.Dropout(0.2, name="top_dropout")(x)

    # Symptom Branch
    symptom_branch = layers.Dense(units=128, activation="relu", name="symptom_dense_1")(symptom_input)
    symptom_branch = layers.Dropout(0.2, name="symptom_dropout_1")(symptom_branch)
    symptom_branch = layers.Dense(units=256, activation="relu", name="symptom_dense_2")(symptom_branch)
    symptom_branch = layers.Dropout(0.2, name="symptom_dropout_2")(symptom_branch)
    symptom_branch = layers.Dense(units=512, activation="relu", name="symptom_dense_3")(symptom_branch)
    symptom_branch = layers.Dropout(0.2, name="symptom_dropout_3")(symptom_branch)

    # Concatenate
    x = layers.Concatenate(name="concatenate")([x, symptom_branch])

    # Output Structure
    x = layers.Dense(units=512, activation="relu", name="dense")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    x = layers.Dense(units=256, activation="relu", name="dense2")(x)
    x = layers.Dropout(0.2, name="dropout2")(x)
    x = layers.Dense(units=128, activation="relu", name="dense3")(x)
    x = layers.Dropout(0.2, name="dropout3")(x)
    outputs = layers.Dense(units=len(CLASS_NAMES), activation="softmax", name="predictions_classes")(x)

    model = Model(inputs=[inputs, symptom_input], outputs=outputs)
    return model

# --- 3. LOAD MODEL GLOBALY ---
print("Loading Model...")
model = build_model()
try:
    model.load_weights(MODEL_PATH)
    print("Weights loaded successfully.")
except Exception as e:
    print("Standard load failed, trying flexible load...")
    model.load_weights(MODEL_PATH, skip_mismatch=True, by_name=True)

# --- 4. ROUTES ---

@app.route('/', methods=['GET'])
def index():
    # Pass the symptom list to the template to generate the form dynamically
    # We replace underscores with spaces for better readability in the UI
    readable_symptoms = [(s, s.replace('symptom_', '').replace('_', ' ').title()) for s in SYMPTOM_LIST]
    return render_template('index.html', symptoms=readable_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # --- A. Process Image ---
    # Convert the file stream directly to a PIL Image
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    
    # Resize to model input size
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    # --- B. Process Symptoms ---
    # Create the vector. Form returns '1' if Yes is selected, otherwise we assume 0
    symptom_vector = []
    for symptom_key in SYMPTOM_LIST:
        # Get value from form (returns '1' or '0')
        val = request.form.get(symptom_key, '0')
        symptom_vector.append(float(val))
    
    symptom_array = np.array(symptom_vector, dtype=np.float32)
    symptom_array = np.expand_dims(symptom_array, axis=0)

    # --- C. Predict ---
    predictions = model.predict([img_array, symptom_array])
    
    idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[idx]
    confidence = predictions[0][idx] * 100

    return render_template('result.html', 
                           prediction=predicted_class, 
                           confidence=f"{confidence:.2f}")

if __name__ == '__main__':
    app.run(debug=True)