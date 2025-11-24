import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Rescaling
from tensorflow.keras import Input
from tensorflow.keras import layers
IMG_SIZE = 380
batch_size = 32


dataset_dir = 'SkinDisease3'
df_lookup = pd.read_csv("SkinDisease3/SkinDisease3/Skin_Disease_Metadata.csv").sort_values(by="label")
SYMPTOM_COL = df_lookup.columns.tolist()[1:]
NUM_SYMPTOMS = len(SYMPTOM_COL)
NUM_CLASSES = len(df_lookup)
CLASS_NAMES = df_lookup["label"].tolist()
print(f"Number of classes: {NUM_CLASSES}")
print(f"Class names: {CLASS_NAMES}")
print(f"Symptom columns: {SYMPTOM_COL}")
print(f"Number of symptoms: {NUM_SYMPTOMS}")

symptom_lookup_table = tf.constant(df_lookup[SYMPTOM_COL].values, dtype = tf.float32)


efficientnet_base = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
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
    #layers.GaussianBlur(3, sigma = 0.7),
    Rescaling(1./255)
], name = "augmentation_and_rescaling")
#rescaling layer to normalize pixel values to the [0, 1] range


#input layer
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

#Apply data augmentation layer, done already in the dataset formatting part
inputs = aug_and_rescale_block(inputs)
# Connect the base model to the output of the rescaling layer
x = efficientnet_base(inputs)
x = layers.GlobalAveragePooling2D(name="image_pooling")(x)
x = layers.Dropout(0.2, name="top_dropout")(x)

symptom_input = Input(shape=(NUM_SYMPTOMS,), name="symptom_input")
symptom_branch = layers.Dense(units = 128, activation  ="relu", name = "symptom_dense_1")(symptom_input)
symptom_branch = layers.Dropout(0.2, name="symptom_dropout_1")(symptom_branch)
symptom_branch = layers.Dense(units = 256, activation  ="relu", name = "symptom_dense_2")(symptom_branch)
symptom_branch = layers.Dropout(0.2, name="symptom_dropout_2")(symptom_branch)
symptom_branch = layers.Dense(units = 512, activation  ="relu", name = "symptom_dense_3")(symptom_branch)
symptom_branch = layers.Dropout(0.2, name="symptom_dropout_3")(symptom_branch)

x = layers.Concatenate(name="concatenate")([x, symptom_branch])
x = layers.Dense(units = 512, activation  ="relu", name = "dense")(x)
x = layers.Dropout(0.2, name="dropout")(x)
x = layers.Dense(units = 256, activation  ="relu", name = "dense2")(x)
x = layers.Dropout(0.2, name="dropout2")(x)
x = layers.Dense(units = 128, activation  ="relu", name = "dense3")(x)
x = layers.Dropout(0.2, name="dropout3")(x)
x = layers.Dense(units = NUM_CLASSES, activation = "softmax", name = "predictions_classes")(x)

model = tf.keras.Model(inputs = [inputs, symptom_input], outputs = x)
print(model.summary())


#training dataset

def lookup_symptoms_and_format(image, label_id):
  symptom_vector = tf.gather(symptom_lookup_table, label_id)
  image = aug_and_rescale_block(image, training=True)
  label_one_hot = tf.one_hot(label_id, depth=NUM_CLASSES)
  return ((image, symptom_vector), label_one_hot)

train_ds = tf.keras.utils.image_dataset_from_directory(
    "SkinDisease3/SkinDisease3/train",
    labels='inferred',
    label_mode='int',
    class_names=CLASS_NAMES,
    image_size=(IMG_SIZE, IMG_SIZE),
    interpolation='nearest',
    batch_size=batch_size,
    shuffle=True
)

train_ds = train_ds.map(lookup_symptoms_and_format,
                        num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

def format_test_data(image, label_id):
  symptom_vector = tf.gather(symptom_lookup_table, label_id)
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  image = Rescaling(1 / 255.0)(image)
  label_one_hot = tf.one_hot(label_id, depth=NUM_CLASSES)
  return ((image, symptom_vector), label_one_hot)

#validation dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    "SkinDisease3/SkinDisease3/test",
    labels='inferred',
    label_mode='int',
    class_names=CLASS_NAMES,
    image_size=(IMG_SIZE, IMG_SIZE),
    interpolation='nearest',
    batch_size=batch_size,
    shuffle=False
)
test_ds = test_ds.map(format_test_data,
                        num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


# Compile the model with your metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

#callback if accuracy < 98%
class EarlyStoppingCallback(tf.keras.callbacks.Callback):

    # Define the correct function signature for on_epoch_end method
    def on_epoch_end(self, epoch, logs=None):

        # Check if the accuracy is greater or equal to 0.98
        if logs['accuracy'] >= 0.98:

            # Stop training once the above condition is met
            self.model.stop_training = True

            print("\nReached 98% accuracy so cancelling training!")


#--- TRAIN THE MODEL ---
history = model.fit(
    train_ds,
    epochs=2,
    validation_data=test_ds,
    callbacks=[EarlyStoppingCallback()]
)

# --- EVALUATE THE MODEL ---
print("\n--- Final Evaluation ---")
results = model.evaluate(test_ds)
print(dict(zip(model.metrics_names, results)))

# ---SAVING THE WEIGHTS ---
save_path = "Skin_Disease_ML_model.h5"
print(f"Saving model to {save_path}...")
model.save(save_path)
print("Model saved successfully!")
