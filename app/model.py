from utils import preprocess_text
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import streamlit as st
import numpy as np

# Variable global para el modelo BiLSTM
bilstm_model = None

# Carga del modelo BiLSTM de TensorFlow
def load_model(keras_model_path: str):
    global bilstm_model
    bilstm_model = tf.keras.models.load_model(keras_model_path)
    print("Modelo BiLSTM de Tensorflow cargado correctamente.")

# Predicción del modelo BiLSTM
def predict(text: str) -> tuple:
    global bilstm_model  # Ensure bilstm_model is accessed as a global variable
    processed_text = preprocess_text(text)

    # Create batch of size 1 for prediction
    batch = np.array([processed_text], dtype=object)
    # Prediction with the TensorFlow model
    probabilities = bilstm_model.predict(batch)
    probability = float(probabilities[0, 0])
    # Get binary label
    label = int(probability > 0.5)

    return label, processed_text, probability

def extract_vectorizer_layer():
    global bilstm_model
    if bilstm_model is None:
        raise ValueError("Model is not loaded. Please load the model first.")
    for layer in bilstm_model.layers:
        if "text_vectorization" in layer.name.lower():
            return layer
    raise ValueError("Vectorizer layer not found in the model.")

def extract_embedding_layer():
    global bilstm_model
    if bilstm_model is None:
        raise ValueError("Model is not loaded. Please load the model first.")
    for layer in bilstm_model.layers:
        if "embedding" in layer.name.lower():
            # Clone the embedding layer with mask_zero=False
            cfg = layer.get_config()
            cfg["mask_zero"] = False
            emb_no_mask = tf.keras.layers.Embedding.from_config(cfg)
            emb_no_mask.build((None, 1024))
            emb_no_mask.set_weights(layer.get_weights())
            return emb_no_mask
    raise ValueError("Embedding layer not found in the model.")

