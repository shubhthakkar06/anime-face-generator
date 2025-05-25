import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the generator model
@st.cache_resource
def load_generator(model_path="models/g_model.h5"):
    return tf.keras.models.load_model(model_path)

# Generate an image using the generator
def generate_image(model, latent_dim=100):
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated_image = model.predict(noise)[0]
    generated_image = (generated_image + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]
    return generated_image

# Streamlit UI
st.title("Anime Face Generator 🎨 (Keras Model)")

model = load_generator()

if st.button("Generate New Face"):
    img = generate_image(model)
    st.image(img, caption="Generated Anime Face", use_column_width=True)
