import os
import ultralytics
import streamlit as st
import torch
from PIL import Image


@st.cache_resource
def load_model():
    return torch.hub.load('yolov5', 'custom', path='best.pt', source='local')

model = load_model()

st.title("🐘 Elephant Detector")
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    with st.spinner('Running detection...'):
        results = model(image, size=640)
        results.render()
        st.image(results.ims[0], caption='Detected Elephants', use_column_width=True)
