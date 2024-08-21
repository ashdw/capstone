# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 22:23:12 2024

@author: TABAD
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from moviepy.editor import VideoFileClip, ImageSequenceClip
import io
import base64

# Set the page config
st.set_page_config(
    page_title="Never2Far: Spacecraft Detection",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the YOLO model with caching
@st.cache_resource
def load_model():
    try:
        model = YOLO('C:/Users/TABAD/runs/detect/train3/weights/best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
    
# Function to process a single image
def process_image(image, model, conf_threshold):
    results = model(image, conf=conf_threshold)
    result_image = results[0].plot()

    # Convert BGR to RGB
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    return result_image_rgb

# Process frames
def process_frames(frames, model, conf_threshold):
    processed_frames = []
    progress_bar = st.progress(0)
    for idx, frame in enumerate(frames):
        results = model(frame, conf=confidence_threshold)
        result_frame = results[0].plot()
        processed_frames.append(result_frame)
        progress_bar.progress((idx + 1) / len(frames))
    return processed_frames

# Process GIF
def process_gif(gif_file, model, conf_threshold):
    gif = Image.open(gif_file)
    frames = []
    for frame in range(gif.n_frames):
        gif.seek(frame)
        rgb_frame = gif.convert('RGB')
        np_frame = np.array(rgb_frame)
        frames.append(np_frame)
    return process_frames(frames, model, conf_threshold)

# Create downloadable link
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This app uses YOLO to detect spacecraft in images, videos, and GIFs. "
    "Upload a file and click 'Detect Spacecraft' to see the results."
)

st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Main content
st.markdown("<h1 class='title'>🛰️ Never2Far: Spacecraft Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image, video, or GIF to detect spacecraft.</p>", unsafe_allow_html=True)

model = load_model()
if model is None:
    st.stop()

uploaded_file = st.file_uploader("Choose a file...", type=["png", "jpg", "jpeg", "bmp", "mp4", "gif"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[1]
    
    try:
        if file_type in ['png', 'jpg', 'jpeg', 'bmp']:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button('🔍 Detect Spacecraft'):
                with st.spinner('Detecting spacecraft...'):
                    #results = model(image, conf=confidence_threshold)
                    processed_image = process_image(image, model, confidence_threshold)
                
                #result_image = results[0].plot()
                st.image(processed_image, caption="Detected Spacecraft", use_column_width=True)
                st.success("Detection complete!")
        
        elif file_type in ['mp4', 'gif']:
            file_bytes = uploaded_file.read()
            if file_type == 'mp4':
                st.video(file_bytes)
            else:
                st.image(file_bytes, caption="Uploaded GIF", use_column_width=True)
            
            if st.button('🔍 Detect Spacecraft'):
                with st.spinner('Detecting spacecraft... This may take a while.'):
                    if file_type == 'mp4':
                        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                            temp_file.write(file_bytes)
                            temp_file.flush()
                            video = VideoFileClip(temp_file.name)
                            frames = [np.array(frame) for frame in video.iter_frames()]
                            processed_frames = process_frames(frames, model, confidence_threshold)
                            fps = video.fps
                    else:
                        processed_frames = process_gif(io.BytesIO(file_bytes), model, confidence_threshold)
                        fps = 10  # Default FPS for GIFs
                
                if processed_frames:
                    clip = ImageSequenceClip(processed_frames, fps=fps)
                    
                    if file_type == 'gif':
                        output_file = "processed_animation.gif"
                        clip.write_gif(output_file, fps=fps)
                        st.image(output_file)
                    else:
                        output_file = "processed_video.mp4"
                        clip.write_videofile(output_file, fps=fps)
                        st.video(output_file)
                    
                    st.success("Detection complete!")
                    st.markdown(get_binary_file_downloader_html(output_file, 'Processed File'), unsafe_allow_html=True)
                else:
                    st.error("No frames were processed. The file may be corrupt or unsupported.")
        
        else:
            st.error("Unsupported file type. Please upload an image, video, or GIF.")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

st.write("---")
st.write("Developed by ADW")