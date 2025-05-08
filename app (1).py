import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load the trained model
model = load_model('hand_gesture_cnn.h5')
gesture_labels = {
    "01_palm": 0, "02_l": 1, "03_fist": 2, "04_fist_moved": 3, "05_thumb": 4,
    "06_index": 5, "07_ok": 6, "08_palm_moved": 7, "09_c": 8, "10_down": 9
}
label_names = list(gesture_labels.keys())

def preprocess_frame(frame, target_size=(64, 64), padding_size=2):
    # Convert to grayscale
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using histogram equalization
    img = cv2.equalizeHist(img)
    
    # Resize to target size
    img_orig = cv2.resize(img, target_size)
    img = cv2.resize(img, target_size)
    
    # Add padding
    img = cv2.copyMakeBorder(img, padding_size, padding_size, padding_size, padding_size, 
                             cv2.BORDER_CONSTANT, value=0)
    
    # Sobel edge detection (optional, can skip if not helpful)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    img_edges = np.uint8(sobel_combined)
    
    # Pad original to match processed size
    img = cv2.copyMakeBorder(img_orig, padding_size, padding_size, padding_size, padding_size, 
                             cv2.BORDER_CONSTANT, value=0)
    
    # Blend with more weight on original image
    img = (0.7 * img + 0.3 * img_edges).astype(np.uint8)
    
    # Crop back to target size
    img = img[padding_size:-padding_size, padding_size:-padding_size]
    
    # Normalize and reshape
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 64, 64, 1)
    return img
# Streamlit app
st.title("Live Hand Gesture Recognition")
run = st.checkbox('Run Webcam')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.write("Error: Could not open webcam.")
    exit()

prev_time = 0

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Error: Failed to capture frame.")
        break
    
    # Preprocess the frame
    processed_frame = preprocess_frame(frame)
    
    # Predict gesture
    prediction = model.predict(processed_frame, verbose=0)
    predicted_label = np.argmax(prediction, axis=1)[0]
    gesture_name = label_names[predicted_label]
    confidence = prediction[0][predicted_label] * 100
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    
    # Overlay prediction and FPS on the frame
    cv2.putText(frame, f"{gesture_name} ({confidence:.1f}%)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the frame in Streamlit
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
st.write("Webcam stopped.")