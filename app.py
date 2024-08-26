import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

model_options = {
    "Person Detection": "yolov8n-person.pt",
    "Face Detection": "yolov8n-face.pt",
    "Football Detection":"yolov8n-football.pt",
    #Model options
}

st.sidebar.title("Model Selection")
selected_model = st.sidebar.selectbox("Choose a YOLO model", list(model_options.keys()))

# Load the selected YOLO model
model = YOLO(model_options[selected_model])

def process_image(image):
    image = np.array(image)
    results = model.predict(image, conf=0.25, imgsz=1280, max_det=1000)
    annotated_image = results[0].plot()
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for result in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, result)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
    
    # Inpaint to remove objects
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return annotated_image, inpainted_image

st.title("YOLOv8 Image Object Detection and Removal")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    labeled_image, removed_objects_image = process_image(image)
    
    #Convert BGR to RGB for accurate display and download
    labeled_image_rgb = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
    removed_objects_image_rgb = cv2.cvtColor(removed_objects_image, cv2.COLOR_BGR2RGB)
    
    #Determine the file extension and MIME type based on the uploaded file
    file_extension = uploaded_file.name.split('.')[-1].lower()
    mime_type = f'image/{file_extension}'
    
    #Display the original, labeled, and object-removed images
    st.image(image, caption="Original Image", use_column_width=True)
    st.image(labeled_image, caption="Image with Labels", use_column_width=True)
    
    st.download_button(
        label="Download Image with Labels",
        data=cv2.imencode(f'.{file_extension}', labeled_image_rgb)[1].tobytes(),
        file_name=f'output_labeled_image.{file_extension}',
        mime=mime_type
    )
    
    st.image(removed_objects_image, caption="Image with Removed Objects", use_column_width=True)
    
    st.download_button(
        label="Download Image without Objects",
        data=cv2.imencode(f'.{file_extension}', removed_objects_image_rgb)[1].tobytes(),
        file_name=f'output_image_without_objects.{file_extension}',
        mime=mime_type
    )
