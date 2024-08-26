from ultralytics import YOLO
import cv2
import numpy as np
import urllib.request

model_url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-person.pt"
urllib.request.urlretrieve(model_url, "yolov8n-person.pt")
model = YOLO('yolov8n-person.pt')  # Replace with the correct model file path

#Input image
image_path = 'road.jpg'
image = cv2.imread(image_path)

#Detect object using the specific model
results = model.predict(image, conf=0.25, imgsz=1280, max_det=1000)

annotated_image = results[0].plot()
output_path = 'output_labeled_image.jpg'
cv2.imwrite(output_path, annotated_image) # save file of labelled

mask = np.zeros(image.shape[:2], dtype=np.uint8) #Inpainting mask

#Fill mask with detected person's area
for result in results[0].boxes.xyxy:
    x1, y1, x2, y2 = map(int, result)
    cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)


inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA) #inpainting


output_path = 'output_image_without_objects.jpg' #save file of removed detected object
cv2.imwrite(output_path, inpainted_image)


cv2.imshow('Original Image',image)
cv2.imshow('Image with Labels',annotated_image)
cv2.imshow('Image with Removed Objects', inpainted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()