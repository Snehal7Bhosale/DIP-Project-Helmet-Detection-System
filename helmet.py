from ultralytics import YOLO
import cv2
import os

# Load the YOLO model
model = YOLO(r"C:\Users\LENOVO\PycharmProjects\helmet\runs\detect\train\weights\best.pt")

# Perform object detection on an image
img = model.predict(r"C:\Users\LENOVO\Downloads\helmettest.png", show=True, conf=0.7, save=True)

# Save the image to the desktop directory

cv2.imwrite(r"C:\Users\LENOVO\Desktop", img)

# Display the image
cv2.imshow("Image", img)

# Wait for a key press to close the window
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
