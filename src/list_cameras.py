import os
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(i)
    if cap.isOpened():
        ok, frame = cap.read()
        print(f"Camera {i}: {'working ' + str(frame.shape[:2]) if ok else 'opened but no frame'}")
    cap.release()
