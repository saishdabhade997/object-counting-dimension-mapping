import cv2

for i in (0, 1):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok, frame = cap.read()
    if ok:
        cv2.imshow(f"Camera {i} - press any key", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cap.release()