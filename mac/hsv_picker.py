import cv2
import numpy as np

def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
        pixel = hsv[y, x]
        h, s, v = pixel
        print(f"HSV at ({x}, {y}): H={h}, S={s}, V={v}")

        preview = np.zeros((100, 300, 3), dtype=np.uint8)
        preview[:] = pixel
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_HSV2BGR)
        cv2.imshow("Picked Color", preview_bgr)

def main():
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("Could not open camera.")
        return

    print("Click on the image to get HSV values. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        display_frame = frame.copy()
        cv2.imshow("HSV Picker", display_frame)
        cv2.setMouseCallback("HSV Picker", pick_color, display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
