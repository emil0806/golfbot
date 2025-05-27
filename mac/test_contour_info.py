import cv2
import numpy as np

clicked_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Contours")
cv2.setMouseCallback("Contours", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()

    # Preprocessing
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 3
    )

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours
    cv2.drawContours(display, contours, -1, (0, 255, 255), 1)

    if clicked_point:
        for cnt in contours:
            if cv2.pointPolygonTest(cnt, clicked_point, False) >= 0:
                # You clicked inside this contour
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    circularity = 0
                else:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                print(f"[CLICKED] Radius: {radius:.2f}, Circularity: {circularity:.3f}")
                break
        clicked_point = None

    cv2.imshow("Contours", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
