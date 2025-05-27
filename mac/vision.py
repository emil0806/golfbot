import cv2
import numpy as np

import cv2
import numpy as np

def detect_balls(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to handle changing lighting
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 3
    )

    cv2.imshow("Adaptive Threshold", thresh)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame, contours, -1, (0, 255, 255), 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ball_positions = []

    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # Shape filter: must be round and reasonably sized
        if 0.4 < circularity < 1.4 and 10 < radius < 30:
            print(f"circle: {circularity}, radius: {radius}")
            # Create a mask for this circle
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)

            # Sample mean HSV inside the circle
            mean_hsv = cv2.mean(hsv, mask=mask)
            h, s, v, _ = mean_hsv

            # Classify color based on average HSV
            if 0 < h < 40 and s > 70 and v > 150:
                color = 1  # orange
            elif s < 80 and v > 180:
                color = 0  # white
            else:
                color = 0  # discard unclassified object

            ball_positions.append((int(x), int(y), int(radius), color))

    print(f"Balls: {ball_positions}")
    return ball_positions


def detect_robot(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    robot_position = None
    front_marker_position = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        corners = len(approx)

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))

        # Create mask for this shape to sample HSV
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_hsv = cv2.mean(hsv, mask=mask)
        h, s, v, _ = mean_hsv

        # Detect front marker (blue triangle)
        if corners == 3 and 80 < h < 150 and s > 100:
            front_marker_position = center

        # Detect back marker (green square)
        elif corners == 4 and 40 < h < 100 and s > 40:
            robot_position = center


    robot_orientation = None
    if robot_position and front_marker_position:
        rx, ry = robot_position
        fx, fy = front_marker_position

        direction_vector = (fx - rx, fy - ry)
        robot_orientation = (robot_position, front_marker_position, direction_vector)

    return robot_orientation

def detect_barriers(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 150])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 150])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    barriers = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10: 
            barriers.append((x, y, w, h))

    return barriers
