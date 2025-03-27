import cv2
import numpy as np

def detect_balls(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 80, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    cv2.imshow("White Color Mask", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_positions = []
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)

        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-5))

        if 0.7 < circularity < 1.2 and radius > 5:
            ball_positions.append((int(x), int(y), int(radius)))

    return ball_positions

def detect_robot(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue1 = np.array([85, 50, 100])  
    upper_blue1 = np.array([100, 255, 255]) 

    lower_blue2 = np.array([100, 50, 100])  
    upper_blue2 = np.array([115, 255, 255])

    mask_blue1 = cv2.inRange(hsv, lower_blue1, upper_blue1)
    mask_blue2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
    mask_blue = cv2.bitwise_or(mask_blue1, mask_blue2) 

    lower_purple1 = np.array([125, 50, 50])  
    upper_purple1 = np.array([145, 255, 255]) 

    lower_purple2 = np.array([145, 50, 50])  
    upper_purple2 = np.array([160, 255, 255])

    mask_purple1 = cv2.inRange(hsv, lower_purple1, upper_purple1)
    mask_purple2 = cv2.inRange(hsv, lower_purple2, upper_purple2)
    mask_purple = cv2.bitwise_or(mask_purple1, mask_purple2)

    kernel = np.ones((7, 7), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_purple, _ = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    robot_position = None
    front_marker_position = None

    if contours_blue:
        largest_contour = max(contours_blue, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        if radius > 10:
            robot_position = (int(x), int(y))

    if contours_purple:
        (x, y), radius = cv2.minEnclosingCircle(max(contours_purple, key=cv2.contourArea))
        if radius > 1:
            front_marker_position = (int(x), int(y))

    robot_orientation = None
    if robot_position and front_marker_position:
        rx, ry = robot_position
        fx, fy = front_marker_position

        direction_vector = (fx - rx, fy - ry)
        robot_orientation = (robot_position, front_marker_position, direction_vector)

    return robot_orientation
