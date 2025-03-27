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

    lower_green = np.array([40, 100, 50])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green) 

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    kernel = np.ones((7, 7), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Green Color Mask", mask_green)
    cv2.imshow("Red Mask 2", mask_red2)
    cv2.imshow("Red Combined", mask_red)

    print("Red mask nonzero:", cv2.countNonZero(mask_red))

    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("Found red contours:", len(contours_red))

    robot_position = None
    front_marker_position = None

    if contours_green:
        largest_contour = max(contours_green, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        if radius > 1:
            robot_position = (int(x), int(y))

    if contours_red:
        print("Red contour area:", cv2.contourArea(max(contours_red, key=cv2.contourArea)))
        (x, y), radius = cv2.minEnclosingCircle(max(contours_red, key=cv2.contourArea))
        if radius > 1:
            front_marker_position = (int(x), int(y))

    robot_orientation = None
    if robot_position and front_marker_position:
        rx, ry = robot_position
        fx, fy = front_marker_position

        direction_vector = (fx - rx, fy - ry)
        robot_orientation = (robot_position, front_marker_position, direction_vector)

    return robot_orientation
