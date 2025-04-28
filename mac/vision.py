import cv2
import numpy as np

def detect_balls(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 80, 255])
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([20, 255, 255])

    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    cv2.imshow("White Color Mask", mask_white)
    cv2.imshow("Orange Color Mask", mask_orange)

    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_positions = []
    for cnt in contours_orange:
        (x, y), radius = cv2.minEnclosingCircle(cnt)

        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-5))
        

        if 0.7 < circularity < 1 and 19 > radius > 14:
            print(circularity)
            ball_positions.append((int(x), int(y), int(radius), 1))
    
    for cnt in contours_white:
        (x, y), radius = cv2.minEnclosingCircle(cnt)

        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-5))
        

        if 0.7 < circularity < 1 and 19 > radius > 14:
            print(circularity)
            ball_positions.append((int(x), int(y), int(radius), 0))


    return ball_positions

def detect_robot(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 20, 100]) 
    upper_green = np.array([90, 120, 230]) 
    mask_green = cv2.inRange(hsv, lower_green, upper_green) 

    lower_front = np.array([140, 40, 80])
    upper_front = np.array([220, 180, 240])
    mask_front = cv2.inRange(hsv, lower_front, upper_front)

    kernel = np.ones((7, 7), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    
    mask_front = cv2.morphologyEx(mask_front, cv2.MORPH_OPEN, kernel)
    mask_front = cv2.morphologyEx(mask_front, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Green Color Mask", mask_green)
    cv2.imshow("Front Mask", mask_front)

    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_front, _ = cv2.findContours(mask_front, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    robot_position = None
    front_marker_position = None

    if contours_green:
        largest_contour = max(contours_green, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        if radius > 2:
            robot_position = (int(x), int(y))

    if contours_front:
        (x, y), radius = cv2.minEnclosingCircle(max(contours_front, key=cv2.contourArea))
        if radius > 2:
            front_marker_position = (int(x), int(y))

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
