import cv2
import numpy as np

egg_location = []

def detect_balls(frame, egg, robot_position=None):
    # Konverter til LAB og split kanaler
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE på L-kanalen
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Genopbyg og konverter til BGR → HSV
    lab_clahe = cv2.merge((l_clahe, a, b))
    frame_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2HSV)

    # Justeret HSV-grænser
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    lower_orange = np.array([12, 85, 230])
    upper_orange = np.array([32, 255, 255])

    # Masker
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # Ekstra L-kanal tærskel for meget lyse områder
    _, l_thresh = cv2.threshold(l_clahe, 220, 255, cv2.THRESH_BINARY)
    mask_white = cv2.bitwise_and(mask_white, l_thresh)

    # Morfologisk rensning
    kernel = np.ones((5, 5), np.uint8)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("White Color Mask", mask_white)
    cv2.imshow("Orange Color Mask", mask_orange)

    # Find konturer
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_positions = []

    def filter_contours(contours, color_id):
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            x, y, radius = int(x), int(y), int(radius)

            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-5))
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)
            aspect_ratio = float(w_rect) / h_rect

            if (
                10 < radius < 20 and
                0.8 < circularity < 1.2 and
                0.9 < aspect_ratio < 1.1 and
                area > 150
            ):
                
                # Tjek at bold ikke er inde i et æg
                is_inside_egg = any(np.linalg.norm(np.array((x, y)) - np.array((ex, ey))) < er for (ex, ey, er, _) in egg)
                is_inside_robot = False
                if robot_position:
                    rx, ry = robot_position
                    is_inside_robot = np.linalg.norm(np.array((x, y)) - np.array((rx, ry))) < 25

                if not is_inside_egg and not is_inside_robot:
                    ball_positions.append((x, y, radius, color_id))
    # Konturfiltrering
    filter_contours(contours_orange, 1)
    filter_contours(contours_white, 0)

    # HoughCircles som supplement (kun på CLAHE billede)
    gray = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=20)

    if circles is not None:
        for (x, y, r) in np.round(circles[0, :]).astype("int"):
            # Skip hvis bolden allerede er fundet via kontur
            if any(abs(x - bx) < 10 and abs(y - by) < 10 for bx, by, _, _ in ball_positions):
                continue

            roi_size = 7
            x1, y1 = max(x - roi_size, 0), max(y - roi_size, 0)
            x2, y2 = min(x + roi_size, hsv.shape[1] - 1), min(y + roi_size, hsv.shape[0] - 1)
            roi = hsv[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            avg_hsv = cv2.mean(roi)[:3]
            h, s, v = avg_hsv

            # Tjek for hvid bold
            is_white = (s < 100 and v > 170)

            # Tjek for orange bold
            is_orange = (12 <= h <= 32 and s >= 85 and v >= 180)

            is_inside_egg = any(np.linalg.norm(np.array((x, y)) - np.array((ex, ey))) < er for (ex, ey, er, _) in egg)
            is_inside_robot = False
            if robot_position:
                rx, ry = robot_position
                is_inside_robot = np.linalg.norm(np.array((x, y)) - np.array((rx, ry))) < 25

            if not is_inside_egg and not is_inside_robot:
                if is_white:
                    ball_positions.append((x, y, r, 0))
                elif is_orange:
                    ball_positions.append((x, y, r, 1))

    return ball_positions

def detect_robot(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Grøn til bagende
    lower_green = np.array([60, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Orange til fronten (rektangel)
    lower_orange = np.array([12, 85, 230])
    upper_orange = np.array([32, 255, 255])
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # Morfologisk rensning
    kernel = np.ones((7, 7), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Green Mask", mask_green)
    cv2.imshow("Orange Mask (Robot Front)", mask_orange)

    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    robot_position = None
    front_position = None

    # Find bagende (grøn)
    if contours_green:
        largest_green = max(contours_green, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_green)
        if radius > 2:
            robot_position = (int(x), int(y))

    # Find forende (orange rektangel)
    if contours_orange:
        largest_orange = max(contours_orange, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_orange)
        front_position = (int(x + w / 2), int(y + h / 2))

    robot_orientation = None
    if robot_position and front_position:
        rx, ry = robot_position
        fx, fy = front_position
        direction_vector = (fx - rx, fy - ry)
        robot_orientation = (robot_position, front_position, direction_vector)

    return robot_orientation


def detect_barriers(frame, robot_position=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rød farve (HSV wraparound)
    lower_red1 = np.array([5, 150, 150])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 150])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Let udglatning og edge detection
    blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find linjer med Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    barriers = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            barriers.append(((x1, y1, x2, y2), (cx, cy)))

    # Debug
    cv2.imshow("Barrier Mask", mask)
    cv2.imshow("Edges", edges)

    if robot_position:
        rx, ry = robot_position
        filtered_barriers = []
        for ((x1, y1, x2, y2), (cx, cy)) in barriers:
            dist = np.linalg.norm(np.array((cx, cy)) - np.array((rx, ry)))
            if dist > 40:  # justér radius hvis nødvendigt
                filtered_barriers.append(((x1, y1, x2, y2), (cx, cy)))
        return filtered_barriers
    else:
        return barriers


def detect_cross(frame, robot_position=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rød farveområde
    lower_red1 = np.array([0, 120, 150])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 150])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Kanter og linjedetektion
    blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=5,
            maxLineGap=10
        )
    
    cross_lines = []


    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if robot_position:
                rx, ry = robot_position
                dist = np.linalg.norm(np.array((cx, cy)) - np.array((rx, ry)))
                if dist < 40:  # Hvis for tæt på robot, skip
                    continue

            cross_lines.append((x1, y1, x2, y2))

    # Debug mask
    cv2.imshow("Cross Mask", mask)
    cv2.imshow("Cross Edges", edges)

    return cross_lines  # Liste af linjer

def detect_egg(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 80, 255])

    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    egg = []

    for cnt in contours_white:
        (x, y), radius = cv2.minEnclosingCircle(cnt)

        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-5))
        

        if 0.8 < circularity and radius > 20:
            egg.append((int(x), int(y), int(radius), 0))

    egg_location = egg
    return egg
