import math
import cv2
import numpy as np
import time

def _make_kf():
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix      = np.array([[1,0,1,0],
                                         [0,1,0,1],
                                         [0,0,1,0],
                                         [0,0,0,1]], np.float32)
    kf.measurementMatrix     = np.eye(2,4, dtype=np.float32)
    kf.processNoiseCov       = np.eye(4, dtype=np.float32)*1e-2
    kf.measurementNoiseCov   = np.eye(2, dtype=np.float32)*1e-1
    kf.errorCovPost          = np.eye(4, dtype=np.float32)
    return kf

front_kf, back_kf = _make_kf(), _make_kf()
last_front = last_back = None
last_seen  = time.time()
MAX_MISS   = 2.0

egg_location = []

def detect_balls(frame, egg, robot_position, front_marker):
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
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([255, 50, 255])
    lower_orange = np.array([10, 0, 0])
    upper_orange = np.array([35, 255, 255])

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
    contours_white, _ = cv2.findContours(
        mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_orange, _ = cv2.findContours(
        mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                is_inside_egg = False
                if egg:
                    is_inside_egg = any(np.linalg.norm(np.array((x, y)) - np.array((ex, ey))) < er for (ex, ey, er, _) in egg)
                is_inside_robot = False
                if robot_position and front_marker:
                    # Brug midtpunkt mellem bagende og front
                    dist_to_back = np.linalg.norm(np.array((x, y)) - np.array(robot_position))
                    dist_to_front = np.linalg.norm(np.array((x, y)) - np.array(front_marker))
                    is_inside_robot = dist_to_back < 80 or dist_to_front < 80


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
            x2, y2 = min(
                x + roi_size, hsv.shape[1] - 1), min(y + roi_size, hsv.shape[0] - 1)
            roi = hsv[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            avg_hsv = cv2.mean(roi)[:3]
            h, s, v = avg_hsv

            # Tjek for hvid bold
            is_white = (s < 100 and v > 170)

            # Tjek for orange bold
            is_orange = (12 <= h <= 32 and s >= 85 and v >= 180)
            is_inside_egg = False
            if egg:
                is_inside_egg = any(np.linalg.norm(np.array((x, y)) - np.array((ex, ey))) < er for (ex, ey, er, _) in egg)
            is_inside_robot = False
            if robot_position:
                dist_to_back = np.linalg.norm(np.array((x, y)) - np.array(robot_position))
                dist_to_front = np.linalg.norm(np.array((x, y)) - np.array(front_marker))
                is_inside_robot = dist_to_back < 60 or dist_to_front < 80

            if not is_inside_egg and not is_inside_robot:
                if is_white:
                    ball_positions.append((x, y, r, 0))
                elif is_orange:
                    ball_positions.append((x, y, r, 1))

    return ball_positions

def detect_robot(frame):
    global last_front, last_back, last_seen

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    frame_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    frame_normalized = np.zeros_like(frame_clahe)
    cv2.normalize(frame_clahe, frame_normalized, 0, 255, cv2.NORM_MINMAX)

    hsv = cv2.cvtColor(frame_normalized, cv2.COLOR_BGR2HSV)

    kernel = np.ones((5, 5), np.uint8)

    color_ranges = {
        "front_left":  {"color": (255, 255, 255), "lower": np.array([88, 0, 0]),  "upper": np.array([130, 255, 255])},
        "front_right": {"color": (0, 255, 0),      "lower": np.array([45, 0, 0]),  "upper": np.array([75, 255, 255])},
        "back_left":   {"color": (255, 0, 255),    "lower": np.array([165, 0, 0]), "upper": np.array([105, 255, 255])},
        "back_right":  {"color": (255, 0, 0),      "lower": np.array([145, 0, 0]), "upper": np.array([170, 255, 255])},
    }

    detected = {}
    masks = {}

    for label, props in color_ranges.items():
        mask = cv2.inRange(hsv, props["lower"], props["upper"])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        masks[label] = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            x, y, radius = int(x), int(y), int(radius)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter ** 2 + 1e-5))
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)
            aspect_ratio = float(w_rect) / h_rect
            if 0.7 < circularity < 1.3 and 0.7 < aspect_ratio < 1.4 and area > 500:
                center = (x, y)
                detected[label] = center
                cv2.circle(frame, center, radius, props["color"], 2)
                cv2.putText(frame, label, (x - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, props["color"], 2)
                break

    for label, mask in masks.items():
        cv2.imshow(f"{label} mask", mask)

    # Brug kendte afstande
    width = 90
    length = 115

    positions = detected.keys()
    front, back = None, None
    direction_vector = None

    if "front_left" in positions and "front_right" in positions:
        front = tuple(np.mean([detected["front_left"], detected["front_right"]], axis=0).astype(int))

    if "back_left" in positions and "back_right" in positions:
        back = tuple(np.mean([detected["back_left"], detected["back_right"]], axis=0).astype(int))

    if "front_left" in positions and "back_left" in positions and (front is None or back is None):
        left_front = np.array(detected["front_left"])
        left_back = np.array(detected["back_left"])
        v = left_front - left_back
        direction = v / (np.linalg.norm(v) + 1e-5)
        center = (left_front + left_back) / 2
        # Vinkelret vektor
        perpendicular = np.array([-direction[1], direction[0]])
        center = center + perpendicular * (width / 2)  # flyt ind mod midten
        front = tuple((center + length/2 * direction).astype(int))
        back = tuple((center - length/2 * direction).astype(int))

    if "front_right" in positions and "back_right" in positions and (front is None or back is None):
        right_front = np.array(detected["front_right"])
        right_back = np.array(detected["back_right"])
        v = right_front - right_back
        direction = v / (np.linalg.norm(v) + 1e-5)
        center = (right_front + right_back) / 2
        # Vinkelret vektor
        perpendicular = np.array([direction[1], -direction[0]])
        center = center + perpendicular * (width / 2)  # flyt ind mod midten
        front = tuple((center + length/2 * direction).astype(int))
        back = tuple((center - length/2 * direction).astype(int))


    if "front_left" in positions and "back_right" in positions and (front is None or back is None):
        front_pt = np.array(detected["front_left"], dtype=np.float32)
        back_pt  = np.array(detected["back_right"], dtype=np.float32)
        center = (front_pt + back_pt) / 2
        direction = front_pt - back_pt
        direction /= (np.linalg.norm(direction) + 1e-5)

        length_dir = direction
        width_dir = np.array([-length_dir[1], length_dir[0]])

        center_front = center + width_dir * (width / 2)
        center_back = center - width_dir * (width / 2)

        front = tuple((center_front + length_dir * (length / 2)).astype(int))
        back  = tuple((center_back - length_dir * (length / 2)).astype(int))


    if "front_right" in positions and "back_left" in positions and (front is None or back is None):
        front_pt = np.array(detected["front_right"], dtype=np.float32)
        back_pt  = np.array(detected["back_left"], dtype=np.float32)
        center = (front_pt + back_pt) / 2
        direction = front_pt - back_pt
        direction /= (np.linalg.norm(direction) + 1e-5)

        length_dir = direction
        width_dir = np.array([-length_dir[1], length_dir[0]])

        center_front = center - width_dir * (width / 2)
        center_back = center + width_dir * (width / 2)

        front = tuple((center_front + length_dir * (length / 2)).astype(int))
        back  = tuple((center_back - length_dir * (length / 2)).astype(int))

    if "front_left" in positions and "front_right" in positions and (front is None or back is None):
        f_left = np.array(detected["front_left"], dtype=np.float32)
        f_right = np.array(detected["front_right"], dtype=np.float32)
        front_center = (f_left + f_right) / 2

        # Vektor fra højre til venstre = bredderetning
        width_vec = f_left - f_right
        width_dir = width_vec / (np.linalg.norm(width_vec) + 1e-5)

        # Længderetning er vinkelret
        length_dir = np.array([width_dir[1], -width_dir[0]])

        front = tuple((front_center).astype(int))
        back = tuple((front_center + length * length_dir).astype(int))

    if "back_left" in positions and "back_right" in positions and (front is None or back is None):
        b_left = np.array(detected["back_left"], dtype=np.float32)
        b_right = np.array(detected["back_right"], dtype=np.float32)
        back_center = (b_left + b_right) / 2

        width_vec = b_left - b_right
        width_dir = width_vec / (np.linalg.norm(width_vec) + 1e-5)
        length_dir = np.array([width_dir[1], -width_dir[0]])

        back = tuple((back_center).astype(int))
        front = tuple((back_center - length * length_dir).astype(int))

    t_now = time.time()

    def _step(kf, meas, last):
        if meas is not None:                 # har vi en måling?
            kf.correct(np.float32([[meas[0]],[meas[1]]]))
            last = meas
        pred = kf.predict()                  # ny forudsigelse
        return (int(pred[0]),int(pred[1])) if meas is None else meas, last

    front, last_front = _step(front_kf, front, last_front)
    back , last_back  = _step(back_kf , back , last_back)

    if front and back:
        d = np.linalg.norm(np.subtract(front,back))
        if 85 <= d <= 120: 
            last_seen = t_now
        else:         
            front = back = None

    if front is None and back is None and (t_now-last_seen) > MAX_MISS:
        return None 

    front = front or last_front
    back  = back  or last_back
    if front is None or back is None:
        return None
    
    if front is not None and back is not None:
        direction_vector = (front[0] - back[0], front[1] - back[1])
        cv2.arrowedLine(frame, back, front, (255, 255, 255), 2)

    cv2.imshow("Robot Debug", frame)

    if front is not None and back is not None:
        return back, front, direction_vector
    else:
        return None



def detect_cross(frame, robot_position=None, front_marker=None, ball_positions=None, 
                FIELD_X_MIN = None, 
                FIELD_X_MAX = None,
                FIELD_Y_MIN = None,
                FIELD_Y_MAX = None):
    
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
        minLineLength=20,
        maxLineGap=10
    )

    cross_lines = []

    if lines is not None:            #TILFØJET FOR AT KØRE KAMERA HJEMME !!!DAVID!!!
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            too_close_to_robot = False
            too_close_to_ball = False

        if robot_position:
            rx, ry = robot_position
            if np.linalg.norm(np.array((cx, cy)) - np.array((rx, ry))) < 150:
                too_close_to_robot = True

        if front_marker:
            fx, fy = front_marker
            if np.linalg.norm(np.array((cx, cy)) - np.array((fx, fy))) < 150:
                too_close_to_robot = True

            if ball_positions:
                for (bx, by, _, _) in ball_positions:
                    if np.linalg.norm(np.array((cx, cy)) - np.array((bx, by))) < 30:
                        too_close_to_ball = True
                        break

        too_close_to_field = (
        cx < FIELD_X_MIN + 50 or
        cx > FIELD_X_MAX - 50 or
        cy < FIELD_Y_MIN + 50 or
        cy > FIELD_Y_MAX - 50
        )

        if not too_close_to_robot and not too_close_to_ball and not too_close_to_field:
            cross_lines.append((x1, y1, x2, y2))


    # Debug mask
    cv2.imshow("Cross Mask", mask)
    cv2.imshow("Cross Edges", edges)

    return cross_lines  # Liste af linjer


def detect_egg(frame, robot_position, front_marker):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([255, 80, 255])

    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    contours_white, _ = cv2.findContours(
        mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    egg = []

    for cnt in contours_white:
        (x, y), radius = cv2.minEnclosingCircle(cnt)

        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-5))

        if 0.6 < circularity and radius > 20:

            is_inside_robot = False
            if robot_position:
                dist_to_back = np.linalg.norm(np.array((x, y)) - np.array(robot_position))
                dist_to_front = np.linalg.norm(np.array((x, y)) - np.array(front_marker))
                is_inside_robot = dist_to_back < 60 or dist_to_front < 80

            if not is_inside_robot:
                egg.append((int(x), int(y), int(radius), 0))

    return egg


def inside_field(segments):
    xs, ys = [], []
    for seg in segments:
        if isinstance(seg[0], tuple):
            x1, y1, x2, y2 = seg[0]
        else:
            x1, y1, x2, y2 = seg
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    return min(xs), max(xs), min(ys), max(ys)


def filter_barriers_inside_field(barriers, frame_shape, margin=20):
    h, w = frame_shape[:2]
    filtered = []
    for (x1, y1, x2, y2), center in barriers:
        cx, cy = center
        if margin < cx < w - margin and margin < cy < h - margin:
            filtered.append(((x1, y1, x2, y2), center))
    return filtered
