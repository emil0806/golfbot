from collections import deque
import math
import cv2
import numpy as np
import time
from robot_controller import RobotController

import globals_config as g

# ----------  BALL STABILISERING  ----------
stable_balls        = []             # [(x,y,r,color,missing_frames)]
ball_history        = deque(maxlen=5)
COLLECT_DIST_PIX      = 40           # hvor tæt bag-markøren skal være før bold "forsvinder"


# ----------  EGG STABILISERING  ----------
egg_history = deque(maxlen=5)
stable_eggs = []  # [(x, y, r, color, missing_frames)]

def stabilize_egg(current_egg, distance_threshold=20, min_frames=5, max_missing=12):
    global egg_history, stable_eggs

    egg_history.append(current_egg)

    # --------- Saml æg-kandidater over tid ---------
    candidates = {}
    for frame in egg_history:
        for (x, y, r, col) in frame:
            key = (round(x / distance_threshold),
                   round(y / distance_threshold))
            candidates.setdefault(key, []).append((x, y, r, col))

    # --------- Tilføj nye stabile æg ---------
    for _, detections in candidates.items():
        if len(detections) >= min_frames:
            xs, ys, rs = zip(*[(d[0], d[1], d[2]) for d in detections])
            avg_x, avg_y, avg_r = int(np.mean(xs)), int(np.mean(ys)), int(np.mean(rs))
            new_egg = (avg_x, avg_y, avg_r, 0, 0)  # sidste = missing counter

            # Undgå dubletter
            if not any(np.hypot(se[0] - avg_x, se[1] - avg_y) < distance_threshold for se in stable_eggs):
                stable_eggs.append(new_egg)

    # --------- Opdater eksisterende stabile æg ---------
    updated_eggs = []
    for (sx, sy, sr, scol, miss) in stable_eggs:
        # Er dette æg stadig synligt i nuværende frame?
        visible = any(np.hypot(sx - cx, sy - cy) < distance_threshold for (cx, cy, _, _) in current_egg)

        if visible:
            updated_eggs.append((sx, sy, sr, scol, 0))  # nulstil miss counter
        else:
            if miss + 1 < max_missing:
                updated_eggs.append((sx, sy, sr, scol, miss + 1))  # inkrementér
            # ellers fjernes ægget automatisk

    stable_eggs = updated_eggs

    return [(x, y, r, col) for (x, y, r, col, _) in stable_eggs]

def stabilize_detections(current_balls, robot_px, controller: RobotController, distance_threshold=10):
    global stable_balls, ball_history

    MIN_FRAMES_FOR_STABLE = 5
    MAX_FRAMES_MISSING_AFTER_ROBOT = 12
    MAX_FRAMES_MISSING_TOTAL = 12

    if(controller.delivery_counter < 1):
        MIN_FRAMES_FOR_STABLE = 1

    # --------- Opdater historik ---------
    ball_history = deque(list(ball_history), maxlen=MIN_FRAMES_FOR_STABLE)
    ball_history.append(list(current_balls))

    # --------- Nye bolde (bliver stabile hvis set i alle frames) ---------
    candidates = {}
    for frame in ball_history:
        for (x, y, r, col) in frame:
            key = (round(x / distance_threshold),
                   round(y / distance_threshold), col)
            candidates.setdefault(key, []).append((x, y, r, col))

    for _, detections in candidates.items():
        if len(detections) >= MIN_FRAMES_FOR_STABLE:
            xs, ys, rs = zip(*[(d[0], d[1], d[2]) for d in detections])
            new_ball = (int(np.mean(xs)), int(np.mean(ys)),
                        int(np.mean(rs)), detections[0][3], 0, 0)  # miss=0, total_miss=0
            if not any(np.hypot(sb[0] - new_ball[0], sb[1] - new_ball[1]) < distance_threshold
                       and sb[3] == new_ball[3] for sb in stable_balls):
                stable_balls.append(new_ball)

    # --------- Opdater eksisterende stabile bolde ---------
    updated = []
    for (sx, sy, sr, scol, miss, total_miss) in stable_balls:
        visible = any(np.hypot(sx - cb[0], sy - cb[1]) < distance_threshold and cb[3] == scol
                      for cb in current_balls)

        if visible:
            updated.append((sx, sy, sr, scol, 0, 0))  # nulstil tællere
            continue

        # Hvis uset: tæl miss og total_miss
        new_miss = miss
        new_total_miss = total_miss + 1

        if robot_px and math.hypot(sx - robot_px[0], sy - robot_px[1]) < COLLECT_DIST_PIX:
            new_miss += 1

        # Fjern bolden hvis én af de to betingelser opfyldes
        if new_miss < MAX_FRAMES_MISSING_AFTER_ROBOT and new_total_miss < MAX_FRAMES_MISSING_TOTAL:
            updated.append((sx, sy, sr, scol, new_miss, new_total_miss))

    stable_balls = updated
    return [(x, y, r, col) for x, y, r, col, _, _ in stable_balls]

def detect_balls(frame, egg, back_marker, front_marker):
    # Konverter til LAB og split kanaler
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE på L-kanalen
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # Genopbyg og konverter til BGR → HSV
    lab_clahe = cv2.merge((l_clahe, a, b))
    frame_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    frame_normalized = np.zeros_like(frame_clahe)
    cv2.normalize(frame_clahe, frame_normalized, 0, 255, cv2.NORM_MINMAX)

    hsv = cv2.cvtColor(frame_normalized, cv2.COLOR_BGR2HSV)

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
    kernel = np.ones((3, 3), np.uint8)
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
                if back_marker and front_marker:
                    # Brug midtpunkt mellem bagende og front
                    dist_to_back = np.linalg.norm(np.array((x, y)) - np.array(back_marker))
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
            if back_marker:
                dist_to_back = np.linalg.norm(np.array((x, y)) - np.array(back_marker))
                dist_to_front = np.linalg.norm(np.array((x, y)) - np.array(front_marker))
                is_inside_robot = False

            if not is_inside_egg and not is_inside_robot:
                if is_white:
                    ball_positions.append((x, y, r, 0))
                elif is_orange:
                    ball_positions.append((x, y, r, 1))

    ball_positions = [(x, y, r, o) for (
                x, y, r, o) in ball_positions if g.FIELD_X_MIN < x < g.FIELD_X_MAX and g.FIELD_Y_MIN < y < g.FIELD_Y_MAX]
    
    return ball_positions

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()

def detect_robot(frame, target_id=42, scale=1.8):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id == target_id:
            c = corners[i][0]  # 4 hjørner i rækkefølge

            # Midtpunkt
            center_pt = np.mean(c, axis=0).astype(int)

            # Brug vektor mellem to hjørner som "retning"
            front_base = ((c[0] + c[1]) / 2).astype(float)
            back_pt = ((c[2] + c[3]) / 2).astype(float)
            direction = front_base - back_pt

            # Ekstrapoler front fremad
            front_pt = (back_pt + scale * direction).astype(int)

            # Tegn på billedet
            cv2.circle(frame, center_pt, 5, (255, 0, 0), -1)
            cv2.arrowedLine(frame, tuple(back_pt.astype(int)), tuple(front_pt), (0, 255, 0), 2)

            return tuple(front_pt), tuple(center_pt), tuple(back_pt.astype(int)), tuple(direction.astype(int))

    return None



def detect_barriers(frame, back_marker=None, ball_positions=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rød farve (HSV wraparound)
    lower_red1 = np.array([0, 50, 40])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 40])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Let udglatning og edge detection
    blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find linjer med Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=100, minLineLength=50, maxLineGap=10)

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

    if back_marker:
        rx, ry = back_marker

        filtered_barriers = []

        for ((x1, y1, x2, y2), (cx, cy)) in barriers:
            # Tjek afstand til robot
            too_close_to_robot = False
            if back_marker:
                rx, ry = back_marker
                if np.linalg.norm(np.array((cx, cy)) - np.array((rx, ry))) < 70:
                    too_close_to_robot = True

            # Tjek afstand til bolde
            too_close_to_ball = False
            if ball_positions:
                for (bx, by, _, _) in ball_positions:
                    if np.linalg.norm(np.array((cx, cy)) - np.array((bx, by))) < 30:
                        too_close_to_ball = True
                        break

            # Hvis ikke for tæt på noget, behold barrieren
            if not too_close_to_robot and not too_close_to_ball:
                filtered_barriers.append(((x1, y1, x2, y2), (cx, cy)))

        return filtered_barriers
    else:
        return barriers

def detect_cross(frame, back_marker=None, front_marker=None, ball_positions=None):    
    
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


    for line in lines:
        x1, y1, x2, y2 = line[0]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        too_close_to_robot = False
        too_close_to_ball = False

        if back_marker:
            rx, ry = back_marker
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
        cx < g.FIELD_X_MIN + 50 or
        cx > g.FIELD_X_MAX - 50 or
        cy < g.FIELD_Y_MIN + 50 or
        cy > g.FIELD_Y_MAX - 50
        )

        if not too_close_to_robot and not too_close_to_ball and not too_close_to_field:
            cross_lines.append((x1, y1, x2, y2))

    # Debug mask
    cv2.imshow("Cross Mask", mask)
    cv2.imshow("Cross Edges", edges)

    return cross_lines  # Liste af linjer


def detect_egg(frame, back_marker, front_marker):
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
            if back_marker:
                dist_to_back = np.linalg.norm(np.array((x, y)) - np.array(back_marker))
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
