import cv2
import numpy as np

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

            is_inside_egg = any(np.linalg.norm(np.array((x, y)) - np.array((ex, ey))) < er for (ex, ey, er, _) in egg)
            is_inside_robot = False
            if robot_position:
                dist_to_back = np.linalg.norm(np.array((x, y)) - np.array(robot_position))
                dist_to_front = np.linalg.norm(np.array((x, y)) - np.array(front_marker))
                is_inside_robot = dist_to_back < 80 or dist_to_front < 80

            if not is_inside_egg and not is_inside_robot:
                if is_white:
                    ball_positions.append((x, y, r, 0))
                elif is_orange:
                    ball_positions.append((x, y, r, 1))

    return ball_positions

def detect_robot(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    front_marker = None
    back_marker = None  

    # Grøn front marker – tilpasset lav saturation og V:
    lower_green = np.array([30, 40, 100])
    upper_green = np.array([100, 100, 255])  # lidt over dine højeste værdier

    # New back marker (H: 161–165, S: 100+, V: 200+)
    lower_back = np.array([140, 50, 150])
    upper_back = np.array([170, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_back = cv2.inRange(hsv, lower_back, upper_back)


    def find_largest_circle(mask, label, color):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius > 20:
            center = (int(x), int(y))
            cv2.circle(frame, center, int(radius), color, 2)
            cv2.putText(frame, label, (center[0] - 20, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            return center
        return None

    front_marker = find_largest_circle(mask_green, "Front", (0, 255, 0))
    back_marker = find_largest_circle(mask_back, "Back", (0, 0, 255))

    # --- Fallback: HoughCircles if markers not found ---
    if front_marker is None or back_marker is None:
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                                   param1=100, param2=25, minRadius=20, maxRadius=100)

        if circles is not None and (front_marker is None or back_marker is None):
            biggest = sorted(circles[0, :], key=lambda c: c[2], reverse=True)[:2]
            candidates = []
            
            for (x, y, r) in biggest:
                x, y, r = int(x), int(y), int(r)
                roi = hsv[max(y - r, 0):min(y + r, hsv.shape[0]),
                        max(x - r, 0):min(x + r, hsv.shape[1])]
                if roi.size == 0:
                    continue
                avg_h, avg_s, avg_v = np.mean(roi, axis=(0, 1))
                candidates.append(((x, y), r, avg_h, avg_s, avg_v))

            for (center, r, h, s, v) in candidates:
                if front_marker is None and 20 <= h <= 80 and 25 <= s <= 100 and 150 <= v <= 255:
                    front_marker = center
                    cv2.circle(frame, center, r, (0, 255, 0), 2)
                    cv2.putText(frame, "Front", (center[0]-20, center[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif back_marker is None and 120 <= h <= 190 and s > 70 and v > 160:
                    back_marker = center
                    cv2.circle(frame, center, r, (255, 0, 255), 2)
                    cv2.putText(frame, "Back", (center[0]-20, center[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # If HSV check fails, assign based on position (e.g. left vs. right or top vs. bottom)
            if len(candidates) == 2 and (front_marker is None or back_marker is None):
                (c1, r1, _, _, _), (c2, r2, _, _, _) = candidates
                # For example: assume the marker with lower y is front
                if c1[1] < c2[1]:
                    front_marker = front_marker or c1
                    back_marker = back_marker or c2
                else:
                    front_marker = front_marker or c2
                    back_marker = back_marker or c1


    # --- Return result if both are found ---
    if front_marker and back_marker:
        cv2.arrowedLine(frame, back_marker, front_marker, (255, 255, 255), 2)
        direction_vector = (front_marker[0] - back_marker[0], front_marker[1] - back_marker[1])
        cv2.imshow("Robot Debug", frame)
        return (back_marker, front_marker, direction_vector)

    cv2.imshow("Green Mask", mask_green)

    cv2.imshow("Robot Debug", frame)
    return None


def detect_barriers(frame, robot_position=None, ball_positions=None):
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
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=100, minLineLength=100, maxLineGap=10)

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
            # Tjek afstand til robot
            too_close_to_robot = False
            if robot_position:
                rx, ry = robot_position
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


def detect_cross(frame, robot_position=None, front_marker=None, ball_positions=None, barriers=None):
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

        if robot_position:
            rx, ry = robot_position
            if np.linalg.norm(np.array((cx, cy)) - np.array((rx, ry))) < 100:
                too_close_to_robot = True

        if front_marker:
            fx, fy = front_marker
            if np.linalg.norm(np.array((cx, cy)) - np.array((fx, fy))) < 100:
                too_close_to_robot = True

        if ball_positions:
            for (bx, by, _, _) in ball_positions:
                if np.linalg.norm(np.array((cx, cy)) - np.array((bx, by))) < 30:
                    too_close_to_ball = True
                    break

        if not too_close_to_robot and not too_close_to_ball:
            cross_lines.append((x1, y1, x2, y2))

    if barriers:
        filtered = []
        for (x1, y1, x2, y2) in cross_lines:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            too_close_to_barrier = any(
                np.linalg.norm(np.array((cx, cy)) - np.array(b_center)) < 40
                for (_, b_center) in barriers
            )
            if not too_close_to_barrier:
                filtered.append((x1, y1, x2, y2))
        cross_lines = filtered

    # Debug mask
    cv2.imshow("Cross Mask", mask)
    cv2.imshow("Cross Edges", edges)

    return cross_lines  # Liste af linjer


def detect_egg(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 80, 255])

    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    contours_white, _ = cv2.findContours(
        mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    egg = []

    for cnt in contours_white:
        (x, y), radius = cv2.minEnclosingCircle(cnt)

        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-5))

        if 0.8 < circularity and radius > 20:
            egg.append((int(x), int(y), int(radius), 0))

    return egg


def inside_field(barriers):
    xs, ys = [], []
    for (x1, y1, x2, y2), _ in barriers:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    FIELD_X_MIN, FIELD_X_MAX = min(xs), max(xs)
    FIELD_Y_MIN, FIELD_Y_MAX = min(ys), max(ys)

    return (FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX)
