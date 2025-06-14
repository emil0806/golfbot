import cv2
import numpy as np
import time

# 2D pos + (meget simpel) hastigheds-model  [x, y, vx, vy]^T
def _make_kf():
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix  = np.array([[1,0,1,0],
                                     [0,1,0,1],
                                     [0,0,1,0],
                                     [0,0,0,1]], np.float32)
    kf.measurementMatrix = np.eye(2,4, dtype=np.float32)
    kf.processNoiseCov   = np.eye(4, dtype=np.float32)*1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32)*1e-1
    kf.errorCovPost      = np.eye(4, dtype=np.float32)
    return kf

front_kf, back_kf = _make_kf(), _make_kf()
last_front = last_back = None
last_seen  = time.time()
MAX_MISS   = 0.7       # sek. begge markører må mangle


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
    global last_front, last_back, last_seen

    # ---------- (1) preprocessing ----------
    lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab); l = cv2.createCLAHE(2,(8,8)).apply(l)
    frame_clahe = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

    # blød rumlig udglatning → mindre støj / skarpere kanter i HSV
    hsv   = cv2.cvtColor(cv2.bilateralFilter(frame_clahe,9,75,75), cv2.COLOR_BGR2HSV)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(9,9),2)

    # ---------- (2) HSV-masker ----------
    mask_front = (
        cv2.inRange(hsv, ( 79,  24,  16), (100,171,200)) |   # range 1
        cv2.inRange(hsv, ( 79,  10,  10), (100,180,120))     # range 2
    )
    mask_back = (
        cv2.inRange(hsv, (155,100,120), (165,255,255)) |     # range 1
        cv2.inRange(hsv, (160,180, 60), (170,255,150))       # range 2
    )

    # ---------- (3) kontur-baseret søgning ----------
    def largest_circle(mask,label,col):
        cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        (x,y),R = cv2.minEnclosingCircle(max(cnts,key=cv2.contourArea))
        if 20 < R < 40:                                       # radius-gate
            cv2.circle(frame,(int(x),int(y)),int(R),col,2)
            cv2.putText(frame,label,(int(x)-20,int(y)-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,2)
            return (int(x),int(y))
        return None

    front = largest_circle(mask_front,"Front",(0,255,0))
    back  = largest_circle(mask_back ,"Back" ,(0,0,255))

    # ---------- (4) HoughCircles-fallback med ROI-HSV-check ----------
    if front is None or back is None:
        circles = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,1.2,30,
                                   param1=100,param2=25,minRadius=30,maxRadius=60)
        if circles is not None:
            for (x,y,R) in np.round(circles[0]).astype(int):
                # gennemsnitlig HSV i cirkel-ROI
                roi = hsv[max(y-R,0):y+R, max(x-R,0):x+R]
                if roi.size == 0: continue
                h,s,v = np.mean(roi.reshape(-1,3), axis=0)

                if front is None and 79 <= h <= 100 and 25 <= s <= 180:
                    front = (x,y)
                elif back is None  and 155<= h <= 170 and s >=100:
                    back  = (x,y)

    # ---------- (5) Kalman → predict & correct ----------
    t_now = time.time()

    def _step(kf, meas, last):
        if meas is not None:
            kf.correct(np.float32([[meas[0]],[meas[1]]]))
            last = meas
        pred = kf.predict()
        return (int(pred[0]),int(pred[1])) if meas is None else meas, last

    front, last_front = _step(front_kf, front, last_front)
    back , last_back  = _step(back_kf , back , last_back)

    # ---------- (6) sanity-check & timeout ----------
    if front and back:
        d = np.linalg.norm(np.subtract(front,back))
        if 100 <= d <= 150:           # gyldig afstand
            last_seen = t_now
        else:                         # åbenlys fejl
            front = back = None

    # drop helt hvis begge mangler for længe
    if front is None and back is None and (t_now-last_seen) > MAX_MISS:
        return None

    # brug sidste kendte hvis kun én mangler
    front = front or last_front
    back  = back  or last_back
    if front is None or back is None:
        return None

    # ---------- (7) tegn & return ----------
    cv2.arrowedLine(frame, back, front, (255,255,255), 2)
    cv2.imshow("Robot Debug", frame)
    cv2.imshow("Green Mask", mask_front)          # hurtig fin-tuning
    direction_vec = (front[0]-back[0], front[1]-back[1])
    return back, front, direction_vec


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

    if lines is not None:            #TILFØJET FOR AT KØRE KAMERA HJEMME !!!DAVID!!!
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
    for ((x1, y1, x2, y2), _) in barriers:
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    if not xs or not ys:                                              #TILFØJET FOR AT KØRE KAMERA HJEMME !!!DAVID!!!
        print("⚠️ Advarsel: Ingen barriers fundet – bruger fallback") #TILFØJET FOR AT KØRE KAMERA HJEMME !!!DAVID!!!
        return 0, 1280, 0, 720                                        #TILFØJET FOR AT KØRE KAMERA HJEMME !!!DAVID!!!

    FIELD_X_MIN, FIELD_X_MAX = min(xs), max(xs)
    FIELD_Y_MIN, FIELD_Y_MAX = min(ys), max(ys)
    return FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX
