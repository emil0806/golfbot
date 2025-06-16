import math
import numpy as np, cv2



# =============== 3-D KORREKTION ====================
# Kamera & bane data  (mm)  – ret kun disse tal hvis noget ændrer sig
FIELD_W, FIELD_H        = 1800.0, 1200.0
CAMERA_HEIGHT           = 1480.0       # højde over gulv
ROBOT_MARKER_HEIGHT     = 140.0        # højde over gulv

# Faktor < 1  (≈ 0.9)  flytter markøren ind mod nadir,
# så vi får dens projektion ned på gulvplanet.
_MARKER_SCALE = (CAMERA_HEIGHT - ROBOT_MARKER_HEIGHT) / CAMERA_HEIGHT

# Kameraets nadir (lodrette nedkast) antages midt på banen
_CAMERA_CENTER_WORLD = (FIELD_W / 2.0, FIELD_H / 2.0)

def _correct_marker(world_pt):
    """
    Flyt et markør-punkt (som er 150 mm højere end gulvet)
    udad langs linjen fra kameraets nadir, så det rammer gulvplanet.
    """
    cx, cy = _CAMERA_CENTER_WORLD
    dx = (world_pt[0] - cx) * _MARKER_SCALE
    dy = (world_pt[1] - cy) * _MARKER_SCALE
    return (cx + dx, cy + dy)
# ==================================================

# ----------------  PIXEL → WORLD  -----------------
# Homografi-matrixen H sættes én gang fra mac_server.py
H = None

def set_homography(H_matrix):
    """Kalds én gang i starten – gemmer homografi-matrixen globalt."""
    global H
    H = H_matrix

def pix2world(pt):
    """Konverter (x,y) pixel → (X,Y) gulvplan via homografi."""
    if H is None:
        return pt          # fallback hvis ingen homografi endnu
    x, y = pt
    uv1 = np.array([[x, y, 1.]], dtype=float).T   # 3×1
    XY1 = H @ uv1
    return (float(XY1[0,0]/XY1[2,0]),
            float(XY1[1,0]/XY1[2,0]))

previous_best_ball = None

def sort_balls_by_distance(ball_positions, front_marker):
    if not ball_positions or not front_marker:
        return []

    fx, fy = front_marker

    sorted_balls = sorted(
        ball_positions,
        key=lambda ball: (
            0 if ball[3] == 1 else 1,
            math.hypot(ball[0] - fx, ball[1] - fy)
            )
    )

    return sorted_balls

def find_best_ball(ball_positions, robot_position, front_marker):
    global previous_best_ball

    if not ball_positions or not robot_position:
        return None

    (rx, ry) = robot_position 
    (fx, fy) = front_marker

    for i, ball in enumerate(ball_positions):
        dist = math.hypot(ball[0] - rx, ball[1] - ry)

    new_best_ball = min(ball_positions, key=lambda ball: math.hypot(ball[0] - fx, ball[1] - fy))
    new_dist = math.hypot(new_best_ball[0] - fx, new_best_ball[1] - fy)

    if previous_best_ball:
        old_dist = math.hypot(previous_best_ball[0] - fx, previous_best_ball[1] - fy)

        if new_dist > old_dist * 0.95:
            return previous_best_ball

    previous_best_ball = new_best_ball
    return new_best_ball


def determine_direction(robot_position, ball_position):
    if not robot_position or not ball_position:
        return "stop"

    # ------- 1. pixel → world (gulvplan) ----------
    bx, by   = pix2world(ball_position[:2])

    (rx_p, ry_p), (fx_p, fy_p), _ = robot_position
    rx_w, ry_w = pix2world((rx_p, ry_p))
    fx_w, fy_w = pix2world((fx_p, fy_p))

    # ------- 2. højde-korrektion -------------------
    rx, ry = _correct_marker((rx_w, ry_w))
    fx, fy = _correct_marker((fx_w, fy_w))
    # -----------------------------------------------

    vector_to_ball = (bx - rx, by - ry)
    vector_front = (fx - rx, fy - ry)   

    dot = vector_front[0] * vector_to_ball[0] + vector_front[1] * vector_to_ball[1]
    mag_f = math.hypot(*vector_front)
    mag_b = math.hypot(*vector_to_ball)
    cos_theta = max(-1, min(1, dot / (mag_f * mag_b)))
    angle_difference = math.degrees(math.acos(cos_theta))

    # Determine if angle is to the left or right using cross product
    cross = -(vector_front[0] * vector_to_ball[1] - vector_front[1] * vector_to_ball[0])

    if angle_difference < 2.5:
        return "forward"
    elif cross < 0:
        if angle_difference > 25:
            return "fast_right"
        elif angle_difference > 15:
            return "right"
        else:
            return "slow_right"
    else:
        if angle_difference > 25:
            return "fast_left"
        elif angle_difference > 15:
            return "left"
        else:
            return "slow_left"


def point_rect_distance(px, py, rect):
    x, y, w, h = rect

    # Hvis punktet er inde i rektanglen, find afstand til nærmeste kant
    if x <= px <= x + w and y <= py <= y + h:
        distances = [
            abs(px - x),           # venstre
            abs(px - (x + w)),     # højre
            abs(py - y),           # top
            abs(py - (y + h))      # bund
        ]
        return min(distances)
    
    # Ellers som normalt
    dx = max(x - px, 0, px - (x + w))
    dy = max(y - py, 0, py - (y + h))
    return math.hypot(dx, dy)


def check_barrier_proximity(point, barriers, threshold=60):
    px, py = point
    for (rect, center) in barriers:
        distance = point_rect_distance(px, py, rect)
        if distance < threshold:
            return True
    return False

def is_corner_ball(ball, field_bounds, margin=150):
    x, y, _, _ = ball
    x_min, x_max, y_min, y_max = field_bounds

    in_top_left     = (x < x_min + margin and y < y_min + margin)
    in_top_right    = (x > x_max - margin and y < y_min + margin)
    in_bottom_left  = (x < x_min + margin and y > y_max - margin)
    in_bottom_right = (x > x_max - margin and y > y_max - margin)

    return in_top_left or in_top_right or in_bottom_left or in_bottom_right


def is_edge_ball(ball, field_bounds, margin=150):
    x, y, _, _ = ball
    x_min, x_max, y_min, y_max = field_bounds

    near_left   = x_min - margin < x < x_min + margin and y_min + margin < y < y_max - margin
    near_right  = x_max - margin < x < x_max + margin and y_min + margin < y < y_max - margin
    near_top    = y_min - margin < y < y_min + margin and x_min + margin < x < x_max - margin
    near_bottom = y_max - margin < y < y_max + margin and x_min + margin < x < x_max - margin

    return near_left or near_right or near_top or near_bottom


def create_staging_point_edge(ball, field_bounds, offset_distance=200):
    x, y, r, o = ball
    x_min, x_max, y_min, y_max = field_bounds

    # Venstre kant
    if abs(x - x_min) < 150:
        return (x + offset_distance, y, r, o)
    # Højre kant
    elif abs(x - x_max) < 150:
        return (x - offset_distance, y, r, o)
    # Øverste kant
    elif abs(y - y_min) < 150:
        return (x, y + offset_distance, r, o)
    # Nederste kant
    elif abs(y - y_max) < 150:
        return (x, y - offset_distance, r, o)

    # Fallback
    return (x - offset_distance, y - offset_distance, r, o)



def create_staging_point_corner(ball, field_bounds, offset_distance=200):
    x, y, r, o = ball
    x_min, x_max, y_min, y_max = field_bounds

    # Øverste venstre hjørne
    if x < x_min + 100 and y < y_min + 100:
        return (x + offset_distance, y + offset_distance, r, o)
    # Øverste højre hjørne
    elif x > x_max - 100 and y < y_min + 100:
        return (x - offset_distance, y + offset_distance, r, o)
    # Nederste venstre hjørne
    elif x < x_min + 100 and y > y_max - 100:
        return (x + offset_distance, y - offset_distance, r, o)
    # Nederste højre hjørne
    elif x > x_max - 100 and y > y_max - 100:
        return (x - offset_distance, y - offset_distance, r, o)

    # Fallback
    return (x, y - offset_distance, r, o)



# ------------------ ÆG-UNDVIGELSE ------------------

def _point_to_segment_distance(px, py, x1, y1, x2, y2):
    if (x1, y1) == (x2, y2):
        return math.hypot(px - x1, py - y1)
    dx, dy = x2 - x1, y2 - y1
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))
    proj_x, proj_y = x1 + t*dx, y1 + t*dy
    return math.hypot(px - proj_x, py - proj_y)


def egg_blocks_path(robot_center, ball, egg, threshold=60):
    """
    Returnerer True hvis ægget ligger < threshold fra linjen
    robot_center → bold.
    """
    rx, ry = robot_center
    bx, by = ball[:2]
    ex, ey, er = egg  # er = æg-radius
    dist = _point_to_segment_distance(ex, ey, rx, ry, bx, by)
    return dist < threshold + er


def create_staging_point_egg(robot_center, ball, egg, offset_distance=200):
    """
    Laver et staging-punkt vinkelret på banen omkring ægget,
    så robotten kan køre uden om.
    """
    rx, ry = robot_center
    bx, by = ball[:2]
    ex, ey, er = egg

    # vektor robot → bold
    vx, vy = bx - rx, by - ry
    # vinkelret enhedsvektor
    perp_x, perp_y = -vy, vx
    mag = math.hypot(perp_x, perp_y) or 1.0
    perp_x, perp_y = perp_x / mag, perp_y / mag

    # staging-punkt forskudt fra ægget
    sx = ex + perp_x * (er + offset_distance)
    sy = ey + perp_y * (er + offset_distance)

    # radius 15 er fint til visualisering; farve-id bevares fra bolden
    return (int(sx), int(sy), 15, ball[3])

def barrier_blocks_path(robot, ball, eggs, crosses, threshold=60):
    fx, fy = robot
    bx, by = ball[:2]
    
    # Tjek æg
    for (ex, ey, er, _) in eggs:
        dist = _point_to_segment_distance(ex, ey, fx, fy, bx, by)
        if dist < threshold + er:
            return True

    # Tjek kryds
    for (x1, y1, x2, y2) in crosses:
        # Find midtpunkt af krydslinje
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        dist = _point_to_segment_distance(cx, cy, fx, fy, bx, by)
        if dist < threshold:
            return True

    return False


def delivery_routine(robot_info):
    # Simple placeholder routine
    # Go forward to an approach point, turn, then reverse
    return "delivery"

def stop_delivery_routine():
    return "continue"

def close_to_barrier(front_marker, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX):

    if FIELD_X_MIN + 40 + 60 > front_marker[0]:
        return True
    if FIELD_X_MAX - 40 - 60 < front_marker[0]:
        return True
    if FIELD_Y_MIN + 40 + 60 > front_marker[1]:
        return True
    if FIELD_Y_MAX - 40 - 60 < front_marker[1]:
        return True
    return False

def determine_robot_quadrant(front_marker, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX):
    fx, fy = front_marker
    mid_x = (FIELD_X_MIN + FIELD_X_MAX) / 2
    mid_y = (FIELD_Y_MIN + FIELD_Y_MAX) / 2

    if fx < mid_x and fy < mid_y:
        return 1
    elif fx >= mid_x and fy < mid_y:
        return 2
    elif fx < mid_x and fy >= mid_y:
        return 3
    elif fx >= mid_x and fy >= mid_y:
        return 4
    else:
        return 5 

def determine_ball_quadrant(best_ball, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX):
    bx, by = best_ball[:2]
    mid_x = (FIELD_X_MIN + FIELD_X_MAX) / 2
    mid_y = (FIELD_Y_MIN + FIELD_Y_MAX) / 2

    if bx < mid_x and by < mid_y:
        return 1
    elif bx >= mid_x and by < mid_y:
        return 2
    elif bx < mid_x and by >= mid_y:
        return 3
    elif bx >= mid_x and by >= mid_y:
        return 4
    else:
        return 5 

    
def determine_staging_point(front_marker, best_ball, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX):
    x_25 = (FIELD_X_MAX - FIELD_X_MIN) / 4
    x_50 = (FIELD_X_MAX - FIELD_X_MIN) / 2
    x_75 = ((FIELD_X_MAX - FIELD_X_MIN) / 4) * 3
    y_25 = (FIELD_Y_MAX - FIELD_Y_MIN) / 4
    y_50 = (FIELD_Y_MAX - FIELD_Y_MIN) / 2
    y_75 = ((FIELD_Y_MAX - FIELD_Y_MIN) / 4) * 3

    robot_quadrant = determine_robot_quadrant(front_marker, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX)
    ball_quadrant = determine_ball_quadrant(best_ball, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX)

    if(robot_quadrant == ball_quadrant):
        return front_marker
    elif((robot_quadrant == 1 and ball_quadrant == 2) or (robot_quadrant == 2 and ball_quadrant == 1)):
        return (x_50, y_25)
    elif((robot_quadrant == 1 and ball_quadrant == 3) or (robot_quadrant == 3 and ball_quadrant == 1)):
        return (x_25, y_50)
    elif((robot_quadrant == 1 and ball_quadrant == 4)or (robot_quadrant == 4 and ball_quadrant == 1)):
        return (x_75, y_25)
    elif((robot_quadrant == 2 and ball_quadrant == 3) or (robot_quadrant == 3 and ball_quadrant == 2)):
        return (x_75, y_75)
    elif((robot_quadrant == 2 and ball_quadrant == 4) or (robot_quadrant == 4 and ball_quadrant == 2)):
        return (x_75, y_50)
    elif((robot_quadrant == 3 and ball_quadrant == 4) or (robot_quadrant == 4 and ball_quadrant == 3)):
        return (x_50, y_75)
    

def is_ball_in_cross(best_ball, CROSS_X_MIN, CROSS_X_MAX, CROSS_Y_MIN, CROSS_Y_MAX):
    bx, by = best_ball[:2]
    if(bx >= CROSS_X_MIN and bx <= CROSS_X_MAX and by >= CROSS_Y_MIN and by <= CROSS_Y_MAX):
        return True
    else:
        return False

def is_ball_and_robot_on_line_with_cross(front_marker, best_ball, CROSS_X_MIN, CROSS_X_MAX, CROSS_Y_MIN, CROSS_Y_MAX, margin=100):
    fx, fy = front_marker
    bx, by = best_ball[:2]

    if((fx >= CROSS_X_MIN - margin and fx <= CROSS_X_MAX + margin) and (bx >= CROSS_X_MIN - margin and bx <= CROSS_X_MAX + margin)):
        return 1
    elif((fy >= CROSS_Y_MIN - margin and fy <= CROSS_Y_MAX + margin) and (by >= CROSS_Y_MIN - margin and by <= CROSS_Y_MAX + margin)):
        return 2
    else:
        return 0
