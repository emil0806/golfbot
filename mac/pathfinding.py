from collections import deque
import math
import numpy as np
import cv2
import globals_config as g


# =============== 3-D KORREKTION ====================
# Kamera & bane data  (mm)  – ret kun disse tal hvis noget ændrer sig
FIELD_W, FIELD_H = 1800.0, 1200.0
CAMERA_HEIGHT = 1510.0       # højde over gulv
ROBOT_MARKER_HEIGHT = 145.0        # højde over gulv

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
    return (float(XY1[0, 0]/XY1[2, 0]),
            float(XY1[1, 0]/XY1[2, 0]))


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


def find_best_ball(ball_positions, back_marker, front_marker):
    global previous_best_ball

    if not ball_positions or not back_marker:
        return None

    (rx, ry) = back_marker
    (fx, fy) = front_marker

    for i, ball in enumerate(ball_positions):
        dist = math.hypot(ball[0] - rx, ball[1] - ry)

    new_best_ball = min(ball_positions, key=lambda ball: math.hypot(
        ball[0] - fx, ball[1] - fy))
    new_dist = math.hypot(new_best_ball[0] - fx, new_best_ball[1] - fy)

    if previous_best_ball:
        old_dist = math.hypot(
            previous_best_ball[0] - fx, previous_best_ball[1] - fy)

        if new_dist > old_dist * 0.95:
            return previous_best_ball

    previous_best_ball = new_best_ball
    return new_best_ball


def determine_direction(robot_info, ball_position):
    if not robot_info or not ball_position:
        return "stop"

    front_marker, center_marker, back_marker, _ = robot_info
    # ------- 1. pixel → world (gulvplan) ----------
    bx, by = pix2world(ball_position[:2])

    (rx_p, ry_p) = back_marker
    (fx_p, fy_p) = front_marker
    rx_w, ry_w = pix2world((rx_p, ry_p))
    fx_w, fy_w = pix2world((fx_p, fy_p))

    # ------- 2. højde-korrektion -------------------
    rx, ry = _correct_marker((rx_w, ry_w))
    fx, fy = _correct_marker((fx_w, fy_w))
    # -----------------------------------------------

    vector_to_ball = (bx - rx, by - ry)
    vector_front = (fx - rx, fy - ry)

    dot = vector_front[0] * vector_to_ball[0] + \
        vector_front[1] * vector_to_ball[1]
    mag_f = math.hypot(*vector_front)
    mag_b = math.hypot(*vector_to_ball)
    cos_theta = max(-1, min(1, dot / (mag_f * mag_b)))
    angle_difference = math.degrees(math.acos(cos_theta))

    # Determine if angle is to the left or right using cross product
    cross = -(vector_front[0] * vector_to_ball[1] -
              vector_front[1] * vector_to_ball[0])

    if angle_difference < 2.5:
        if slow_down_close_to_barrier(front_marker, g.FIELD_X_MIN, g.FIELD_X_MAX, g.FIELD_Y_MIN, g.FIELD_Y_MAX):
            return "slow_forward"
        else: 
            return "forward"    
    elif cross < 0:
        if angle_difference > 25:
            return "fast_right"
        elif angle_difference > 15:
            return "right"
        else:
            return "medium_right"
    else:
        if angle_difference > 25:
            return "fast_left"
        elif angle_difference > 15:
            return "left"
        else:
            return "medium_left"


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


def is_corner_ball(ball, field_bounds, margin=150):
    x, y, _, _ = ball
    x_min, x_max, y_min, y_max = field_bounds

    in_top_left = (x < x_min + margin and y < y_min + margin)
    in_top_right = (x > x_max - margin and y < y_min + margin)
    in_bottom_left = (x < x_min + margin and y > y_max - margin)
    in_bottom_right = (x > x_max - margin and y > y_max - margin)

    return in_top_left or in_top_right or in_bottom_left or in_bottom_right


def is_edge_ball(ball, field_bounds, margin=150):
    x, y, _, _ = ball
    x_min, x_max, y_min, y_max = field_bounds

    near_left = x_min - margin < x < x_min + \
        margin and y_min + margin < y < y_max - margin
    near_right = x_max - margin < x < x_max + \
        margin and y_min + margin < y < y_max - margin
    near_top = y_min - margin < y < y_min + \
        margin and x_min + margin < x < x_max - margin
    near_bottom = y_max - margin < y < y_max + \
        margin and x_min + margin < x < x_max - margin

    return near_left or near_right or near_top or near_bottom


def create_staging_point_edge(ball, offset_distance=200):
    x, y, r, o = ball
    x_min, x_max, y_min, y_max = g.get_field_bounds

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

    return (x, y, r, o)


def create_staging_point_corner(ball, field_bounds, offset_distance=350):
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


def barrier_blocks_path(robot, ball, eggs, crosses, robot_radius=80, threshold=40):
    # Robot front marker
    fx, fy = robot
    # Bold position
    bx, by = ball[:2]

    # Step 1) Definer forward_vector som tuple
    forward_vector = (fx - bx, fy - by)

    # Step 2) Fundament til at normalisere forward_vector
    mag = math.hypot(*forward_vector) or 1.0
    ux, uy = forward_vector[0] / mag, forward_vector[1] / mag

    # Step 3) Normal vector
    nx, ny = -uy, ux

    # Step 4) Definer offsets
    offs_x, offs_y = nx * robot_radius, ny * robot_radius

    # Step 5) Linje højre og venstre for robot
    line1 = ((bx + offs_x, by + offs_y), (fx + offs_x, fy + offs_y))
    line2 = ((bx - offs_x, by - offs_y), (fx - offs_x, fy - offs_y))

   # Hjælpefunktioner:
    def dist_to_center(px, py):
        return _point_to_segment_distance(px, py, bx, by, fx, fy)

    def dist_to_edges(px, py):
        d1 = _point_to_segment_distance(
            px, py, line1[0][0], line1[0][1], line1[1][0], line1[1][1])
        d2 = _point_to_segment_distance(
            px, py, line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        return min(d1, d2)

    # Tjek æg
    for ex, ey, er, _ in eggs:
        if dist_to_center(ex, ey) <= threshold + er:
            return True
        if dist_to_edges(ex, ey) <= threshold + er:
            return True

    # Tjek kryds
    for (x1, y1, x2, y2) in crosses:
        midx, midy = (x1 + x2) / 2, (y1 + y2) / 2
        if dist_to_center(midx, midy) <= threshold:
            return True
        if dist_to_edges(midx, midy) <= threshold:
            return True

    return False

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

def slow_down_close_to_barrier(front_marker, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX):
    fx, fy = front_marker

    if FIELD_X_MIN + 200 > fx:
        return True
    if FIELD_X_MAX - 200 < fx:
        return True
    if FIELD_Y_MIN + 200 > fy:
        return True
    if FIELD_Y_MAX - 200 < fy:
        return True
    return False

def determine_robot_quadrant(center_robot, cross_center):
    fx, fy = center_robot
    cx, cy = cross_center

    if fx < cx and fy < cy:
        return 1
    elif fx >= cx and fy < cy:
        return 2
    elif fx < cx and fy >= cy:
        return 3
    elif fx >= cx and fy >= cy:
        return 4
    else:
        return 5


def determine_ball_quadrant(best_ball, cross_center):
    bx, by = best_ball[:2]
    cx, cy = cross_center

    if bx < cx and by < cy:
        return 1
    elif bx >= cx and by < cy:
        return 2
    elif bx < cx and by >= cy:
        return 3
    elif bx >= cx and by >= cy:
        return 4
    else:
        return 5


def determine_staging_point(center_robot, best_ball, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX, CROSS_CENTER):
    fx, fy = center_robot
    bx, by = best_ball[:2]

    x_25 = ((FIELD_X_MAX - FIELD_X_MIN) * 0.15) + FIELD_X_MIN
    x_50 = ((FIELD_X_MAX - FIELD_X_MIN) * 0.5) + FIELD_X_MIN
    x_75 = ((FIELD_X_MAX - FIELD_X_MIN) * 0.85) + FIELD_X_MIN
    y_25 = ((FIELD_Y_MAX - FIELD_Y_MIN) * 0.15) + FIELD_Y_MIN
    y_50 = ((FIELD_Y_MAX - FIELD_Y_MIN) * 0.5) + FIELD_Y_MIN
    y_75 = ((FIELD_Y_MAX - FIELD_Y_MIN) * 0.85) + FIELD_Y_MIN

    robot_quadrant = determine_robot_quadrant(
        center_robot, CROSS_CENTER)
    ball_quadrant = determine_ball_quadrant(
        best_ball, CROSS_CENTER)
    print(f"robot_q: {robot_quadrant}")
    print(f"ball_q: {ball_quadrant}")
    if (robot_quadrant == ball_quadrant):
        return center_robot
    elif ((robot_quadrant == 1 and ball_quadrant == 2) or (robot_quadrant == 2 and ball_quadrant == 1)):
        return (x_50, y_25)
    elif ((robot_quadrant == 1 and ball_quadrant == 3) or (robot_quadrant == 3 and ball_quadrant == 1)):
        return (x_25, y_50)
    elif ((robot_quadrant == 1 and ball_quadrant == 4) or (robot_quadrant == 4 and ball_quadrant == 1)):
        return (fx, by)
    elif ((robot_quadrant == 2 and ball_quadrant == 3) or (robot_quadrant == 3 and ball_quadrant == 2)):
        return (bx, fy)
    elif ((robot_quadrant == 2 and ball_quadrant == 4) or (robot_quadrant == 4 and ball_quadrant == 2)):
        return (x_75, y_50)
    elif ((robot_quadrant == 3 and ball_quadrant == 4) or (robot_quadrant == 4 and ball_quadrant == 3)):
        return (x_50, y_75)

def is_ball_in_cross(best_ball, CROSS_X_MIN, CROSS_X_MAX, CROSS_Y_MIN, CROSS_Y_MAX):
    bx, by = best_ball[:2]
    if (bx >= CROSS_X_MIN and bx <= CROSS_X_MAX and by >= CROSS_Y_MIN and by <= CROSS_Y_MAX):
        return True
    else:
        return False


def is_ball_and_robot_on_line_with_cross(center_robot, best_ball, CROSS_X_MIN, CROSS_X_MAX, CROSS_Y_MIN, CROSS_Y_MAX, CROSS_CENTER, margin=150):
    fx, fy = center_robot
    bx, by = best_ball[:2]
    if is_ball_and_robot_in_same_quadrant(center_robot, best_ball, CROSS_CENTER):
        return 0
    if (((fx >= CROSS_X_MIN - margin) and (fx <= CROSS_X_MAX + margin)) and ((bx >= CROSS_X_MIN - margin) and (bx <= CROSS_X_MAX + margin))):
        if(bx <= CROSS_CENTER[0]):
            return 1
        elif(bx >= CROSS_CENTER[0]):
            return 3
    elif (((fy >= CROSS_Y_MIN - margin) and (fy <= CROSS_Y_MAX + margin)) and ((by >= CROSS_Y_MIN - margin) and (by <= CROSS_Y_MAX + margin))):
        if(by <= CROSS_CENTER[1]):
            return 2
        elif(by >= CROSS_CENTER[1]):
            return 4
    elif ((fx >= CROSS_X_MIN - margin) and (fx <= CROSS_X_MAX + margin)):
        if(bx <= CROSS_CENTER[0]):
            return 1
        elif(bx >= CROSS_CENTER[0]):
            return 3
    elif ((fy >= CROSS_Y_MIN - margin) and (fy <= CROSS_Y_MAX + margin)):
        if(by <= CROSS_CENTER[1]):
            return 2
        elif(by >= CROSS_CENTER[1]):
            return 4
    elif ((bx >= CROSS_X_MIN - margin) and (bx <= CROSS_X_MAX + margin)):
        if(fx <= CROSS_CENTER[0]):
            return 1
        elif(fx >= CROSS_CENTER[0]):
            return 3
    elif ((by >= CROSS_Y_MIN - margin) and (by <= CROSS_Y_MAX + margin)):
        if(fy <= CROSS_CENTER[1]):
            return 2
        elif(fy >= CROSS_CENTER[1]):
            return 4 
    else:
        return 5

def is_ball_and_robot_in_same_quadrant(front_marker, best_ball, CROSS_CENTER):
    ball_q = determine_ball_quadrant(best_ball, CROSS_CENTER)
    robot_q = determine_robot_quadrant(front_marker, CROSS_CENTER)

    if ball_q == robot_q:
        return True
    else:
        return False

def draw_lines(robot, ball, eggs, crosses, robot_radius=80, threshold=60):
    # Robot front marker
    fx, fy = robot
    # Bold position
    bx, by = ball[:2]

    # Step 1) Definer forward_vector som tuple
    forward_vector = (fx - bx, fy - by)

    # Step 2) Fundament til at normalisere forward_vector
    mag = math.hypot(*forward_vector) or 1.0
    ux, uy = forward_vector[0] / mag, forward_vector[1] / mag

    # Step 3) Normal vector
    nx, ny = -uy, ux

    # Step 4) Definer offsets
    offs_x, offs_y = nx * robot_radius, ny * robot_radius

    # Step 5) Linje højre og venstre for robot
    line1 = ((bx + offs_x, by + offs_y), (fx + offs_x, fy + offs_y))
    line2 = ((bx - offs_x, by - offs_y), (fx - offs_x, fy - offs_y))

    return line1, line2


def get_cross_zones():
    zone_width = (g.FIELD_X_MAX - g.FIELD_X_MIN) / 7
    zone_height = (g.FIELD_Y_MAX - g.FIELD_Y_MIN) / 7

    cross_zones = set()
    for row in range(7):
        for col in range(7):
            zx_min = g.FIELD_X_MIN + col * zone_width
            zx_max = zx_min + zone_width
            zy_min = g.FIELD_Y_MIN + row * zone_height
            zy_max = zy_min + zone_height

            if not (g.CROSS_X_MAX < zx_min or g.CROSS_X_MIN > zx_max or
                    g.CROSS_Y_MAX < zy_min or g.CROSS_Y_MIN > zy_max):
                cross_zones.add((row, col))

    return cross_zones

def get_zone_for_position(x, y):
    zone_width = (g.FIELD_X_MAX - g.FIELD_X_MIN) / 7
    zone_height = (g.FIELD_Y_MAX - g.FIELD_Y_MIN) / 7

    col = int((x - g.FIELD_X_MIN) / zone_width)
    row = int((y - g.FIELD_Y_MIN) / zone_height)

    col = max(0, min(7 - 1, col))
    row = max(0, min(7 - 1, row))

    return (row, col)

def zone_to_position(row, col):
    zone_width = (g.FIELD_X_MAX - g.FIELD_X_MIN) / 7
    zone_height = (g.FIELD_Y_MAX - g.FIELD_Y_MIN) / 7

    x = g.FIELD_X_MIN + col * zone_width + zone_width / 2
    y = g.FIELD_Y_MIN + row * zone_height + zone_height / 2
    return (int(x), int(y))

def get_zone_center(zone):
    row, col = zone
    zone_width = (g.FIELD_X_MAX - g.FIELD_X_MIN) / 7
    zone_height = (g.FIELD_Y_MAX - g.FIELD_Y_MIN) / 7
    x = g.FIELD_X_MIN + (col + 0.5) * zone_width
    y = g.FIELD_Y_MIN + (row + 0.5) * zone_height
    return x, y


def bfs_path(start_zone, goal_zone, forbidden_zones):
    queue = deque()
    queue.append((start_zone, [start_zone]))
    visited = set()
    visited.add(start_zone)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  

    while queue:
        current, path = queue.popleft()
        if current == goal_zone:
            return path

        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < 7 and 0 <= nc < 7:
                neighbor = (nr, nc)
                if neighbor not in visited and neighbor not in forbidden_zones:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

    return None 

def get_simplified_target(path_zones, robot_pos, egg, cross):
    for zone in reversed(path_zones[1:]):
        target_pos = zone_to_position(*zone)
        dummy_ball = (*target_pos, 10, (255, 255, 255))
        if not barrier_blocks_path(robot_pos, dummy_ball, egg, cross):
            return dummy_ball 
    next_zone = path_zones[1]
    return (*zone_to_position(*next_zone), 10, (255, 255, 255))


def get_grid_thresholds(FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX):
    x1 = FIELD_X_MIN + (FIELD_X_MAX - FIELD_X_MIN) * (1 / 7)
    x2 = FIELD_X_MIN + (FIELD_X_MAX - FIELD_X_MIN) * (2 / 7)
    x3 = FIELD_X_MIN + (FIELD_X_MAX - FIELD_X_MIN) * (3 / 7)
    x4 = FIELD_X_MIN + (FIELD_X_MAX - FIELD_X_MIN) * (4 / 7)
    x5 = FIELD_X_MIN + (FIELD_X_MAX - FIELD_X_MIN) * (5 / 7)
    x6 = FIELD_X_MIN + (FIELD_X_MAX - FIELD_X_MIN) * (6 / 7)

    y1 = FIELD_Y_MIN + (FIELD_Y_MAX - FIELD_Y_MIN) * (1 / 7)
    y2 = FIELD_Y_MIN + (FIELD_Y_MAX - FIELD_Y_MIN) * (2 / 7)
    y3 = FIELD_Y_MIN + (FIELD_Y_MAX - FIELD_Y_MIN) * (3 / 7)
    y4 = FIELD_Y_MIN + (FIELD_Y_MAX - FIELD_Y_MIN) * (4 / 7)
    y5 = FIELD_Y_MIN + (FIELD_Y_MAX - FIELD_Y_MIN) * (5 / 7)
    y6 = FIELD_Y_MIN + (FIELD_Y_MAX - FIELD_Y_MIN) * (6 / 7)

    return x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6