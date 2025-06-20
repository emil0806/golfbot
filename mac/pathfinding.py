from collections import deque
import math
import numpy as np
import cv2
import globals_config as g


# =============== 3-D KORREKTION ====================
### FIELD IN MM ###
FIELD_W, FIELD_H = 1800.0, 1200.0
CAMERA_HEIGHT = 1510.0       # Hight above floor
ROBOT_MARKER_HEIGHT = 145.0  # Robot hight above floor

# Factor < 1  (≈ 0.9)
_MARKER_SCALE = (CAMERA_HEIGHT - ROBOT_MARKER_HEIGHT) / CAMERA_HEIGHT

_CAMERA_CENTER_WORLD = (FIELD_W / 2.0, FIELD_H / 2.0)


def _correct_marker(world_pt):
    cx, cy = _CAMERA_CENTER_WORLD
    dx = (world_pt[0] - cx) * _MARKER_SCALE
    dy = (world_pt[1] - cy) * _MARKER_SCALE
    return (cx + dx, cy + dy)

H = None


def set_homography(H_matrix):
    global H
    H = H_matrix

def pix2world(pt):
    if H is None:
        return pt       
    x, y = pt
    uv1 = np.array([[x, y, 1.]], dtype=float).T   
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

def determine_direction(robot_info, ball_position):
    if not robot_info or not ball_position:
        return "stop"

    front_marker, center_marker, back_marker, _ = robot_info
    ### ----- 1. pixel → world (floor plan) -----
    bx, by = pix2world(ball_position[:2])

    (rx_p, ry_p) = back_marker
    (fx_p, fy_p) = front_marker
    rx_w, ry_w = pix2world((rx_p, ry_p))
    fx_w, fy_w = pix2world((fx_p, fy_p))

    ### ----- 2. HIGHT-correction -----
    rx, ry = _correct_marker((rx_w, ry_w))
    fx, fy = _correct_marker((fx_w, fy_w))

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
        if slow_down_close_to_barrier(front_marker, back_marker):
            return "slow_forward"
        elif close_to_barrier(front_marker):
            return "slow_backward"
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

    if x <= px <= x + w and y <= py <= y + h:
        distances = [
            abs(px - x),           
            abs(px - (x + w)),     
            abs(py - y),           
            abs(py - (y + h))      
        ]
        return min(distances)

    dx = max(x - px, 0, px - (x + w))
    dy = max(y - py, 0, py - (y + h))
    return math.hypot(dx, dy)

def is_corner_ball(ball, margin=150):
    x, y, _, _ = ball
    x_min, x_max, y_min, y_max = g.get_field_bounds()

    in_top_left = (x < x_min + margin and y < y_min + margin)
    in_top_right = (x > x_max - margin and y < y_min + margin)
    in_bottom_left = (x < x_min + margin and y > y_max - margin)
    in_bottom_right = (x > x_max - margin and y > y_max - margin)

    return in_top_left or in_top_right or in_bottom_left or in_bottom_right

def is_edge_ball(ball, margin=150):
    x, y, _, _ = ball
    x_min, x_max, y_min, y_max = g.get_field_bounds()

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
    x_min, x_max, y_min, y_max = g.get_field_bounds()

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


def barrier_blocks_path(center_marker, ball, eggs, crosses, robot_radius=80, threshold=40):
    # Robot front marker
    if len(center_marker) == 2:
        cx, cy = center_marker
    else:
        cx, cy = center_marker[:2]

    # Bold position
    bx, by = ball[:2]

    # Step 1) Definer forward_vector som tuple
    forward_vector = (cx - bx, cy - by)

    # Step 2) Fundament til at normalisere forward_vector
    mag = math.hypot(*forward_vector) or 1.0
    ux, uy = forward_vector[0] / mag, forward_vector[1] / mag

    # Step 3) Normal vector
    nx, ny = -uy, ux

    # Step 4) Definer offsets
    offs_x, offs_y = nx * robot_radius, ny * robot_radius

    # Step 5) Linje højre og venstre for robot
    line1 = ((bx + offs_x, by + offs_y), (cx + offs_x, cy + offs_y))
    line2 = ((bx - offs_x, by - offs_y), (cx - offs_x, cy - offs_y))

   # Hjælpefunktioner:
    def dist_to_center(px, py):
        return _point_to_segment_distance(px, py, bx, by, cx, cy)

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

def close_to_barrier(front_marker):

    if g.FIELD_X_MIN + 150 > front_marker[0]:
        return True
    if g.FIELD_X_MAX - 150 < front_marker[0]:
        return True
    if g.FIELD_Y_MIN + 150 > front_marker[1]:
        return True
    if g.FIELD_Y_MAX - 150 < front_marker[1]:
        return True
    return False

def slow_down_close_to_barrier(front_marker, back_marker, threshold=300):
    fx, fy = front_marker
    bx, by = back_marker
    fx_w, fy_w = _correct_marker(pix2world((fx, fy)))
    bx_w, by_w = _correct_marker(pix2world((bx, by)))

    # Retningsvektor fra back til front i world-koordinater
    dx = fx_w - bx_w
    dy = fy_w - by_w
    norm = math.hypot(dx, dy)
    if norm == 0:
        return False
    dx /= norm
    dy /= norm

    # Projektion fremad mod nærmeste væg
    max_dist = 9999
    end_x, end_y = fx_w, fy_w

    if dx > 0:
        dist_x = (g.FIELD_X_MAX - end_x) / dx
    elif dx < 0:
        dist_x = (g.FIELD_X_MIN - end_x) / dx
    else:
        dist_x = max_dist

    if dy > 0:
        dist_y = (g.FIELD_Y_MAX - end_y) / dy
    elif dy < 0:
        dist_y = (g.FIELD_Y_MIN - end_y) / dy
    else:
        dist_y = max_dist

    # Mindste afstand før vi rammer en væg
    travel_dist = min(dist_x, dist_y)

    return travel_dist < threshold

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

def get_simplified_path(path_zones, center_marker, ball_pos, eggs, crosses):

    simplified_path = []
    current_pos = center_marker
    simplified_path.append((int(current_pos[0]), int(current_pos[1])))

    i = 0
    while i < len(path_zones):
        found = False
        for j in range(len(path_zones) - 1, i, -1):
            zone = path_zones[j]
            target_pos = zone_to_position(*zone)
            dummy_target = (*target_pos, 10, (255, 255, 255))

            if not barrier_blocks_path(current_pos, dummy_target, eggs, crosses):
                simplified_path.append(target_pos)
                current_pos = target_pos
                i = j
                found = True
                break

        if not found:
            zone = path_zones[i]
            target_pos = ball_pos
            simplified_path.pop()
            simplified_path.append(target_pos)
            current_pos = target_pos
            i += 1

    return simplified_path

def get_simplified_target(path_zones, center_marker, egg, cross):
    if len(path_zones) <= 1:
        return (*zone_to_position(*path_zones[-1]), 10, (255, 255, 255))

    for zone in reversed(path_zones[1:]):
        target_pos = zone_to_position(*zone)
        dummy_ball = (*target_pos, 10, (255, 255, 255))
        if not barrier_blocks_path(center_marker, dummy_ball, egg, cross):
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