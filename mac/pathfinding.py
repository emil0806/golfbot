from collections import deque
import math
import numpy as np
import cv2
import globals_config as g


# =============== 3-D KORREKTION ====================
### FIELD IN MM ###
FIELD_W, FIELD_H = 1800.0, 1200.0
CAMERA_HEIGHT = 1510.0       # Hight above floor
ROBOT_MARKER_HEIGHT = 190.0  # Robot hight above floor
BALL_HEIGHT = 40.0

# Factor < 1  (≈ 0.9)
_MARKER_SCALE = (CAMERA_HEIGHT - ROBOT_MARKER_HEIGHT) / CAMERA_HEIGHT

_CAMERA_CENTER_WORLD = None


def _correct_marker(world_pt):
    cx, cy = _CAMERA_CENTER_WORLD
    dx = (world_pt[0] - cx) * _MARKER_SCALE
    dy = (world_pt[1] - cy) * _MARKER_SCALE
    return (cx + dx, cy + dy)

def _correct_ball(world_pt):
    BALL_SCALE = (CAMERA_HEIGHT - BALL_HEIGHT) / CAMERA_HEIGHT
    cx, cy = _CAMERA_CENTER_WORLD
    dx = (world_pt[0] - cx) * BALL_SCALE
    dy = (world_pt[1] - cy) * BALL_SCALE
    return (cx + dx, cy + dy)

H = None


def set_homography(H_matrix, image_width=640, image_height=480):
    global H, _CAMERA_CENTER_WORLD
    H = H_matrix

    # Find kameracentrum i pixel og konverter til world-koordinater
    image_center_px = (image_width / 2, image_height / 2)
    _CAMERA_CENTER_WORLD = pix2world(image_center_px)

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



def determine_direction(robot_info, ball_position, crosses=None):
    if not robot_info or not ball_position:
        return "stop"

    front_marker, center_marker, back_marker, _ = robot_info
    crosses = crosses or []

    bx_raw, by_raw = pix2world(ball_position[:2])
    bx, by = _correct_ball((bx_raw, by_raw))
    (rx_p, ry_p), (fx_p, fy_p) = back_marker, front_marker
    rx_w, ry_w = pix2world((rx_p, ry_p))
    fx_w, fy_w = pix2world((fx_p, fy_p))
    rx, ry = _correct_marker((rx_w, ry_w))
    fx, fy = _correct_marker((fx_w, fy_w))

    vector_to_ball = (bx - rx, by - ry)
    vector_front = (fx - rx, fy - ry)

    dot = vector_front[0] * vector_to_ball[0] + vector_front[1] * vector_to_ball[1]
    mag_f = math.hypot(*vector_front)
    mag_b = math.hypot(*vector_to_ball)
    cos_theta = max(-1, min(1, dot / (mag_f * mag_b)))
    angle_difference = math.degrees(math.acos(cos_theta))

    cross = -(vector_front[0] * vector_to_ball[1] - vector_front[1] * vector_to_ball[0])
    center = ((fx + rx) / 2, (fy + ry) / 2)
    rotation_risk = will_rotation_hit_cross(center, radius=130, cross_lines=crosses)

    if angle_difference < 4:
        if slow_down_close_to_barrier(front_marker, back_marker):
            return "slow_forward"
        elif close_to_barrier(front_marker, back_marker) or close_to_cross(front_marker, back_marker):
            return "slow_backward"
        else:
            return "forward"
    elif cross < 0:
        #if rotation_risk:
        #    if prefer_forward_if_safe(front_marker, back_marker, ball_position, crosses):
        #        return "forward"
        #    return "slow_backward"        
        if angle_difference > 25:
                        return "fast_right"
        elif angle_difference > 15:
            return "right"
        else:
            return "medium_right"
    else:
        #if rotation_risk:
        #    if prefer_forward_if_safe(front_marker, back_marker, ball_position, crosses):
        #        return "forward"
        #    return "slow_backward" 
        if angle_difference > 25:
            return "fast_left"
        elif angle_difference > 15:
            return "left"
        else:
            return "medium_left"
        
def will_rotation_hit_cross(center, radius, cross_lines, angle_step=30, threshold=40):
    for deg in range(0, 360, angle_step):
        rad = math.radians(deg)
        px = center[0] + radius * math.cos(rad)
        py = center[1] + radius * math.sin(rad)
        for (x1, y1, x2, y2) in cross_lines:
            d = _point_to_segment_distance(px, py, x1, y1, x2, y2)
            if d < threshold:
                return True
    return False

def prefer_forward_if_safe(front_marker, back_marker, ball_position, crosses, forward_distance=30):
    fx, fy = front_marker
    rx, ry = back_marker
    bx, by = ball_position[:2]

    direction_vec = (fx - rx, fy - ry)
    mag = math.hypot(*direction_vec) or 1.0
    ux, uy = direction_vec[0] / mag, direction_vec[1] / mag
    forward_pos = (fx + ux * forward_distance, fy + uy * forward_distance)

    dist_now = math.hypot(bx - fx, by - fy)
    dist_next = math.hypot(bx - forward_pos[0], by - forward_pos[1])

    dummy_forward = (*forward_pos, 10, (255, 255, 255))

    if not barrier_blocks_path(front_marker, dummy_forward, [], crosses):
        if dist_next < dist_now:
            return True

    return False

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


def barrier_blocks_path(center_marker, ball, egg, cross, robot_radius=80, threshold=40):
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
    for ex, ey, er, _ in egg:
        if dist_to_center(ex, ey) <= threshold + er:
            return True
        if dist_to_edges(ex, ey) <= threshold + er:
            return True

    # Tjek kryds
    for (x1, y1, x2, y2) in cross:
        midx, midy = (x1 + x2) / 2, (y1 + y2) / 2
        if dist_to_center(midx, midy) <= threshold:
            return True
        if dist_to_edges(midx, midy) <= threshold:
            return True


    return False

def close_to_cross(front_marker, back_marker, threshold=150):
    fx, fy = front_marker
    bx, by = back_marker
    fx_w, fy_w = _correct_marker(pix2world((fx, fy)))
    bx_w, by_w = _correct_marker(pix2world((bx, by)))

    dx = fx_w - bx_w
    dy = fy_w - by_w
    norm = math.hypot(dx, dy)
    if norm == 0:
        return False
    dx /= norm
    dy /= norm

    # Forlæng robotretning til en lang linje fremad
    scale = 10000
    rx1, ry1 = fx_w, fy_w
    rx2, ry2 = fx_w + dx * scale, fy_w + dy * scale
    robot_line = (rx1, ry1, rx2, ry2)

    min_dist = float("inf")
    for cross_line in g.get_cross_lines():
        intersection = find_line_intersection_from_lines(robot_line, cross_line)
        if intersection:
            ix, iy = intersection
            dist = math.hypot(ix - fx_w, iy - fy_w)
            min_dist = min(min_dist, dist)

    return min_dist < threshold

def close_to_barrier(front_marker, back_marker, threshold=150):
    fx, fy = front_marker
    bx, by = back_marker

    fx_w, fy_w = _correct_marker(pix2world((fx, fy)))
    bx_w, by_w = _correct_marker(pix2world((bx, by)))

    # Forlæng robotlinjen (front -> en langt punkt i samme retning)
    dx = fx_w - bx_w
    dy = fy_w - by_w
    norm = math.hypot(dx, dy)
    if norm == 0:
        return False
    dx /= norm
    dy /= norm

    extension = 5000  # mm – langt nok til at krydse en banegrænse
    extended_front = (fx_w + dx * extension, fy_w + dy * extension)

    robot_line = ((bx_w, by_w), extended_front)
    closest_dist = None

    for edge_line in g.FIELD_LINES:
        intersection = find_line_intersection_from_lines(robot_line, edge_line)
        if intersection:
            ix, iy = intersection

            # Er skæringspunktet foran robotten?
            to_point = (ix - bx_w, iy - by_w)
            forward = (fx_w - bx_w, fy_w - by_w)
            dot = to_point[0]*forward[0] + to_point[1]*forward[1]

            if dot > 0:
                dist = math.hypot(ix - fx_w, iy - fy_w)
                if closest_dist is None or dist < closest_dist:
                    closest_dist = dist

    return closest_dist is not None and closest_dist < threshold

def slow_down_close_to_barrier(front_marker, back_marker, threshold=200):
    fx, fy = front_marker
    bx, by = back_marker

    fx_w, fy_w = _correct_marker(pix2world((fx, fy)))
    bx_w, by_w = _correct_marker(pix2world((bx, by)))

    # Forlæng robotlinjen (front -> en langt punkt i samme retning)
    dx = fx_w - bx_w
    dy = fy_w - by_w
    norm = math.hypot(dx, dy)
    if norm == 0:
        return False
    dx /= norm
    dy /= norm

    extension = 5000  # mm – langt nok til at krydse en banegrænse
    extended_front = (fx_w + dx * extension, fy_w + dy * extension)

    robot_line = ((bx_w, by_w), extended_front)
    closest_dist = None

    for edge_line in g.FIELD_LINES:
        intersection = find_line_intersection_from_lines(robot_line, edge_line)
        if intersection:
            ix, iy = intersection

            # Er skæringspunktet foran robotten?
            to_point = (ix - bx_w, iy - by_w)
            forward = (fx_w - bx_w, fy_w - by_w)
            dot = to_point[0]*forward[0] + to_point[1]*forward[1]

            if dot > 0:
                dist = math.hypot(ix - fx_w, iy - fy_w)
                if closest_dist is None or dist < closest_dist:
                    closest_dist = dist

    return closest_dist is not None and closest_dist < threshold
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

def bfs_path(start_zone, goal_zone, eggs, crosses, ball_position=None):
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
                if neighbor not in visited:

                    if neighbor == goal_zone and ball_position is not None:
                        dummy_ball = (*ball_position, 10, (255, 255, 255))
                        start_pos = zone_to_position(*current)
                        if barrier_blocks_path(start_pos, dummy_ball, eggs, crosses):
                            continue
                    else:
                        if zone_path_is_blocked(current, neighbor, eggs, crosses):
                            continue

                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

    return None


def get_simplified_path(path_zones, center_marker, ball_pos, eggs, crosses):

    simplified_path = []
    current_pos = center_marker

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
            if len(simplified_path) > 0:
                simplified_path.pop()
            simplified_path.append(target_pos)
            current_pos = target_pos
            i += 1

    return simplified_path

def zone_path_is_blocked(zone1, zone2, eggs, crosses):
    target_pos = zone_to_position(*zone2)
    dummy_ball = (*target_pos, 10, (255, 255, 255))

    center_pos = zone_to_position(*zone1)

    return barrier_blocks_path(center_pos, dummy_ball, eggs, crosses, robot_radius=80, threshold=50)


def get_grid_thresholds():
    x1 = g.FIELD_X_MIN + (g.FIELD_X_MAX - g.FIELD_X_MIN) * (1 / 7)
    x2 = g.FIELD_X_MIN + (g.FIELD_X_MAX - g.FIELD_X_MIN) * (2 / 7)
    x3 = g.FIELD_X_MIN + (g.FIELD_X_MAX - g.FIELD_X_MIN) * (3 / 7)
    x4 = g.FIELD_X_MIN + (g.FIELD_X_MAX - g.FIELD_X_MIN) * (4 / 7)
    x5 = g.FIELD_X_MIN + (g.FIELD_X_MAX - g.FIELD_X_MIN) * (5 / 7)
    x6 = g.FIELD_X_MIN + (g.FIELD_X_MAX - g.FIELD_X_MIN) * (6 / 7)

    y1 = g.FIELD_Y_MIN + (g.FIELD_Y_MAX - g.FIELD_Y_MIN) * (1 / 7)
    y2 = g.FIELD_Y_MIN + (g.FIELD_Y_MAX - g.FIELD_Y_MIN) * (2 / 7)
    y3 = g.FIELD_Y_MIN + (g.FIELD_Y_MAX - g.FIELD_Y_MIN) * (3 / 7)
    y4 = g.FIELD_Y_MIN + (g.FIELD_Y_MAX - g.FIELD_Y_MIN) * (4 / 7)
    y5 = g.FIELD_Y_MIN + (g.FIELD_Y_MAX - g.FIELD_Y_MIN) * (5 / 7)
    y6 = g.FIELD_Y_MIN + (g.FIELD_Y_MAX - g.FIELD_Y_MIN) * (6 / 7)

    return x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6

def is_ball_in_cross(best_ball):
    bx, by = best_ball[:2]
    if (bx >= g.CROSS_X_MIN and bx <= g.CROSS_X_MAX and by >= g.CROSS_Y_MIN and by <= g.CROSS_Y_MAX):
        return True
    else:
        return False
    
def find_line_intersection_from_lines(line1, line2):

    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Parallelle linjer

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - 
          (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - 
          (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    return (px, py)