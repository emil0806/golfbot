from collections import deque
import math
import numpy as np
import globals_config as g


# =============== 3-D CORRECTION ====================
### FIELD IN MM ###
FIELD_W, FIELD_H = 1800.0, 1200.0
CAMERA_HEIGHT = 1510.0
ROBOT_MARKER_HEIGHT = 200.0
BALL_HEIGHT = 40.0
BARRIER_HEIGHT = 80.0


_CAMERA_CENTER_WORLD = None

# Image size (pixel)
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# Center of image
PIXEL_CX = IMAGE_WIDTH / 2
PIXEL_CY = IMAGE_HEIGHT / 2


def _correct_projection(pixel_pt, object_height):
    scale = (CAMERA_HEIGHT - object_height) / CAMERA_HEIGHT
    dx = (pixel_pt[0] - PIXEL_CX) * scale
    dy = (pixel_pt[1] - PIXEL_CY) * scale
    return (PIXEL_CX + dx, PIXEL_CY + dy)


def _correct_marker(pixel_pt):
    return _correct_projection(pixel_pt, ROBOT_MARKER_HEIGHT)

def _correct_ball(pixel_pt):
    return _correct_projection(pixel_pt, BALL_HEIGHT)

def _correct_barrier(pixel_pt):
    return _correct_projection(pixel_pt, BARRIER_HEIGHT)



def set_homography(H_matrix, image_width, image_height):
    global H, _CAMERA_CENTER_WORLD
    H = H_matrix

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
    cx, cy = g.get_cross_center()

    if not ball_positions or not front_marker:
        return []

    fx, fy = front_marker

    orange_balls = [b for b in ball_positions if b[3] == 1]
    regular_balls = [b for b in ball_positions if b[3] != 1]

    def distance_to_front(b):
        return math.hypot(b[0] - fx, b[1] - fy)

    q1 = [] 
    q2 = [] 
    q3 = [] 
    q4 = [] 

    for b in regular_balls:
        x, y = b[:2]
        if x < cx and y > cy:
            q1.append(b)
        elif x < cx and y < cy:
            q2.append(b)
        elif x > cx and y < cy:
            q3.append(b)
        elif x > cx and y > cy:
            q4.append(b)

    q1.sort(key=distance_to_front)
    q2.sort(key=distance_to_front)
    q3.sort(key=distance_to_front)
    q4.sort(key=distance_to_front)

    sorted_balls = q1 + q2 + q3 + q4 + sorted(orange_balls, key=distance_to_front)

    return sorted_balls

def determine_direction(robot_info, ball_position, egg, crosses=None):
    if not robot_info or not ball_position:
        return "stop"

    front_marker, center_marker, back_marker, _ = robot_info
    crosses = crosses or []

    bx, by = _correct_ball(ball_position[:2])
    fx, fy = _correct_marker(front_marker)
    rx, ry = _correct_marker(back_marker)


    vector_to_ball = (bx - rx, by - ry)
    vector_front = (fx - rx, fy - ry)

    dot = vector_front[0] * vector_to_ball[0] + vector_front[1] * vector_to_ball[1]
    mag_f = math.hypot(*vector_front)
    mag_b = math.hypot(*vector_to_ball)
    cos_theta = max(-1, min(1, dot / (mag_f * mag_b)))
    angle_difference = math.degrees(math.acos(cos_theta))

    cross = -(vector_front[0] * vector_to_ball[1] - vector_front[1] * vector_to_ball[0])

    if angle_difference < 4:
        if close_to_barrier(front_marker, back_marker) or close_to_cross(front_marker, back_marker) or close_to_egg(front_marker, back_marker, egg):
            return "slow_backward"
        elif slow_down_close_to_barrier(front_marker, back_marker):
            return "slow_forward"
        else:
            return "forward"
    elif cross < 0:
        if (close_to_barrier(front_marker, back_marker, threshold=120) or close_to_cross(front_marker, back_marker, threshold=150)) and angle_difference > 10:
            return "medium_backward"         
        elif angle_difference > 25:
            return "fast_right"
        elif angle_difference > 15:
            return "right"
        else:
            return "medium_right"
    else:
        if (close_to_barrier(front_marker, back_marker, threshold=120) or close_to_cross(front_marker, back_marker, threshold=150)) and angle_difference > 10:
            return "medium_backward"  
        elif angle_difference > 25:
            return "fast_left"
        elif angle_difference > 15:
            return "left"
        else:
            return "medium_left"
        

def is_corner_ball(ball, margin=150):
    x, y, _, _ = ball
    x_min, x_max, y_min, y_max = g.get_field_bounds()

    in_top_left = (x < x_min + margin and y < y_min + margin)
    in_top_right = (x > x_max - margin and y < y_min + margin)
    in_bottom_left = (x < x_min + margin and y > y_max - margin)
    in_bottom_right = (x > x_max - margin and y > y_max - margin)

    return in_top_left or in_top_right or in_bottom_left or in_bottom_right

def is_edge_ball(ball, margin=70):
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

def is_ball_near_egg(ball, eggs, threshold=100):
    bx, by, *_ = ball
    for ex, ey, er, _ in eggs:
        dist = math.hypot(bx - ex, by - ey)
        if dist < threshold + er:
            return True
    return False


def create_staging_point_edge(ball, offset_distance=180):
    x, y, r, o = ball
    x_min, x_max, y_min, y_max = g.get_field_bounds()

    # Left edge 
    if abs(x - x_min) < 150:
        return (x + offset_distance, y, r, o)
    # Right edge
    elif abs(x - x_max) < 150:
        return (x - offset_distance, y, r, o)
    # Top edge
    elif abs(y - y_min) < 150:
        return (x, y + offset_distance, r, o)
    # Bottom edge
    elif abs(y - y_max) < 150:
        return (x, y - offset_distance, r, o)

    return (x, y, r, o)


def create_staging_point_corner(ball, offset_distance=200):
    x, y, r, o = ball
    x_min, x_max, y_min, y_max = g.get_field_bounds()
    print("corner")
    # Top left corner
    if x < x_min + 150 and y < y_min + 150:
        return (x + offset_distance, y + offset_distance, r, o)
    # Top right corner
    elif x > x_max - 150 and y < y_min + 150:
        return (x - offset_distance, y + offset_distance, r, o)
    # Bottom left corner
    elif x < x_min + 150 and y > y_max - 150:
        return (x + offset_distance, y - offset_distance, r, o)
    # Bottom right corner
    elif x > x_max - 150 and y > y_max - 150:
        return (x - offset_distance, y - offset_distance, r, o)

    return (x, y - offset_distance, r, o)


def create_staging_point_cross(ball, offset_distance=300):
    bx, by, r, o = ball
    Xmin, Xmax, Ymin, Ymax = g.get_cross_bounds()

    # Midtpunktet af krydset
    cx = (Xmin + Xmax) / 2
    cy = (Ymin + Ymax) / 2

    # Vektor fra kryds til bold
    dx = bx - cx
    dy = by - cy
    mag = math.hypot(dx, dy) or 1.0  # Beskyt mod 0-division

    # Enhedsvektor i den retning
    ux, uy = dx / mag, dy / mag

    # Forskyd stagingpunktet væk fra krydset
    sx = cx + ux * offset_distance
    sy = cy + uy * offset_distance

    return (int(sx), int(sy), r, o)

def create_staging_point_egg(ball, eggs, offset_distance=350):
    bx, by, r, o = ball

    nearest_egg = None
    nearest_dist = None
    for ex, ey, _, _ in eggs:
        dist = math.hypot(bx - ex, by - ey)
        if nearest_dist is None or dist < nearest_dist:
            nearest_dist = dist
            nearest_egg = (ex, ey)

    if nearest_egg is None:
        return ball 

    ex, ey = nearest_egg

    dx = bx - ex
    dy = by - ey
    mag = math.hypot(dx, dy) or 1.0
    ux, uy = dx / mag, dy / mag

    sx = bx + ux * offset_distance
    sy = by + uy * offset_distance

    return (int(sx), int(sy), r, o)



def _point_to_segment_distance(px, py, x1, y1, x2, y2):
    if (x1, y1) == (x2, y2):
        return math.hypot(px - x1, py - y1)
    dx, dy = x2 - x1, y2 - y1
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))
    proj_x, proj_y = x1 + t*dx, y1 + t*dy
    return math.hypot(px - proj_x, py - proj_y)


def barrier_blocks_path(center_marker, ball, egg, robot_radius=80, threshold=40):
    if len(center_marker) == 2:
        cx, cy = center_marker
    else:
        cx, cy = center_marker[:2]

    bx, by = ball[:2]

    forward_vector = (cx - bx, cy - by)

    mag = math.hypot(*forward_vector) or 1.0
    ux, uy = forward_vector[0] / mag, forward_vector[1] / mag

    nx, ny = -uy, ux

    offs_x, offs_y = nx * robot_radius, ny * robot_radius

    line1 = ((bx + offs_x, by + offs_y), (cx + offs_x, cy + offs_y))
    line2 = ((bx - offs_x, by - offs_y), (cx - offs_x, cy - offs_y))

    def dist_to_center(px, py):
        return _point_to_segment_distance(px, py, bx, by, cx, cy)

    def dist_to_edges(px, py):
        d1 = _point_to_segment_distance(
            px, py, line1[0][0], line1[0][1], line1[1][0], line1[1][1])
        d2 = _point_to_segment_distance(
            px, py, line2[0][0], line2[0][1], line2[1][0], line2[1][1])
        return min(d1, d2)

    for ex, ey, er, _ in egg:
        if dist_to_center(ex, ey) <= 20 + er:
            return True
        if dist_to_edges(ex, ey) <= 20 + er:
            return True

    cross_lines = g.get_cross_lines()
    for (x1, y1, x2, y2) in cross_lines:
        steps = 20
        for i in range(steps + 1):
            t = i / steps
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            if dist_to_center(px, py) <= threshold or dist_to_edges(px, py) <= threshold:
                return True


    return False

def close_to_cross(front_marker, back_marker, threshold=100):

    fx_w, fy_w = _correct_marker(front_marker)
    bx_w, by_w = _correct_marker(back_marker)

    dx = fx_w - bx_w
    dy = fy_w - by_w
    norm = math.hypot(dx, dy)
    if norm == 0:
        return False
    ux = dx / norm
    uy = dy / norm

    extension = 5000 
    extended_front = (bx_w + ux * extension, by_w + uy * extension)

    robot_line = ((bx_w, by_w), extended_front)
    closest_dist = None

    for cross_line in g.get_cross_lines():
        x1, y1, x2, y2 = cross_line
        x1c, y1c = _correct_barrier((x1, y1))
        x2c, y2c = _correct_barrier((x2, y2))
        corrected_line = (x1c, y1c, x2c, y2c)
        intersection = find_line_intersection_from_lines(robot_line, corrected_line)
        if intersection:
            ix, iy = intersection

            to_point = (ix - bx_w, iy - by_w)
            forward = (fx_w - bx_w, fy_w - by_w)
            dot = to_point[0]*forward[0] + to_point[1]*forward[1]

            if dot > 0:
                dist = math.hypot(ix - fx_w, iy - fy_w)
                if closest_dist is None or dist < closest_dist:
                    closest_dist = dist

    return closest_dist is not None and closest_dist < threshold

def close_to_barrier(front_marker, back_marker, threshold=55):

    fx_w, fy_w = _correct_marker(front_marker)
    bx_w, by_w = _correct_marker(back_marker)

    dx = fx_w - bx_w
    dy = fy_w - by_w
    norm = math.hypot(dx, dy)
    if norm == 0:
        return False
    ux = dx / norm
    uy = dy / norm

    extension = 5000  
    extended_front = (bx_w + ux * extension, by_w + uy * extension)

    robot_line = ((bx_w, by_w), extended_front)
    closest_dist = None

    for edge_line in g.get_field_lines():
        x1, y1, x2, y2 = edge_line
        x1c, y1c = _correct_barrier((x1, y1))
        x2c, y2c = _correct_barrier((x2, y2))
        corrected_line = (x1c, y1c, x2c, y2c)
        intersection = find_line_intersection_from_lines(robot_line, corrected_line)
        if intersection:
            ix, iy = intersection
            to_point = (ix - bx_w, iy - by_w)
            forward = (fx_w - bx_w, fy_w - by_w)
            dot = to_point[0]*forward[0] + to_point[1]*forward[1]

            if dot > 0:
                dist = math.hypot(ix - fx_w, iy - fy_w)
                if closest_dist is None or dist < closest_dist:
                    closest_dist = dist

    return closest_dist is not None and closest_dist < threshold

def close_to_egg(front_marker, back_marker, eggs, threshold=120):
    fx_w, fy_w = _correct_marker(front_marker)
    bx_w, by_w = _correct_marker(back_marker)

    dx = fx_w - bx_w
    dy = fy_w - by_w
    norm = math.hypot(dx, dy)
    if norm == 0:
        return False
    ux = dx / norm
    uy = dy / norm

    extension = 5000
    extended_front = (bx_w + ux * extension, by_w + uy * extension)

    closest_dist = None

    for ex, ey, er, _ in eggs:
        ex_w, ey_w = _correct_barrier((ex, ey)) 
        dist = _point_to_segment_distance(ex_w, ey_w, bx_w, by_w, extended_front[0], extended_front[1])
        if dist <= threshold + er:
            direct_dist = math.hypot(ex_w - fx_w, ey_w - fy_w)
            if closest_dist is None or direct_dist < closest_dist:
                closest_dist = direct_dist

    return closest_dist is not None and closest_dist < threshold


def slow_down_close_to_barrier(front_marker, back_marker, threshold=150):

    fx_w, fy_w = _correct_marker(front_marker)
    bx_w, by_w = _correct_marker(back_marker)

    dx = fx_w - bx_w
    dy = fy_w - by_w
    norm = math.hypot(dx, dy)
    if norm == 0:
        return False
    ux = dx / norm
    uy = dy / norm

    extension = 5000 
    extended_front = (bx_w + ux * extension, by_w + uy * extension)

    robot_line = ((bx_w, by_w), extended_front)
    closest_dist = None

    for edge_line in g.FIELD_LINES:
        x1, y1, x2, y2 = edge_line
        x1c, y1c = _correct_barrier((x1, y1))
        x2c, y2c = _correct_barrier((x2, y2))
        corrected_line = (x1c, y1c, x2c, y2c)
        intersection = find_line_intersection_from_lines(robot_line, corrected_line)
        if intersection:
            ix, iy = intersection

            to_point = (ix - bx_w, iy - by_w)
            forward = (fx_w - bx_w, fy_w - by_w)
            dot = to_point[0]*forward[0] + to_point[1]*forward[1]

            if dot > 0:
                dist = math.hypot(ix - fx_w, iy - fy_w)
                if closest_dist is None or dist < closest_dist:
                    closest_dist = dist

    return closest_dist is not None and closest_dist < threshold

def draw_lines(robot, ball, eggs, crosses, robot_radius=80, threshold=60):
    fx, fy = robot
    bx, by = ball[:2]

    forward_vector = (fx - bx, fy - by)

    mag = math.hypot(*forward_vector) or 1.0
    ux, uy = forward_vector[0] / mag, forward_vector[1] / mag

    nx, ny = -uy, ux

    offs_x, offs_y = nx * robot_radius, ny * robot_radius

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

def bfs_path(start_zone, goal_zone, eggs, ball_position=None):
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
                        if barrier_blocks_path(start_pos, dummy_ball, eggs):
                            continue
                    else:
                        if zone_path_is_blocked(current, neighbor, eggs):
                            continue

                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
    return None


def get_simplified_path(path_zones, center_marker, ball_pos, eggs):

    simplified_path = []
    current_pos = center_marker

    i = 0
    while i < len(path_zones):
        found = False
        for j in range(len(path_zones) - 1, i, -1):
            zone = path_zones[j]
            target_pos = zone_to_position(*zone)
            dummy_target = (*target_pos, 10, (255, 255, 255))

            if not barrier_blocks_path(current_pos, dummy_target, eggs):
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

def zone_path_is_blocked(zone1, zone2, eggs):
    target_pos = zone_to_position(*zone2)
    dummy_ball = (*target_pos, 10, (255, 255, 255))

    center_pos = zone_to_position(*zone1)

    return barrier_blocks_path(center_pos, dummy_ball, eggs, robot_radius=80, threshold=50)


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
    if (bx >= g.CROSS_X_MIN - 50 and bx <= g.CROSS_X_MAX + 50 and by >= g.CROSS_Y_MIN - 50 and by <= g.CROSS_Y_MAX + 50):
        return True
    else:
        return False
    
def find_line_intersection_from_lines(line1, line2):
    if len(line1) == 4:
        x1, y1, x2, y2 = line1
    else:
        (x1, y1), (x2, y2) = line1

    if len(line2) == 4:
        x3, y3, x4, y4 = line2
    else:
        (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) -
          (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) -
          (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    def on_segment(xa, ya, xb, yb, x, y):
        return min(xa, xb) - 1e-6 <= x <= max(xa, xb) + 1e-6 and \
               min(ya, yb) - 1e-6 <= y <= max(ya, yb) + 1e-6

    if not (on_segment(x1, y1, x2, y2, px, py) and
            on_segment(x3, y3, x4, y4, px, py)):
        return None

    return (px, py)
