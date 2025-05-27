import math

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

    print("Robot Center: ", (rx, ry))
    print("Robot Front: ", (fx, fy))

    print("All ball positions:")
    for i, ball in enumerate(ball_positions):
        dist = math.hypot(ball[0] - rx, ball[1] - ry)
        print(f"  Ball {i}: {ball}, Distance: {dist:.2f}")

    new_best_ball = min(ball_positions, key=lambda ball: math.hypot(ball[0] - fx, ball[1] - fy))
    new_dist = math.hypot(new_best_ball[0] - fx, new_best_ball[1] - fy)
    print(f"New best candidate: {new_best_ball}, Distance: {new_dist:.2f}")

    if previous_best_ball:
        old_dist = math.hypot(previous_best_ball[0] - fx, previous_best_ball[1] - fy)
        print(f"Previous best ball: {previous_best_ball}, Distance: {old_dist:.2f}")

        if new_dist > old_dist * 0.95:
            print("New best is not significantly better. Keeping previous best.")
            return previous_best_ball
        else:
            print("New best is significantly better. Updating.")


    previous_best_ball = new_best_ball
    return new_best_ball


def determine_direction(robot_position, ball_position):
    if not robot_position or not ball_position:
        return "stop"

    bx, by = ball_position[:2] 

    (rx, ry), (fx, fy), _ = robot_position 

    print("Robot Center: ", (rx, ry))
    print("Robot Front: ", (fx, fy))
    print("Ball: ", (bx, by))

    vector_to_ball = (bx - rx, by - ry)
    vector_front = (fx - rx, fy - ry)
    print("V Ball: ", vector_to_ball)
    print("V Front: ", vector_front)    

    dot = vector_front[0] * vector_to_ball[0] + vector_front[1] * vector_to_ball[1]
    mag_f = math.hypot(*vector_front)
    mag_b = math.hypot(*vector_to_ball)
    cos_theta = max(-1, min(1, dot / (mag_f * mag_b)))
    angle_difference = math.degrees(math.acos(cos_theta))

    # Determine if angle is to the left or right using cross product
    cross = -(vector_front[0] * vector_to_ball[1] - vector_front[1] * vector_to_ball[0])

    print(f"Angle: {angle_difference:.2f}°, Cross: {cross:.2f}")

    if angle_difference < 5:
        return "forward"
    elif cross < 0:
        return "right"
    else:
        return "left"


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

def is_corner_ball(ball, margin=150):
    x, y, _, _ = ball

    in_top_left     = (x < 250 + margin and y < 50 + margin)
    in_top_right    = (x > 1600 - margin and y < 50 + margin)
    in_bottom_left  = (x < 250 + margin and y > 1050 - margin)
    in_bottom_right = (x > 1600 - margin and y > 1050 - margin)

    return in_top_left or in_top_right or in_bottom_left or in_bottom_right


def is_edge_ball(ball, margin=150):
    x, y, _, _ = ball

    # check each edge, ignoring corners (corners are covered by is_corner_ball)
    near_left   = 250 - margin < x < 250 + margin and 50 + margin < y < 1050 - margin
    near_right  = 1600 - margin < x < 1600 + margin and 50 + margin < y < 1050 - margin
    near_top    = 50 - margin < y < 50 + margin and 250 + margin < x < 1600 - margin
    near_bottom = 1050 - margin < y < 1050 + margin and 250 + margin < x < 1600 - margin

    return near_left or near_right or near_top or near_bottom



def create_staging_point_edge(ball, offset_distance=200):
    x, y, r, o = ball

    # Venstre kant
    if x < 400:
        return (x + offset_distance, y, r, o)
    # Højre kant
    elif x > 1200:
        return (x - offset_distance, y, r, o)
    # Øverste kant
    elif y < 400:
        return (x, y + offset_distance, r, o)
    # Nederste kant
    elif y > 800:
        return (x, y - offset_distance, r, o)

    # Fallback – midt i banen
    return (x - offset_distance, y - offset_distance, r, o)


def create_staging_point_corner(ball, offset_distance=200):
    x, y, r, o = ball

    # Koordinatsystem: 0,0 = øverste venstre hjørne
    if x < 500 and y < 500:
        # Øverste venstre hjørne
        return (x + offset_distance, y + offset_distance, r, o)
    elif x > 1000 and y < 500:
        # Øverste højre hjørne
        return (x - offset_distance, y + offset_distance, r, o)
    elif x < 500 and y > 500:
        # Nederste venstre hjørne
        return (x + offset_distance, y - offset_distance, r, o)
    elif x > 1000 and y > 500:
        # Nederste højre hjørne
        return (x - offset_distance, y - offset_distance, r, o)
    
    # Fallback: staging lidt opad
    return (x, y - offset_distance, r, o)


# ------------------ ÆG-UNDVIGELSE ------------------

def _point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Korteste afstand fra punkt (px,py) til linjesegment (x1,y1)-(x2,y2)."""
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

def delivery_routine(robot_info):
    # Simple placeholder routine
    # Go forward to an approach point, turn, then reverse
    return "delivery"

def stop_delivery_routine():
    return "continue"
