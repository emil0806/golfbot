import math

previous_best_ball = None


def find_best_ball(ball_positions, robot_position):
    global previous_best_ball

    if not ball_positions or not robot_position:
        return None

    rx, ry = robot_position

    new_best_ball = min(ball_positions, key=lambda ball: math.hypot(ball[0] - rx, ball[1] - ry))

    if previous_best_ball:
        old_dist = math.hypot(previous_best_ball[0] - rx, previous_best_ball[1] - ry)
        new_dist = math.hypot(new_best_ball[0] - rx, new_best_ball[1] - ry)

        if new_dist > old_dist * 0.95: 
            return previous_best_ball

    previous_best_ball = new_best_ball
    print(f"Ball: {new_best_ball}")
    return new_best_ball


def determine_direction(robot_position, ball_position):
    if not robot_position or not ball_position:
        return "stop"

    bx, by = ball_position[:2] 

    (rx, ry), (fx, fy), _ = robot_position 

    vector_to_ball = (bx - rx, by - ry)
    vector_front = (fx - rx, fy - ry)    

    dot_product = vector_to_ball[0] * vector_front[0] + vector_to_ball[1] * vector_front[1]

    magnitude_ball = math.sqrt(vector_to_ball[0] ** 2 + vector_to_ball[1] ** 2)
    magnitude_front = math.sqrt(vector_front[0] ** 2 + vector_front[1] ** 2)

    if magnitude_ball == 0 or magnitude_front == 0:
        return "stop"

    angle_difference = math.degrees(math.acos(dot_product / (magnitude_ball * magnitude_front)))

    cross_product = vector_to_ball[0] * vector_front[1] - vector_to_ball[1] * vector_front[0]
    if cross_product < 0:
        angle_difference = -angle_difference

    print(f"Angle Difference: {angle_difference}")

    if abs(angle_difference) < 30:
        return "forward"
    elif angle_difference > 0:
        return "left" 
    else:
        return "right" 
