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

