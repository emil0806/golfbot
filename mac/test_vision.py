import cv2
import time
import numpy as np
from vision import detect_balls, detect_robot, detect_barriers, detect_egg, detect_cross, inside_field
from pathfinding import (determine_direction, find_best_ball, sort_balls_by_distance,
    is_corner_ball, is_edge_ball, create_staging_point_corner, create_staging_point_edge,
    egg_blocks_path, create_staging_point_egg, delivery_routine, stop_delivery_routine, 
    barrier_blocks_path, close_to_barrier, set_homography, determine_staging_point, is_ball_and_robot_on_line_with_cross, is_ball_in_cross, draw_lines, get_grid_thresholds, determine_staging_point_16, determine_zone)

cap = cv2.VideoCapture(0)
last_print_time = time.time()
check = 0
has_staging = False
staged_ball = None
barrier_call = 0
at_blocked_staging = False


barriers = []
cross = []
egg = None

FIELD_X_MIN = None
FIELD_X_MAX = None
FIELD_Y_MIN = None
FIELD_Y_MAX = None

CROSS_X_MIN = None
CROSS_X_MAX = None
CROSS_Y_MIN = None
CROSS_Y_MAX = None
CROSS_CENTER = None

while barrier_call < 5:
    ret, frame = cap.read()
    if not ret:
        print("Camera error, no frame captured")
        continue
    robot_info = detect_robot(frame)

    if robot_info:
        robot_position, front_marker, direction = robot_info

        egg = detect_egg(frame, robot_position, front_marker)
        ball_positions = detect_balls(frame, egg,
                                       robot_position, front_marker)

        bar = detect_barriers(frame, robot_position, ball_positions)
        barriers.append(bar)

        cross_line = detect_cross(frame,
                                  robot_position,
                                  front_marker,
                                  ball_positions,
                                  bar)
        cross.append(cross_line)
    barrier_call += 1

if barriers:
    flat_barriers = [b for sublist in barriers for b in sublist]
    FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX = inside_field(flat_barriers)
    barriers = flat_barriers
else:
    barriers = []
    FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX = 0, frame.shape[
        1], 0, frame.shape[0]
    
    # Homografi til test  (samme logik som i mac_server.py)
    PIX_CORNERS = np.float32([
        [FIELD_X_MIN, FIELD_Y_MIN],
        [FIELD_X_MAX, FIELD_Y_MIN],
        [FIELD_X_MAX, FIELD_Y_MAX],
        [FIELD_X_MIN, FIELD_Y_MAX]
    ])
    FIELD_W, FIELD_H = 1800, 1200
    WORLD_CORNERS = np.float32([
        [0,        0],
        [FIELD_W,  0],
        [FIELD_W,  FIELD_H],
        [0,        FIELD_H]
    ])
    H, _ = cv2.findHomography(PIX_CORNERS, WORLD_CORNERS)
    set_homography(H)

if cross:
    flat_cross = [c for sublist in cross for c in sublist]
    CROSS_X_MIN, CROSS_X_MAX, CROSS_Y_MIN, CROSS_Y_MAX = inside_field(
        flat_cross)
    CROSS_CENTER = (
            (CROSS_X_MIN + CROSS_X_MAX) // 2,
            (CROSS_Y_MIN + CROSS_Y_MAX) // 2
        )
    cross = flat_cross

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture image")
        continue

    frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=0)
    
    robot_info = detect_robot(frame)
        
    staged_balls = []
    ball_positions = None
    best_staging = None
    best_ball = None
    ball_positions = None

    if robot_info:
        robot_position, front_marker, direction = robot_info

        egg = detect_egg(frame, robot_position, front_marker)

        rx, ry = robot_position
        fx, fy = front_marker
        cm_x = (fx + rx) // 2
        cm_y = (fy + ry) // 2
        center_robot = (cm_x, cm_y)
        ball_positions = detect_balls(frame, egg, robot_position, front_marker)
    
        ball_positions = [(x, y, r, o) for (x, y, r, o) in ball_positions if FIELD_X_MIN + 10 <
                        x < FIELD_X_MAX - 10 and FIELD_Y_MIN + 10 < y < FIELD_Y_MAX - 10]

        # Filtrér bolde for afstand til robot
        COLLECTION_RADIUS = 20
        ball_positions = [
            (x, y, r, o) for (x, y, r, o) in ball_positions
            if np.linalg.norm(np.array((x, y)) - np.array((rx, ry))) > COLLECTION_RADIUS
        ]

        sorted_balls = sort_balls_by_distance(ball_positions, front_marker)
        best_ball = sorted_balls[0] if sorted_balls else None

        # Vis staging til alle kant/hjørnebolde (debug formål)
        field_bounds = (FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX)
        for ball in ball_positions:
            if is_corner_ball(ball, field_bounds):
                staged_balls.append((create_staging_point_corner(ball, field_bounds)))
            elif is_edge_ball(ball, field_bounds):
                staged_balls.append((create_staging_point_edge(ball, field_bounds)))

        if best_ball:
            # Hvis best_ball er edge eller corner
            if is_corner_ball(best_ball, field_bounds):
                staging = create_staging_point_corner(best_ball, field_bounds)
            elif is_edge_ball(best_ball, field_bounds):
                staging = create_staging_point_edge(best_ball, field_bounds)
            else:
                staging = None

            if staging:
                staging_dist = np.linalg.norm(
                    np.array(staging[:2]) - np.array(front_marker))
                ball_dist = np.linalg.norm(
                    np.array(best_ball[:2]) - np.array(front_marker))

                robot_vector = np.array(front_marker) - \
                    np.array(robot_position)
                ball_vector = np.array(
                    best_ball[:2]) - np.array(robot_position)

                dot = np.dot(robot_vector, ball_vector)
                mag_r = np.linalg.norm(robot_vector)
                mag_b = np.linalg.norm(ball_vector)
                cos_theta = max(-1, min(1, dot / (mag_r * mag_b + 1e-6)))
                angle_diff = np.degrees(np.arccos(cos_theta))

                if (staging_dist > 80 and angle_diff > 10) or ball_dist > 120:
                    best_staging = staging
                    best_ball = staging

            dist_to_ball = 0 if staged_ball is None else np.linalg.norm(
                np.array(staged_ball[:2]) - np.array(robot_position))
            line1, line2 = draw_lines(front_marker, best_ball, egg, cross)

            # Tegn linje 1
            cv2.line(frame, 
                    (int(line1[0][0]), int(line1[0][1])), 
                    (int(line1[1][0]), int(line1[1][1])), 
                    (255, 255, 0), 2)

            # Tegn linje 2
            cv2.line(frame, 
                    (int(line2[0][0]), int(line2[0][1])), 
                    (int(line2[1][0]), int(line2[1][1])), 
                    (0, 255, 255), 2)

            if barrier_blocks_path(center_robot, best_ball, egg, cross):
                y = 0
                x = 0
                in_line = is_ball_and_robot_on_line_with_cross(center_robot, best_ball, CROSS_X_MIN, CROSS_X_MAX, CROSS_Y_MIN, CROSS_Y_MAX, CROSS_CENTER)
                # Venstre for kryds
                if (in_line == 1):
                    print("in_line 1")
                    y = ((FIELD_Y_MAX - FIELD_Y_MIN) * 0.50) + FIELD_Y_MIN
                    x = ((FIELD_X_MAX - FIELD_X_MIN) * 0.15) + FIELD_X_MIN
                # Over kryds
                elif(in_line == 2):
                    print("in_line 2")
                    y = ((FIELD_Y_MAX - FIELD_Y_MIN) * 0.15) + FIELD_Y_MIN
                    x = ((FIELD_X_MAX - FIELD_X_MIN) * 0.50) + FIELD_X_MIN
                # Højre for kryds
                elif(in_line == 3):
                    y = ((FIELD_Y_MAX - FIELD_Y_MIN) * 0.50) + FIELD_Y_MIN
                    x = ((FIELD_X_MAX - FIELD_X_MIN) * 0.85) + FIELD_X_MIN
                # Under kryds
                elif(in_line == 4):
                    y = ((FIELD_Y_MAX - FIELD_Y_MIN) * 0.85) + FIELD_Y_MIN
                    x = ((FIELD_X_MAX - FIELD_X_MIN) * 0.50) + FIELD_X_MIN
                elif(in_line == 5):
                    x, y = determine_staging_point(center_robot, best_ball, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX, CROSS_CENTER)
                
                staging = (x, y, best_ball[2], best_ball[3])
                staging_dist = np.linalg.norm(
                    np.array(staging[:2]) - np.array((cm_x, cm_y)))
                
                if (staging_dist < 30):
                    at_blocked_staging = True
                
                if not at_blocked_staging:                        
                    best_ball = staging  
                    staged_balls.append(best_ball)

    # --- Debug print hver 5. sek ---
    if time.time() - last_print_time >= 5:
        print("Robot info:", robot_info)
        print("Ball positions:", ball_positions)
        print("Best ball:", best_ball)
        last_print_time = time.time()

    # --- Tegn originale bolde (grøn) ---
    if ball_positions:
        for (x, y, radius, color) in ball_positions:
            cv2.circle(frame, (x, y), int(radius), (0, 255, 0), 2)
            cv2.putText(frame, "Ball", (x - 20, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- Tegn staging-points (lilla) ---
    if staged_balls:
        for (x, y, r, o) in staged_balls:
            cv2.circle(frame, (int(x), int(y)), int(r), (255, 0, 255), 2)
            cv2.putText(frame, "Staging", (int(x) - 25, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    if best_staging:
        x, y, r, _ = best_staging
        cv2.circle(frame, (int(x), int(y)), int(r), (255, 0, 0), 2)
        cv2.putText(frame, "Best Staging", (int(x) - 35, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # --- Tegn robot ---
    if robot_info:
        (rx, ry), (fx, fy), _ = robot_info
        cv2.circle(frame, (rx, ry), 10, (255, 0, 0), 2)
        cv2.putText(frame, "Robot", (rx - 20, ry - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (fx, fy), 10, (0, 165, 255), 2)
        cv2.putText(frame, "Front", (fx - 20, fy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        cv2.arrowedLine(frame, (rx, ry), (fx, fy), (0, 255, 0), 2)

    if barriers:
    # --- Tegn barriers og kryds ---
        for (x1, y1, x2, y2), center in barriers:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cx, cy = center
            cv2.putText(frame, "Barrier", (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if cross:
        for (x1, y1, x2, y2) in cross:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(frame, "Cross", (cx - 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # --- Tegn æg ---
    if egg:
        for (x, y, radius, color) in egg:
            cv2.circle(frame, (x, y), int(radius), (0, 255, 255), 2)
            cv2.putText(frame, "Egg", (x - 20, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
    x1, x2, x3, y1, y2, y3 = get_grid_thresholds(FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX)
        
    for x in [x1, x2, x3]:
        cv2.line(frame, (int(x), int(FIELD_Y_MIN)), (int(x), int(FIELD_Y_MAX)), (255, 255, 0), 2)
    for y in [y1, y2, y3]:
        cv2.line(frame, (int(FIELD_X_MIN), int(y)), (int(FIELD_X_MAX), int(y)), (255, 255, 0), 2)


    cv2.imshow("Staging Ball Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
