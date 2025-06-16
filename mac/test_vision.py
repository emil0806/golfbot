import cv2
import time
import numpy as np
from vision import detect_balls, detect_robot, detect_barriers, detect_egg, detect_cross, inside_field
from pathfinding import (
    find_best_ball, determine_direction,
    is_edge_ball, is_corner_ball,
    create_staging_point_edge, create_staging_point_corner,
    barrier_blocks_path, sort_balls_by_distance, set_homography, determine_staging_point, draw_lines
)

cap = cv2.VideoCapture(0)
last_print_time = time.time()
check = 0
has_staging = False
staged_ball = None
barrier_call = 0

barriers = []
cross = []

FIELD_X_MIN = None
FIELD_X_MAX = None
FIELD_Y_MIN = None
FIELD_Y_MAX = None

while barrier_call < 5:
    ret, frame = cap.read()
    if not ret:
        print("Camera error, no frame captured")
        continue
    robot_info = detect_robot(frame)

    if robot_info:
        robot_position, front_marker, direction = robot_info

        egg = detect_egg(frame)
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
    cross = flat_cross

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture image")
        continue

    frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=0)
    
    egg = detect_egg(frame)

    robot_info = detect_robot(frame)
        
    staged_balls = []
    ball_positions = None
    best_staging = None
    best_ball = None
    ball_positions = None

    if robot_info:
        robot_position, front_marker, direction = robot_info
        rx, ry = robot_position
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

            if barrier_blocks_path(front_marker, best_ball, egg, cross):
                    point_for_staging = determine_staging_point(front_marker, best_ball, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX)
                    x, y = point_for_staging
                    # Lav stagingpunkt (fx direkte vertikal med robotens x og boldens y)
                    staging = (x, y, best_ball[2], best_ball[3])
                    best_ball = staging  # brug stagingpunkt som mål
                    staged_balls.append(best_ball)
                    staged_ball = staging
                    has_staging = True
            elif (has_staging and dist_to_ball > 50):
                staging = (best_ball[0], robot_position[1],
                            best_ball[2], best_ball[3])
                best_ball = staging  # brug stagingpunkt som mål
                staged_balls.append(best_ball)
                staged_ball = staging
                has_staging = True
            else:
                has_staging = False
                staged_ball = None

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

    cv2.imshow("Staging Ball Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
