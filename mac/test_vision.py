import cv2
import time
import numpy as np
from vision import detect_balls, detect_robot, detect_barriers, detect_egg, detect_cross
from pathfinding import (
    find_best_ball, determine_direction,
    is_edge_ball, is_corner_ball,
    create_staging_point_edge, create_staging_point_corner,
    barrier_blocks_path, sort_balls_by_distance
)

cap = cv2.VideoCapture(0)
last_print_time = time.time()
check = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture image")
        continue

    egg = detect_egg(frame)
    ball_positions = detect_balls(frame, egg)
    robot_info = detect_robot(frame)



    staged_balls = []
    best_staging = None
    best_ball = None

    if robot_info:
        robot_position, front_marker, direction = robot_info
        rx, ry = robot_position

    if check == 0:
        cross_lines = detect_cross(frame, robot_position)
        barriers = detect_barriers(frame, robot_position)
        check = 1

        # Filtrér bolde for afstand til robot
        COLLECTION_RADIUS = 20
        ball_positions = [
            (x, y, r, o) for (x, y, r, o) in ball_positions
            if np.linalg.norm(np.array((x, y)) - np.array((rx, ry))) > COLLECTION_RADIUS
        ]

        sorted_balls = sort_balls_by_distance(ball_positions, front_marker)
        best_ball = sorted_balls[0] if sorted_balls else None

        # Vis staging til alle kant/hjørnebolde (debug formål)
        for ball in ball_positions:
            if is_corner_ball(ball):
                staged_balls.append((create_staging_point_corner(ball), "purple"))
            elif is_edge_ball(ball):
                staged_balls.append((create_staging_point_edge(ball), "purple"))

        if best_ball:
            # Hvis best_ball er edge eller corner
            if is_corner_ball(best_ball):
                staging = create_staging_point_corner(best_ball)
            elif is_edge_ball(best_ball):
                staging = create_staging_point_edge(best_ball)
            else:
                staging = None

            if staging:
                staging_dist = np.linalg.norm(np.array(staging[:2]) - np.array(front_marker))
                ball_dist = np.linalg.norm(np.array(best_ball[:2]) - np.array(front_marker))

                robot_vector = np.array(front_marker) - np.array(robot_position)
                ball_vector = np.array(best_ball[:2]) - np.array(robot_position)

                dot = np.dot(robot_vector, ball_vector)
                mag_r = np.linalg.norm(robot_vector)
                mag_b = np.linalg.norm(ball_vector)
                cos_theta = max(-1, min(1, dot / (mag_r * mag_b + 1e-6)))
                angle_diff = np.degrees(np.arccos(cos_theta))

                if (staging_dist > 80 and angle_diff > 10) or ball_dist > 120:
                    best_staging = staging
                    best_ball = staging

            if barrier_blocks_path(robot_position, best_ball, egg, cross_lines):
                staging = (best_ball[0], robot_position[1], best_ball[2], best_ball[3])
                best_staging = staging
                best_ball = staging

    # --- Debug print hver 5. sek ---
    if time.time() - last_print_time >= 5:
        print("Robot info:", robot_info)
        print("Ball positions:", ball_positions)
        print("Best ball:", best_ball)
        last_print_time = time.time()

    # --- Tegn originale bolde (grøn) ---
    for (x, y, radius, color) in ball_positions:
        cv2.circle(frame, (x, y), int(radius), (0, 255, 0), 2)
        cv2.putText(frame, "Ball", (x - 20, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- Tegn staging-points (lilla) ---
    for (staging, color) in staged_balls:
        x, y, r, _ = staging
        draw_color = (255, 0, 255) if color == "purple" else (255, 0, 0)
        cv2.circle(frame, (x, y), int(r), draw_color, 2)
        label = "Staging" if color == "purple" else "Best Staging"
        cv2.putText(frame, label, (x - 30, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)

    if best_staging:
        x, y, r, _ = best_staging
        cv2.circle(frame, (x, y), int(r), (255, 0, 0), 2)
        cv2.putText(frame, "Best Staging", (x - 35, y - 10),
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

    # --- Tegn barriers og kryds ---
    for (x1, y1, x2, y2), center in barriers:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cx, cy = center
        cv2.putText(frame, "Barrier", (cx - 20, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for (x1, y1, x2, y2) in cross_lines:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(frame, "Cross", (cx - 15, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # --- Tegn æg ---
    for (x, y, radius, color) in egg:
        cv2.circle(frame, (x, y), int(radius), (0, 255, 255), 2)
        cv2.putText(frame, "Egg", (x - 20, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imshow("Staging Ball Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
