import cv2
import time
from vision import detect_balls, detect_robot, detect_barriers
from pathfinding import (
    find_best_ball, determine_direction,
    is_edge_ball, is_corner_ball,
    create_staging_point_edge, create_staging_point_corner
)

cap = cv2.VideoCapture(0)
last_print_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture image")
        continue

    ball_positions = detect_balls(frame)
    barriers = detect_barriers(frame)
    robot_info = detect_robot(frame)

    # --- Calculate staging points ---
    staged_balls = []
    for ball in ball_positions:
        if is_corner_ball(ball):
            staged = create_staging_point_corner(ball)
            staged_balls.append(staged)
        elif is_edge_ball(ball):
            staged = create_staging_point_edge(ball)
            staged_balls.append(staged)

    # --- Print debug info every 5 seconds ---
    if time.time() - last_print_time >= 5:
        if robot_info:
            robot_position, front_marker, direction = robot_info
            best_ball = find_best_ball(ball_positions, robot_position, front_marker)
            movement_command = determine_direction(robot_info, best_ball)
            print("Movement command:", movement_command)
        print("Robot info:", robot_info)
        print("Ball positions:", ball_positions)
        print("Staging positions:", staged_balls)
        last_print_time = time.time()

    # --- Draw original detected balls ---
    for (x, y, radius, color) in ball_positions:
        cv2.circle(frame, (x, y), int(radius), (0, 255, 0), 2)
        cv2.putText(frame, "Ball", (x - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- Draw staging balls ---
    for (x, y, radius, color) in staged_balls:
        cv2.circle(frame, (x, y), int(radius), (255, 0, 255), 2)
        cv2.putText(frame, "Staging", (x - 25, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # --- Draw robot ---
    if robot_info:
        robot_position, front_marker, direction = robot_info
        rx, ry = robot_position
        fx, fy = front_marker

        cv2.circle(frame, (rx, ry), 10, (255, 0, 0), 2)
        cv2.putText(frame, "Robot", (rx - 20, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame, (fx, fy), 10, (0, 165, 255), 2)
        cv2.putText(frame, "Front", (fx - 20, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        cv2.arrowedLine(frame, (rx, ry), (fx, fy), (0, 255, 0), 2)

    # --- Draw barriers ---
    for (rect, _) in barriers:
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Barrier", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Staging Ball Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
