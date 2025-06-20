from collections import deque
import traceback
import cv2
import time
from robot_controller import RobotController
from robot_state import RobotState
from setup import setup_cross_lines, setup_homography
import numpy as np
from vision import detect_balls, detect_robot, detect_egg, stabilize_detections
from pathfinding import (bfs_path, get_cross_zones, get_grid_thresholds, get_simplified_path, get_simplified_target, get_zone_center, get_zone_for_position, sort_balls_by_distance,
    is_corner_ball, is_edge_ball, create_staging_point_corner, create_staging_point_edge, zone_to_position)
import globals_config as g

### CAMERA FEED ###
cap = cv2.VideoCapture(0)

### CONTROLLER ###
controller = RobotController()

### STAGING ###
has_staging = False
at_staging = False
at_blocked_staging = False
staged_ball = None
staging_target = None
staged_ball = None
staged_balls = []

### ROBOT ###
last_robot_info = None

### BALLS ###
prev_ball_count = 11
ball_history = deque(maxlen=5)  # Saves latest five frames

### CROSS ###
cross = []

### BARRIER ### 
barrier_call = 0
barriers = []

### CORNER ###
corner_stage = 0
corner_timer = 0
corner_ball = None

### EGG ###
egg = None

### OTHER ###
timer = 0
last_command = None


# ------ SETUP ------
try:
    cross, cross_center, egg, last_robot_info = setup_cross_lines(cap, last_robot_info)

    H = setup_homography()
except Exception as e:
    print(f"[ERROR] Programmet stødte på en fejl: {e}")
    traceback.print_exc()
    time.sleep(1) 

state = RobotState.COLLECTION

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture image")
            continue

        frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=0)
        
        robot_info = detect_robot(frame)
            
        if robot_info:
                last_robot_info = robot_info
        if not robot_info:
            print("Robot not detected - skipping this frame")
            continue

        if robot_info:
            front_marker, center_marker, back_marker, _ = robot_info

            fx, fy = front_marker
            cx, cy = center_marker
            rx, ry = back_marker

            egg = detect_egg(frame, back_marker, front_marker)

            current_balls = detect_balls(frame, egg, back_marker, front_marker)

            stable_balls = stabilize_detections(current_balls, ball_history)

            ball_history.append(current_balls)

            ball_positions = stable_balls

            pre_sorted_balls = sort_balls_by_distance(ball_positions, front_marker)
            original_ball = pre_sorted_balls[0] if pre_sorted_balls else None
            bx, by = original_ball[:2]

            if is_edge_ball(original_ball):
                target_ball = create_staging_point_edge(original_ball)
                controller.edge_alignment_active = True
            else:
                target_ball = original_ball
                controller.edge_alignment_active = False

            bx, by = target_ball[:2]

            field_bounds = g.get_field_bounds()
            for ball in pre_sorted_balls:
                if is_corner_ball(ball):
                    staged_balls.append((create_staging_point_corner(ball, field_bounds)))
                elif is_edge_ball(ball):
                    staged_balls.append((create_staging_point_edge(ball)))

            robot_zone = get_zone_for_position(cx, cy)
            ball_zone = get_zone_for_position(bx, by)
            forbidden_zones = get_cross_zones()

            path = bfs_path(robot_zone, ball_zone, forbidden_zones)

            if path and len(path) > 1:
                controller.path_to_target = path
            else:
                controller.path_to_target = None

            if controller.path_to_target:
                simplified_path = get_simplified_path(controller.path_to_target, center_marker, target_ball, egg, cross)
                controller.simplified_path = simplified_path
                print(f"simple path: {simplified_path}")

            if controller.edge_alignment_active and controller.path_to_target == []:
                robot_zone = get_zone_for_position(cx, cy)
                ball_zone = get_zone_for_position(original_ball[0], original_ball[1])
                forbidden_zones = get_cross_zones()

                path = bfs_path(robot_zone, ball_zone, forbidden_zones)
                if path and len(path) > 1:
                    controller.path_to_target = path[1:]
                controller.edge_alignment_active = False 

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

        if target_ball:
            x, y, r, _ = target_ball
            cv2.circle(frame, (int(x), int(y)), int(r), (50, 50, 50), 2)
            cv2.putText(frame, "Target ball", (int(x) - 35, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # --- Tegn robot ---
        if robot_info:
            (fx, fy),(cx, cy), (rx, ry), _ = robot_info
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
                
        x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6 = get_grid_thresholds(g.FIELD_X_MIN, g.FIELD_X_MAX, g.FIELD_Y_MIN, g.FIELD_Y_MAX)
            
        for x in [x1, x2, x3, x4, x5, x6]:
            cv2.line(frame, (int(x), int(g.FIELD_Y_MIN)), (int(x), int(g.FIELD_Y_MAX)), (255, 255, 0), 2)
        for y in [y1, y2, y3, y4, y5, y6]:
            cv2.line(frame, (int(g.FIELD_X_MIN), int(y)), (int(g.FIELD_X_MAX), int(y)), (255, 255, 0), 2)

        if stable_balls:
            for x, y, r, color in stable_balls:
                color_bgr = (0, 140, 255) if color == 1 else (255, 255, 255)  # Orange eller hvid
                cv2.circle(frame, (x, y), r, color_bgr, 2)
                cv2.putText(frame, "Stabilized", (int(x) - 40, int(y) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.imshow("Stabilized Balls", frame)
        
        if hasattr(controller, 'simplified_path') and controller.simplified_path:
            path_points = [(cx, cy)] + controller.simplified_path

            for i in range(len(path_points) - 1):
                x1, y1 = path_points[i][:2]
                x2, y2 = path_points[i + 1][:2]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)


        cv2.imshow("Staging Ball Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except Exception as e:
        print(f"[ERROR] Programmet stødte på en fejl: {e}")
        traceback.print_exc()
        time.sleep(1)  # Undgå at spamme CPU og console hvis fejl sker i loop
        continue

cap.release()
cv2.destroyAllWindows()
