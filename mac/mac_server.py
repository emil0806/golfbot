import socket
import time
import cv2
from pathfinding import (determine_direction, sort_balls_by_distance,
    is_corner_ball, is_edge_ball, create_staging_point_corner, create_staging_point_edge,
    barrier_blocks_path, close_to_barrier, set_homography, determine_staging_point, is_ball_and_robot_on_line_with_cross, draw_lines)
import numpy as np
from vision import detect_balls, detect_robot, detect_barriers, detect_egg, detect_cross, inside_field, filter_barriers_inside_field
from config import EV3_IP, PORT
import time


# Initialize socket server to send data to EV3
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", PORT))
server_socket.listen(1)

print(f"Server lytter på 0.0.0.0:{PORT}...")

print(f"Waiting for EV3 connection on port {PORT}...")

conn, addr = server_socket.accept()
print(f"Connection established with EV3 at {addr}")

# Open the camera feed
cap = cv2.VideoCapture(0)

last_command = None
timer = 0
barrier_call = 0
has_staging = False
at_staging = False
at_blocked_staging = False
staged_ball = None
delivery_stage = 0
last_delivery_count = 11
prev_ball_count = 11
corner_stage = 0
corner_timer = 0
corner_ball = None
staging_target = None
last_robot_info = None

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

while barrier_call < 8:
    ret, frame = cap.read()
    if not ret:
        print("Camera error, no frame captured")
        continue

    frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=0)
    robot_info = detect_robot(frame)
    if robot_info:
        robot_position, front_marker, _ = robot_info
    else:
        robot_position = front_marker = None
    ball_positions = detect_balls(frame, egg, robot_position, front_marker)

    if robot_info:
        robot_position, front_marker, direction = robot_info
        egg = detect_egg(frame, robot_position, front_marker)

        bar = detect_barriers(frame, robot_position, ball_positions)
        bar = filter_barriers_inside_field(bar, frame.shape)
        barriers.append(bar)
        cross.append(
            detect_cross(frame,
                         robot_position,
                         front_marker,
                         ball_positions,
                         bar)
        )
    barrier_call += 1
if barriers:
    flat_barriers = [b for sublist in barriers for b in sublist]
    FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX = inside_field(
        flat_barriers)
    
    barriers = flat_barriers

    xs = []
    ys = []
    for ((x1, y1, x2, y2), _) in barriers:
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    if len(xs) >= 4 and len(ys) >= 4:
        # Sorter og fjern outliers vha. percentil
        FIELD_X_MIN = int(np.percentile(xs, 10))
        FIELD_X_MAX = int(np.percentile(xs, 90))
        FIELD_Y_MIN = int(np.percentile(ys, 10))
        FIELD_Y_MAX = int(np.percentile(ys, 90))

    # ----------  BEREGN HOMOGRAFI  ---------------
    PIX_CORNERS = np.float32([
        [FIELD_X_MIN, FIELD_Y_MIN],   # top-left
        [FIELD_X_MAX, FIELD_Y_MIN],   # top-right
        [FIELD_X_MAX, FIELD_Y_MAX],   # bottom-right
        [FIELD_X_MIN, FIELD_Y_MAX]    # bottom-left
    ])

    # Kendt faktisk bane-størrelse i mm  (tilpas hvis nødvendigt)
    FIELD_W, FIELD_H = 1800, 1200
    WORLD_CORNERS = np.float32([
        [0,        0],
        [FIELD_W,  0],
        [FIELD_W,  FIELD_H],
        [0,        FIELD_H]
    ])

    H, _ = cv2.findHomography(PIX_CORNERS, WORLD_CORNERS)
    set_homography(H)                    # gem matrixen globalt i pathfinding.py

else:
    barriers = []
    FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX = 0, frame.shape[
        1], 0, frame.shape[0]

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
        print("Camera error, no frame captured")
        continue
    frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=0)

    robot_info = detect_robot(frame)

    if robot_info:
        robot_position, front_marker, direction = robot_info

        rx, ry = robot_position
        fx, fy = front_marker
        cm_x = (fx + rx) / 2
        cm_y = (fy + ry) / 2
        center_robot = (cm_x, cm_y)

        current_time = time.time()

        egg = detect_egg(frame, robot_position, front_marker)
        ball_positions = detect_balls(frame, egg, robot_position, front_marker)
        ball_positions = [(x, y, r, o) for (
            x, y, r, o) in ball_positions if FIELD_X_MIN < x < FIELD_X_MAX and FIELD_Y_MIN < y < FIELD_Y_MAX]
        timer = current_time

        COLLECTION_RADIUS = 20
        ball_positions = [
            (x, y, r, o) for (x, y, r, o) in ball_positions
            if np.linalg.norm(np.array((x, y)) - np.array((rx, ry))) > COLLECTION_RADIUS
        ]

        staged_balls = []

        if (len(ball_positions) != prev_ball_count):
            at_staging = False
            at_blocked_staging = False
            prev_ball_count = len(ball_positions)

        ###   Delivery   ###
        if (len(ball_positions) in [0, 4, 8] and last_delivery_count != len(ball_positions)):
            if delivery_stage == 0:
                print("Initiating delivery routine...")
                delivery_stage = 1

                staging_target = (FIELD_X_MAX - 300,
                                  (FIELD_Y_MIN + FIELD_Y_MAX) // 2)
                back_alignment_target = (
                    FIELD_X_MAX - 20, (FIELD_Y_MIN + FIELD_Y_MAX) // 2)

            if delivery_stage == 1:
                cm_x = (front_marker[0] + robot_position[0]) / 2
                cm_y = (front_marker[1] + robot_position[1]) / 2
                dist_to_staging = np.linalg.norm(
                    np.array((cm_x, cm_y)) - np.array(staging_target))
                print(f"[Stage 1] Distance to staging: {dist_to_staging:.2f}")
                if dist_to_staging > 100:
                    dummy_target = (*staging_target, 10, (255, 255, 255))
                    if barrier_blocks_path(front_marker, dummy_target, egg, cross):
                        y = 0
                        x = 0
                        if (cm_x <= ((FIELD_X_MAX - FIELD_X_MIN) * 0.5)):
                            if(cm_y <= (FIELD_Y_MAX - FIELD_Y_MIN) * 0.5):
                                y = ((FIELD_Y_MAX - FIELD_Y_MIN) * 0.20) + FIELD_Y_MIN
                                x = ((FIELD_X_MAX - FIELD_X_MIN) * 0.50) + FIELD_X_MIN
                            elif (cm_y >= (FIELD_Y_MAX - FIELD_Y_MIN) * 0.5):
                                y = ((FIELD_Y_MAX - FIELD_Y_MIN) * 0.80) + FIELD_Y_MIN
                                x = ((FIELD_X_MAX - FIELD_X_MIN) * 0.50) + FIELD_X_MIN
                        else:
                            x = FIELD_X_MAX - 200
                            y = (FIELD_Y_MIN + FIELD_Y_MAX) // 2

                        staging = (x, y, dummy_target[2], dummy_target[3])
                        staging_dist = np.linalg.norm(
                            np.array(staging[:2]) - np.array((cm_x, cm_y)))
                        
                        if (staging_dist < 30):
                            at_blocked_staging = True
                            has_staging = False
                        
                        if not at_blocked_staging:                        
                            dummy_target = staging  
                            staged_balls.append(dummy_target)
                            staged_ball = staging
                            has_staging = True
                    movement_command = determine_direction(
                        robot_info, dummy_target, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX)
                    if movement_command != last_command:
                        conn.sendall(movement_command.encode())
                        last_command = command
                else:
                    delivery_stage = 2

            if delivery_stage == 2:
                robot_vector = np.array(
                    robot_position) - np.array(front_marker)
                desired_vector = np.array(
                    back_alignment_target) - np.array(robot_position)

                dot = np.dot(robot_vector, desired_vector)
                mag_r = np.linalg.norm(robot_vector)
                mag_d = np.linalg.norm(desired_vector)
                cos_theta = max(-1, min(1, dot / (mag_r * mag_d + 1e-6)))
                angle_diff = np.degrees(np.arccos(cos_theta))

                print(f"[Stage 2] Angle to target: {angle_diff:.2f}")

                if angle_diff > 1.5:
                    robot_3d = np.append(robot_vector, 0)
                    desired_3d = np.append(desired_vector, 0)
                    cross_product = np.cross(robot_3d, desired_3d)[
                        2] 
                    if angle_diff > 25:
                        movement_command = "left"
                    elif angle_diff > 15:
                        movement_command = "medium_left"
                    else:
                        movement_command = "slow_left"
                    if movement_command != last_command:
                        conn.sendall(movement_command.encode())
                        last_command = movement_command
                else:
                    delivery_stage = 3

            if delivery_stage == 3:
                dist_back = np.linalg.norm(
                    np.array(robot_position) - np.array(back_alignment_target))
                print(f"[Stage 3] Distance to back_alignment: {dist_back:.2f}")
                if dist_back > 95:
                    movement_command = "slow_backward"
                    if movement_command != last_command:
                        conn.sendall(movement_command.encode())
                        last_command = movement_command
                else:
                    delivery_stage = 4

            if delivery_stage == 4:
                command = "delivery"
                if command != last_command:
                    print("[Stage 4] Sending delivery command")
                    conn.sendall(command.encode())
                    last_command = command
                time.sleep(4)
                command = "continue"
                if command != last_command:
                    print("[Stage 4.2] Sending continue command")
                    conn.sendall(command.encode())
                    last_command = command
                delivery_stage = 0 
                last_delivery_count = len(ball_positions)

        else:
            print("test")
            pre_sorted_balls = sort_balls_by_distance(
                ball_positions, front_marker)
            best_ball = pre_sorted_balls[0] if pre_sorted_balls else None

            field_bounds = (FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX)
            corner_balls = [b for b in ball_positions if is_corner_ball(b, field_bounds)]

            if((len(ball_positions) == 8 or len(ball_positions) == 4) and len(corner_balls) > 0):
                if(corner_ball is None or (current_time - corner_timer) > 3):
                        corner_timer = current_time
                        corner_ball = corner_balls[0] if corner_balls else None
                        best_ball = corner_balls[0] if corner_balls else None
                else: 
                    best_ball = corner_ball
                if corner_stage == 0:
                        corner_stage = 1
            else:
                if (len(ball_positions) == len(corner_balls)):
                    if(corner_ball is None or (current_time - corner_timer) > 3):
                        corner_timer = current_time
                        corner_ball = pre_sorted_balls[0] if pre_sorted_balls else None
                        best_ball = pre_sorted_balls[0] if pre_sorted_balls else None
                    else: 
                        best_ball = corner_ball
                    if corner_stage == 0:
                        corner_stage = 1
                else:
                    for ball in pre_sorted_balls:
                        if(ball[3] == 1):
                            best_ball == ball
                            break
                        elif(not is_corner_ball(ball, field_bounds)):
                            best_ball = ball
                            break

            if best_ball:
                
                if is_corner_ball(best_ball, field_bounds):
                    staging = create_staging_point_corner(
                        best_ball, field_bounds)
                    if corner_stage == 0:
                        corner_stage = 1
                elif is_edge_ball(best_ball, field_bounds):
                    staging = create_staging_point_edge(
                        best_ball, field_bounds)
                else:
                    staging = None

                if staging:
                    staging_dist = np.linalg.norm(
                        np.array(staging[:2]) - np.array(front_marker))

                    robot_vector = np.array(front_marker) - \
                        np.array(robot_position)
                    ball_vector = np.array(
                        best_ball[:2]) - np.array(robot_position)

                    dot = np.dot(robot_vector, ball_vector)
                    mag_r = np.linalg.norm(robot_vector)
                    mag_b = np.linalg.norm(ball_vector)
                    cos_theta = max(-1, min(1, dot / (mag_r * mag_b + 1e-6)))
                    angle_diff = np.degrees(np.arccos(cos_theta))

                    if (staging_dist < 100):
                        at_staging = True

                    if not at_staging:
                        staged_balls.append(staging)
                        best_ball = staging  

                dist_to_ball = 0 if best_ball is None else np.linalg.norm(
                    np.array(best_ball[:2]) - np.array(front_marker))

                line1, line2 = draw_lines(front_marker, best_ball, egg, cross)

                cv2.line(frame, 
                        (int(line1[0][0]), int(line1[0][1])), 
                        (int(line1[1][0]), int(line1[1][1])), 
                        (255, 255, 0), 2)

                cv2.line(frame, 
                        (int(line2[0][0]), int(line2[0][1])), 
                        (int(line2[1][0]), int(line2[1][1])), 
                        (0, 255, 255), 2)
                
                if barrier_blocks_path(center_robot, best_ball, egg, cross):
                    y = 0
                    x = 0
                    in_line = is_ball_and_robot_on_line_with_cross(center_robot, best_ball, CROSS_X_MIN, CROSS_X_MAX, CROSS_Y_MIN, CROSS_Y_MAX, CROSS_CENTER)
                    if (in_line == 1):
                        y = ((FIELD_Y_MAX - FIELD_Y_MIN) * 0.50) + FIELD_Y_MIN
                        x = ((FIELD_X_MAX - FIELD_X_MIN) * 0.20) + FIELD_X_MIN
                    elif(in_line == 2):
                        y = ((FIELD_Y_MAX - FIELD_Y_MIN) * 0.20) + FIELD_Y_MIN
                        x = ((FIELD_X_MAX - FIELD_X_MIN) * 0.50) + FIELD_X_MIN
                    elif(in_line == 3):
                        print("fuck")
                        x, y = determine_staging_point(center_robot, best_ball, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX, CROSS_CENTER)
                    
                    staging = (x, y, best_ball[2], best_ball[3])
                    staging_dist = np.linalg.norm(
                        np.array(staging[:2]) - np.array((cm_x, cm_y)))
                    
                    if (staging_dist < 30):
                        at_blocked_staging = True
                    
                    if not at_blocked_staging:                        
                        best_ball = staging  
                        staged_balls.append(best_ball)
                        staged_ball = staging
                
            if close_to_barrier(front_marker, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX) and delivery_stage < 1 and len(ball_positions) > 0:
                movement_command = "stop"
                conn.sendall(movement_command.encode())
                time.sleep(2)
                movement_command = "medium_backward"
                conn.sendall(movement_command.encode())
                time.sleep(1)
                last_command = "medium_backward"

            if corner_stage == 1:
                print("Corner stage 1")
                # Naviger til staging
                movement_command = determine_direction(robot_info, staging, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX)
                if np.hypot(fx - staging[0], fy - staging[1]) < 50:
                    command = "delivery"
                    conn.sendall(command.encode())
                    last_command = "delivery"
                    corner_stage = 2
                    print("Next corner stage")
                elif movement_command != last_command:
                    conn.sendall(movement_command.encode())
                    last_command = movement_command
                print(f"Command: {movement_command}")

            elif corner_stage == 2:
                # Roter korrekt mod bolden (ligesom i delivery_stage 2)
                robot_vector = np.array(robot_position) - np.array(front_marker)
                desired_vector = np.array(best_ball[:2]) - np.array(robot_position)

                dot = np.dot(robot_vector, desired_vector)
                mag_r = np.linalg.norm(robot_vector)
                mag_d = np.linalg.norm(desired_vector)
                cos_theta = max(-1, min(1, dot / (mag_r * mag_d + 1e-6)))
                angle_diff = np.degrees(np.arccos(cos_theta))

                print(f"[Corner Stage 2] Angle to target: {angle_diff:.2f}")

                if angle_diff > 0.5:
                    robot_3d = np.append(robot_vector, 0)
                    desired_3d = np.append(desired_vector, 0)
                    cross_product = np.cross(robot_3d, desired_3d)[2]
                    if angle_diff > 30:
                        movement_command = "left"
                    elif angle_diff > 20:
                        movement_command = "medium_left"
                    elif angle_diff > 10:
                        movement_command = "slow_left"
                    else:
                        movement_command = "very_slow_left"
                    if movement_command != last_command:
                        conn.sendall(movement_command.encode())
                        last_command = movement_command
                else:
                    corner_stage = 3
            elif corner_stage == 3:
                print("Corner stage 3")
                if best_ball:
                    bx, by, _, _ = best_ball
                    if np.hypot(rx - bx, ry - by) < 120:
                        robot_vector = np.array(robot_position) - np.array(front_marker)
                        desired_vector = np.array(best_ball[:2]) - np.array(robot_position)

                        dot = np.dot(robot_vector, desired_vector)
                        mag_r = np.linalg.norm(robot_vector)
                        mag_d = np.linalg.norm(desired_vector)
                        cos_theta = max(-1, min(1, dot / (mag_r * mag_d + 1e-6)))
                        angle_diff = np.degrees(np.arccos(cos_theta))

                        print(f"[Corner Stage 3] Angle to target: {angle_diff:.2f}")

                        if angle_diff > 0.5:
                            robot_3d = np.append(robot_vector, 0)
                            desired_3d = np.append(desired_vector, 0)
                            cross_product = np.cross(robot_3d, desired_3d)[2]
                            if angle_diff > 30:
                                movement_command = "left"
                            elif angle_diff > 20:
                                movement_command = "medium_left"
                            elif angle_diff > 10:
                                movement_command = "slow_left"
                            else:
                                movement_command = "very_slow_left"
                            if movement_command != last_command:
                                conn.sendall(movement_command.encode())
                                last_command = movement_command
                        else:
                            corner_stage = 4
                    elif last_command != "slow_backward":
                        conn.sendall(b"slow_backward")
                        last_command = "slow_backward"
                        print("back")
            elif corner_stage == 4:
                print("Corner stage 4")
                if best_ball:
                    bx, by, _, _ = best_ball
                    if np.hypot(rx - bx, ry - by) < 60:
                        stop_command = "stop"
                        conn.sendall(stop_command.encode())  
                        command = "continue"
                        conn.sendall(command.encode())  
                        last_command = "continue"
                        corner_stage = 0
                        corner_ball = None
                        last_delivery_count -= 1
                        print("continue")
                        time.sleep(3)
                        print("after sleep")
                    elif last_command != "slow_backward":
                        conn.sendall(b"slow_backward")
                        last_command = "slow_backward"
                        print("back")
            else:
                movement_command = determine_direction(robot_info, best_ball, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX)

                # Tilføj slow logik - kun forward, left og right bliver slow
                ### DETTE SKAL IMPLEMENTERES ###
                """
                if movement_command in ["left", "right", "forward", "backward", "fast_forward", "fast_left", "fast_right", "fast_backward", "medium_left", "medium_right", "medium_forward", "medium_backward"]:
                    tx, ty = best_ball[:2]
                    fx, fy = front_marker[:2]
                    dist = np.hypot(tx - fx, ty - fy)

                    # Hvis mål er hjørne eller kan, så slow ned
                    if (is_corner_ball(best_ball, field_bounds) or is_edge_ball(best_ball, field_bounds)) and dist < 150:
                        movement_command = "slow_" + movement_command
                """
                # Send kun kun hvis ny kommando
                if movement_command != last_command:
                    print(f"Sending command: {movement_command}")
                    conn.sendall(movement_command.encode())
                    last_command = movement_command

        # --- Draw actual balls in green ---
        # Tegn alle bolde (grøn)
        if (ball_positions):
            for (x, y, r, o) in ball_positions:
                cv2.circle(frame, (x, y), int(r), (0, 255, 0), 2)
                cv2.putText(frame, "Ball", (x - 20, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tegn staging-punkter (lilla)
        if (staged_balls):
            for (x, y, r, o) in staged_balls:
                cv2.circle(frame, (int(x), int(y)), int(r), (255, 0, 255), 2)
                cv2.putText(frame, "Staging", (int(x) - 25, int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
        # Tegn æg (gul)
        if (egg):
            for (ex, ey, er, _) in egg:
                cv2.circle(frame, (ex, ey), int(er), (0, 255, 255), 2)
                cv2.putText(frame, "Egg", (ex - 20, ey - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if (cross):
            for (x1, y1, x2, y2) in cross:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.putText(frame, "Cross", (cx - 15, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        if (barriers):
            for (x1, y1, x2, y2), center in barriers:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cx, cy = center
                cv2.putText(frame, "Barrier", (cx - 20, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if robot_info:
            cv2.circle(frame, (rx, ry), 10, (255, 0, 0), 2)
            cv2.putText(frame, "Back", (rx - 20, ry - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.circle(frame, (fx, fy), 10, (0, 165, 255), 2)
            cv2.putText(frame, "Front", (fx - 20, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            cv2.arrowedLine(frame, (rx, ry), (fx, fy), (0, 255, 0), 2)

    cv2.imshow("Ball & Robot Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# movement_command = "quit"
# conn.sendall(movement_command.encode())
cap.release()
cv2.destroyAllWindows()
conn.close()
server_socket.close()
