import socket
import time
import cv2
from pathfinding import determine_direction, find_best_ball, sort_balls_by_distance, is_corner_ball, is_edge_ball, create_staging_point_corner, create_staging_point_edge, egg_blocks_path, create_staging_point_egg, delivery_routine, stop_delivery_routine, barrier_blocks_path, close_to_barrier
import numpy as np
from vision import detect_balls, detect_robot, detect_barriers, detect_egg, detect_cross, inside_field
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
staged_ball = None
delivery_stage = 0  
last_delivery_count = 0

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
    egg = detect_egg(frame)
    ball_positions = detect_balls(frame, egg)

    if robot_info:
        robot_position, front_marker, direction = robot_info

        barriers.append(detect_barriers(frame, robot_position, ball_positions))
        cross.append(detect_cross(frame, robot_position, front_marker, ball_positions))
        barrier_call += 1

if barriers:
    flat_barriers = [b for sublist in barriers for b in sublist]
    FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX = inside_field(flat_barriers)
    barriers = flat_barriers

else:
    barriers = []
    FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX = 0, frame.shape[
        1], 0, frame.shape[0]
    
if cross:
    flat_cross = [c for sublist in cross for c in sublist]
    cross = flat_cross

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error, no frame captured")
        continue

    robot_info = detect_robot(frame)

    if robot_info:
        robot_position, front_marker, direction = robot_info

        rx, ry = robot_position  
        fx, fy = front_marker 

        current_time = time.time()

        egg = detect_egg(frame)
        ball_positions = detect_balls(frame, egg)
        ball_positions = [(x, y, r, o) for (
            x, y, r, o) in ball_positions if FIELD_X_MIN < x < FIELD_X_MAX and FIELD_Y_MIN < y < FIELD_Y_MAX]
        timer = current_time

        COLLECTION_RADIUS = 20
        ball_positions = [
            (x, y, r, o) for (x, y, r, o) in ball_positions
            if np.linalg.norm(np.array((x, y)) - np.array((rx, ry))) > COLLECTION_RADIUS
        ]
        
        staged_balls = []

        ###   Delivery   ###
        if (len(ball_positions) == 0):
            if delivery_stage == 0:
                print("Initiating delivery routine...")
                delivery_stage = 1

                staging_target = (FIELD_X_MAX - 200, (FIELD_Y_MIN + FIELD_Y_MAX) // 2)
                back_alignment_target = (FIELD_X_MAX - 20, (FIELD_Y_MIN + FIELD_Y_MAX) // 2)

            if delivery_stage == 1:
                cm_x = (front_marker[0] + robot_position[0]) / 2
                cm_y = (front_marker[1] + robot_position[1]) / 2
                dist_to_staging = np.linalg.norm(np.array((cm_x, cm_y)) - np.array(staging_target))
                print(f"[Stage 1] Distance to staging: {dist_to_staging:.2f}")
                if dist_to_staging > 50:
                    dummy_target = (*staging_target, 10, (255, 255, 255))
                    movement_command = determine_direction(robot_info, dummy_target)
                    if movement_command != last_command:
                        conn.sendall(movement_command.encode())
                        last_command = movement_command
                else:
                    delivery_stage = 2

            if delivery_stage == 2:
                robot_vector = np.array(robot_position) - np.array(front_marker)
                desired_vector = np.array(back_alignment_target) - np.array(robot_position)

                dot = np.dot(robot_vector, desired_vector)
                mag_r = np.linalg.norm(robot_vector)
                mag_d = np.linalg.norm(desired_vector)
                cos_theta = max(-1, min(1, dot / (mag_r * mag_d + 1e-6)))
                angle_diff = np.degrees(np.arccos(cos_theta))

                print(f"[Stage 2] Angle to target: {angle_diff:.2f}")

                if angle_diff > 1.5:
                    # Brug 3D vektorer til at finde drejeretning (z-komponenten af krydsprodukt)
                    robot_3d = np.append(robot_vector, 0)
                    desired_3d = np.append(desired_vector, 0)
                    cross_product = np.cross(robot_3d, desired_3d)[2]  # kun Z-aksen er relevant
                    print(f"cross product: {cross_product: .2f}")
                    if angle_diff > 15:
                        movement_command = "left"
                    else:
                        movement_command = "slow_left"
                    if movement_command != last_command:
                        conn.sendall(movement_command.encode())
                        last_command = movement_command
                else:
                    delivery_stage = 3

            if delivery_stage == 3:
                dist_back = np.linalg.norm(np.array(robot_position) - np.array(back_alignment_target))
                print(f"[Stage 3] Distance to back_alignment: {dist_back:.2f}")
                if dist_back > 50:
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
                time.sleep(3)
                command = "continue"
                if command != last_command:
                    print("[Stage 4.2] Sending continue command")
                    conn.sendall(command.encode())
                    last_command = command
                delivery_stage = 0  # reset
                last_delivery_count = len(ball_positions)

        else:
            pre_sorted_balls = sort_balls_by_distance(ball_positions, front_marker)
            best_ball = pre_sorted_balls[0] if pre_sorted_balls else None

            if best_ball:
                # Lav staging-punkt hvis bolden er i hjørne eller ved kant
                field_bounds = (FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX)

                if is_corner_ball(best_ball, field_bounds):
                    staging = create_staging_point_corner(best_ball, field_bounds)
                elif is_edge_ball(best_ball, field_bounds):
                    staging = create_staging_point_edge(best_ball, field_bounds)
                else:
                    staging = None

                # Æg-undvigelse
                if staging:
                    # Check afstand og vinkel til staging
                    staging_dist = np.linalg.norm(
                        np.array(staging[:2]) - np.array(front_marker))
                    ball_dist = np.linalg.norm(
                        np.array(best_ball[:2]) - np.array(front_marker))

                    # Vinkel mellem robot og bold
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
                        # Erstat best_ball med staging
                        staged_balls.append(staging)
                        best_ball = staging  # overskriv best_ball med staging-punktet

                dist_to_staged_ball = 0 if staged_ball is None else np.linalg.norm(
                    np.array(staged_ball[:2]) - np.array(robot_position))

                if barrier_blocks_path(robot_position, best_ball, egg, cross):
                    y = 0
                    x = 0
                    if (robot_position[1] > 250 and robot_position[1] < 750 and best_ball[1] > 250 and best_ball[1] < 750):
                        if (robot_position[1] <= 550):
                            y = 200
                            x = 950
                        else:
                            y = 800
                            x = 950
                    else:
                        y = robot_position[1]
                        x = best_ball[0]
                    # Lav stagingpunkt (fx direkte vertikal med robotens x og boldens y)
                    staging = (x, y, best_ball[2], best_ball[3])
                    best_ball = staging  # brug stagingpunkt som mål
                    staged_balls.append(best_ball)
                    staged_ball = staging
                    has_staging = True
                elif(has_staging and dist_to_staged_ball > 50):
                    staging = (best_ball[0], robot_position[1], best_ball[2], best_ball[3])
                    best_ball = staging  # brug stagingpunkt som mål
                    staged_balls.append(best_ball)
                    staged_ball = staging
                    has_staging = True
                else:
                    has_staging = False
                    staged_ball = None

            if close_to_barrier(front_marker, FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX) and delivery_stage < 1:
                movement_command = "stop"
                conn.sendall(movement_command.encode())
                time.sleep(3)
                movement_command = "backward"
                conn.sendall(movement_command.encode())
                time.sleep(2)
                last_command = "backward"

            movement_command = determine_direction(robot_info, best_ball)

            if movement_command != last_command:
                print(f"Sending command:  {movement_command}")

                conn.sendall(movement_command.encode())
                last_command = movement_command

        # --- Draw actual balls in green ---
        # Tegn alle bolde (grøn)
        if(ball_positions):
            for (x, y, r, o) in ball_positions:
                cv2.circle(frame, (x, y), int(r), (0, 255, 0), 2)
                cv2.putText(frame, "Ball", (x - 20, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tegn staging-punkter (lilla)
        if(staged_balls):
            for (x, y, r, o) in staged_balls:
                cv2.circle(frame, (x, y), int(r), (255, 0, 255), 2)
                cv2.putText(frame, "Staging", (x - 25, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Tegn æg (gul)
        if (egg):
            for (ex, ey, er, _) in egg:
                cv2.circle(frame, (ex, ey), int(er), (0, 255, 255), 2)
                cv2.putText(frame, "Egg", (ex - 20, ey - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if(cross):
            for (x1, y1, x2, y2) in cross:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.putText(frame, "Barrier", (cx - 15, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        if(barriers):
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
