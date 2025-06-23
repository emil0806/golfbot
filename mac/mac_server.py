from collections import deque
import socket
import time
import cv2
from collection_state import handle_collection
from delivery_state import handle_delivery
from robot_controller import RobotController
from robot_state import RobotState
from pathfinding import draw_lines, get_grid_thresholds, set_homography
import numpy as np
from vision import detect_balls, detect_robot, detect_egg, stabilize_detections
from setup import setup_cross_lines, setup_homography, setup_field_lines
from config import EV3_IP, PORT
import time
import traceback

import globals_config as g

# Initialize socket server to send data to EV3
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", PORT))
server_socket.listen(1)

print(f"Server lytter på 0.0.0.0:{PORT}...")

print(f"Waiting for EV3 connection on port {PORT}...")

conn, addr = server_socket.accept()
print(f"Connection established with EV3 at {addr}")

### CAMERA FEED ###
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

### CONTROLLER ###
controller = RobotController(conn)

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
    setup_field_lines()
    setup_cross_lines(cap, last_robot_info)
    controller.set_delivery_targets()
    H = setup_homography()
    set_homography(H, frame_width, frame_height)
except Exception as e:
    print(f"[ERROR] Programmet stødte på en fejl: {e}")
    traceback.print_exc()
    time.sleep(1) 

controller.state = RobotState.COLLECTION


while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Camera error, no frame captured")
            continue
        frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=0)

        robot_info = detect_robot(frame) or last_robot_info

        if robot_info:
            last_robot_info = robot_info
        if not robot_info:
            print("Robot not detected - skipping this frame")
            continue

        front_marker, center_marker, back_marker, _ = robot_info

        fx, fy = front_marker
        cx, cy = center_marker
        rx, ry = back_marker

        egg = detect_egg(frame, back_marker, front_marker)
        
        current_balls = detect_balls(frame, egg, back_marker, front_marker)
                
        stable_balls = stabilize_detections(current_balls, back_marker)

        ball_positions = stable_balls
        controller.delivery_counter += 1

        if controller.state == RobotState.COLLECTION:
            print("test")
            new_state = handle_collection(robot_info, ball_positions, egg, cross, controller)
        elif controller.state == RobotState.DELIVERY:
            new_state = handle_delivery(robot_info, ball_positions, egg, cross, controller)
        elif controller.state == RobotState.COMPLETE:
            break

        controller.update_state(new_state)

        if controller.current_target is not None:
            line1, line2 = draw_lines(front_marker, controller.current_target, egg, cross)
            cv2.line(frame, (int(line1[0][0]), int(line1[0][1])), (int(line1[1][0]), int(line1[1][1])), (255, 255, 0), 2)
            cv2.line(frame, (int(line2[0][0]), int(line2[0][1])), (int(line2[1][0]), int(line2[1][1])), (0, 255, 255), 2)

            cv2.line(frame, 
                    (int(line1[0][0]), int(line1[0][1])), 
                    (int(line1[1][0]), int(line1[1][1])), 
                    (255, 255, 0), 2)

            cv2.line(frame, 
                    (int(line2[0][0]), int(line2[0][1])), 
                    (int(line2[1][0]), int(line2[1][1])), 
                    (0, 255, 255), 2)
        
        # --- Draw actual balls in green ---
        # Tegn alle bolde (grøn)
        if (ball_positions):
            for (x, y, r, o) in ball_positions:
                cv2.circle(frame, (x, y), int(r), (0, 255, 0), 2)
                cv2.putText(frame, "Ball", (x - 20, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tegn staging-punkter (lilla)
        if (staged_balls and len(staged_balls) > 0):
            for ball in staged_balls:
                if ball is not None:
                    x, y, r, o = ball
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
            (fx, fy),(cx, cy), (rx, ry), _ = robot_info
            cv2.circle(frame, (rx, ry), 10, (255, 0, 0), 2)
            cv2.putText(frame, "Back", (rx - 20, ry - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.circle(frame, (fx, fy), 10, (0, 165, 255), 2)
            cv2.putText(frame, "Front", (fx - 20, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            cv2.arrowedLine(frame, (rx, ry), (fx, fy), (0, 255, 0), 2)

            center_x = int((fx + rx) / 2)
            center_y = int((fy + ry) / 2)
            rotation_radius = 130

            cv2.circle(frame, (center_x, center_y), rotation_radius, (255, 255, 255), 2)

        x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6 = get_grid_thresholds()
        
        for x in [x1, x2, x3, x4, x5, x6]:
            cv2.line(frame, (int(x), int(g.FIELD_Y_MIN)), (int(x), int(g.FIELD_Y_MAX)), (255, 255, 0), 2)
        for y in [y1, y2, y3, y4, y5, y6]:
            cv2.line(frame, (int(g.FIELD_X_MIN), int(y)), (int(g.FIELD_X_MAX), int(y)), (255, 255, 0), 2)

        if hasattr(controller, 'simplified_path') and controller.simplified_path:
            path_points = [(cx, cy)] + controller.simplified_path

            for i in range(len(path_points) - 1):
                x1, y1 = path_points[i][:2]
                x2, y2 = path_points[i + 1][:2]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        # Tegn FIELD_LINES (blå)
        for x1, y1, x2, y2 in g.get_field_lines():
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(frame, "Field", (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
        # Tegn FIELD_LINES (blå)
        for x1, y1, x2, y2 in g.get_cross_lines():
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(frame, "Field", (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)


        cv2.imshow("Ball & Robot Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as e:
        print(f"[ERROR] Programmet stødte på en fejl: {e}")
        traceback.print_exc()
        try:
            command = "stop"
            conn.sendall(command.encode())
        except:
            print("Kunne ikke sende stop-kommando til robot.")
        time.sleep(1)  # Undgå at spamme CPU og console hvis fejl sker i loop
        continue

# movement_command = "quit"
# conn.sendall(movement_command.encode())
cap.release()
cv2.destroyAllWindows()
conn.close()
server_socket.close()
