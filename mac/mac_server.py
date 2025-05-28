import socket
import time
import cv2
from pathfinding import determine_direction, find_best_ball, sort_balls_by_distance, is_corner_ball, is_edge_ball, create_staging_point_corner, create_staging_point_edge, egg_blocks_path, create_staging_point_egg, delivery_routine, stop_delivery_routine, barrier_blocks_path
import numpy as np
from vision import detect_balls, detect_robot, detect_barriers, detect_egg, detect_cross
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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error, no frame captured")
        continue
    
    robot_info = detect_robot(frame)

    if robot_info:
        robot_position, front_marker, direction = robot_info

        if barrier_call == 0:
            barriers = detect_barriers(frame, robot_position)
            cross = detect_cross(frame, robot_position)
            egg = detect_egg(frame)
            barrier_call = 1
       
       
        rx, ry = robot_position  
        fx, fy = front_marker 

        current_time = time.time()

        if current_time - timer >= 1:
            egg = detect_egg(frame)
            ball_positions = detect_balls(frame, egg, robot_position)
            timer = current_time

        COLLECTION_RADIUS = 20
        ball_positions = [
            (x, y, r, o) for (x, y, r, o) in ball_positions
            if np.linalg.norm(np.array((x, y)) - np.array((rx, ry))) > COLLECTION_RADIUS
        ]

        # Main loop
        if len(ball_positions) > 11:
            command = delivery_routine(robot_info)
            if command != last_command:
                conn.sendall(command.encode())
                last_command = command
                
                time.sleep(5) 

                command = stop_delivery_routine()
                conn.sendall(command.encode())
                last_command = command
            continue

        pre_sorted_balls = sort_balls_by_distance(ball_positions, front_marker)
        best_ball = pre_sorted_balls[0] if pre_sorted_balls else None
        staged_balls = []

        if best_ball:
            # Lav staging-punkt hvis bolden er i hjørne eller ved kant
            if is_corner_ball(best_ball):
                staging = create_staging_point_corner(best_ball)
            elif is_edge_ball(best_ball):
                staging = create_staging_point_edge(best_ball)
            else:
                staging = None

            # Æg-undvigelse
            if staging:
                # Check afstand og vinkel til staging
                staging_dist = np.linalg.norm(np.array(staging[:2]) - np.array(front_marker))
                ball_dist = np.linalg.norm(np.array(best_ball[:2]) - np.array(front_marker))

                # Vinkel mellem robot og bold
                robot_vector = np.array(front_marker) - np.array(robot_position)
                ball_vector = np.array(best_ball[:2]) - np.array(robot_position)

                dot = np.dot(robot_vector, ball_vector)
                mag_r = np.linalg.norm(robot_vector)
                mag_b = np.linalg.norm(ball_vector)
                cos_theta = max(-1, min(1, dot / (mag_r * mag_b + 1e-6)))
                angle_diff = np.degrees(np.arccos(cos_theta))

                if (staging_dist > 80 and angle_diff > 10) or ball_dist > 120:
                    # Erstat best_ball med staging
                    staged_balls.append(staging)
                    best_ball = staging  # overskriv best_ball med staging-punktet
            
            dist_to_staged_ball = 0 if staged_ball is None else np.linalg.norm(np.array(staged_ball[:2]) - np.array(robot_position))
            

            if barrier_blocks_path(robot_position, best_ball, egg, cross):
                y = 0
                x = 0 
                if(robot_position[1] > 250 and robot_position[1] < 750 and best_ball[1] > 250 and best_ball[1] < 750):
                    if(robot_position[1] <= 550):
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

        movement_command = determine_direction(robot_info, best_ball)

        if movement_command != last_command:
            print(f"Sending command:  {movement_command}")

            conn.sendall(movement_command.encode()) 
            last_command = movement_command

        # --- Draw actual balls in green ---
        # Tegn alle bolde (grøn)
        for (x, y, r, o) in ball_positions:
            cv2.circle(frame, (x, y), int(r), (0, 255, 0), 2)
            cv2.putText(frame, "Ball", (x - 20, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tegn staging-punkter (lilla)
        for (x, y, r, o) in staged_balls:
            cv2.circle(frame, (x, y), int(r), (255, 0, 255), 2)
            cv2.putText(frame, "Staging", (x - 25, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Tegn æg (gul)
        for (ex, ey, er, _) in egg:
            cv2.circle(frame, (ex, ey), int(er), (0, 255, 255), 2)
            cv2.putText(frame, "Egg", (ex - 20, ey - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
        for (x1, y1, x2, y2) in cross:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(frame, "Barrier", (cx - 15, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        for (x1, y1, x2, y2), center in barriers:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cx, cy = center
            cv2.putText(frame, "Barrier", (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    

        if robot_info:
            cv2.circle(frame, (rx, ry), 10, (255, 0, 0), 2)
            cv2.putText(frame, "Back", (rx - 20, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.circle(frame, (fx, fy), 10, (0, 165, 255), 2) 
            cv2.putText(frame, "Front", (fx - 20, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            cv2.arrowedLine(frame, (rx, ry), (fx, fy), (0, 255, 0), 2) 

    cv2.imshow("Ball & Robot Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#movement_command = "quit"
#conn.sendall(movement_command.encode())
cap.release()
cv2.destroyAllWindows()
conn.close()
server_socket.close()
