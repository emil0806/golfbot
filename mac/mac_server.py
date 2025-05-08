import socket
import time
import cv2
from pathfinding import determine_direction, find_best_ball, sort_balls_by_distance, is_corner_ball, is_edge_ball, create_staging_point_corner, create_staging_point_edge, delivery_routine
import numpy as np
from vision import detect_balls, detect_robot, detect_barriers
from config import EV3_IP, PORT


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

        if current_time - timer >= 2:
            ball_positions = detect_balls(frame)
            barriers = detect_barriers(frame)

            COLLECTION_RADIUS = 20
            ball_positions = [
                (x, y, r, o) for (x, y, r, o) in ball_positions
                if np.linalg.norm(np.array((x, y)) - np.array((rx, ry))) > COLLECTION_RADIUS
            ]

            if len(ball_positions) not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                command = delivery_routine(robot_info)
                if command != last_command:
                    conn.sendall(command.encode())
                    last_command = command
                continue

            staged_balls = []
            for ball in ball_positions:
                if is_corner_ball(ball, barriers):
                    staged_balls.append(create_staging_point_corner(ball, barriers))
                elif is_edge_ball(ball, barriers):
                    staged_balls.append(create_staging_point_edge(ball, barriers))
                else:
                    staged_balls.append(ball)

            sorted_balls = sort_balls_by_distance(ball_positions, front_marker)
            best_ball = sorted_balls[0] if sorted_balls else None

            movement_command = determine_direction(robot_info, best_ball)

            if movement_command != last_command:
                print(f"Sending command:  {movement_command}")

                conn.sendall(movement_command.encode()) 
                last_command = movement_command

            for (x, y, r, o) in ball_positions: 
                cv2.circle(frame, (x, y), int(r), (0, 255, 0), 2)  
                cv2.putText(frame, "Ball", (x - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
