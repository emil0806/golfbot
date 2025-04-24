import socket
import cv2
from pathfinding import determine_direction, find_best_ball
import numpy as np
from vision import detect_balls, detect_robot
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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error, no frame captured")
        continue

    ball_positions = detect_balls(frame)
    robot_info = detect_robot(frame)

    if robot_info:
        robot_position, front_marker, direction = robot_info

        rx, ry = robot_position  
        fx, fy = front_marker 

        best_ball = find_best_ball(ball_positions, robot_position, front_marker)

        movement_command = determine_direction(robot_info, best_ball)

        if movement_command != last_command:
            print(f"Sending command:  {movement_command}")
            conn.sendall(movement_command.encode()) 
            last_command = movement_command

        for (x, y, r) in ball_positions: 
            cv2.circle(frame, (x, y), int(r), (0, 255, 0), 2)  
            cv2.putText(frame, "Ball", (x - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if robot_info:
            cv2.circle(frame, (rx, ry), 10, (255, 0, 0), 2)
            cv2.putText(frame, "Robot", (rx - 20, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.circle(frame, (fx, fy), 10, (0, 165, 255), 2) 
            cv2.putText(frame, "Front", (fx - 20, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            cv2.arrowedLine(frame, (rx, ry), (fx, fy), (0, 255, 0), 2) 

    cv2.imshow("Ball & Robot Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
server_socket.close()
