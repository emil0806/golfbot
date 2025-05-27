import cv2
import time
from vision import detect_balls, detect_robot, detect_barriers
from pathfinding import find_best_ball, determine_direction

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

    if time.time() - last_print_time >= 5:
        if(robot_info):
            robot_position, front_marker, direction = robot_info
            best_ball = find_best_ball(ball_positions, robot_position, front_marker)
            movement_command = determine_direction(robot_info, best_ball)
        print("Robot info:", robot_info)
        print("Ball position:", ball_positions)
        last_print_time = time.time()

    for (x, y, radius, color) in ball_positions:
        if color == 0:  # white ball
            circle_color = (0, 255, 0)  # green
            label = "White"
        elif color == 1:  # orange ball
            circle_color = (255, 0, 0)  # blue
            label = "Orange"
        else:
            circle_color = (0, 0, 0)
            label = "Unknown"

        cv2.circle(frame, (x, y), int(radius), circle_color, 2)
        cv2.putText(frame, label, (x - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, circle_color, 2)


    if robot_info:
        robot_position, front_marker, direction = robot_info
        rx, ry = robot_position  
        fx, fy = front_marker 

        cv2.circle(frame, (rx, ry), 10, (255, 0, 0), 2)
        cv2.putText(frame, "Robot", (rx - 20, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.circle(frame, (fx, fy), 10, (0, 165, 255), 2) 
        cv2.putText(frame, "Front", (fx - 20, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        cv2.arrowedLine(frame, (rx, ry), (fx, fy), (0, 255, 0), 2) 
    
    for (x, y, w, h) in barriers:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Barrier", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    
    cv2.imshow("Image Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
