import cv2
from vision import detect_balls, detect_robot

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture image")
        continue

    ball_positions = detect_balls(frame)
    robot_info = detect_robot(frame)

    print("Robot info:", robot_info)

    for (x, y, radius) in ball_positions:
        cv2.circle(frame, (x, y), int(radius), (0, 255, 0), 2)
        cv2.putText(frame, "Ball", (x-20, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if robot_info:
        robot_position, front_marker, direction = robot_info
        rx, ry = robot_position  
        fx, fy = front_marker 

        cv2.circle(frame, (rx, ry), 10, (255, 0, 0), 2)
        cv2.putText(frame, "Robot", (rx - 20, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.circle(frame, (fx, fy), 10, (0, 165, 255), 2) 
        cv2.putText(frame, "Front", (fx - 20, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        cv2.arrowedLine(frame, (rx, ry), (fx, fy), (0, 255, 0), 2) 

    
    cv2.imshow("Image Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
