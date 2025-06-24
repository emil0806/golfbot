from collections import deque
import cv2
from pathfinding import set_homography
import numpy as np
from vision import detect_balls, detect_robot, detect_egg, detect_cross, inside_field
import time
import globals_config as g

def setup_cross_lines(cap, last_robot_info):
    cross_lines = []
    barrier_call = 0

    while barrier_call < 8:
        ret, frame = cap.read()
        if not ret:
            print("Camera error, no frame captured")
            continue

        frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=0)

        robot_info = detect_robot(frame)
        
        if robot_info:
            last_robot_info = robot_info
        else:
            robot_info = last_robot_info
        
        front_marker, center_marker, back_marker, _ = robot_info

        if robot_info:
            egg = detect_egg(frame, back_marker, front_marker)
            ball_positions = detect_balls(frame, egg, back_marker, front_marker)

            cross_line = detect_cross(
                frame,
                back_marker,
                front_marker,
                ball_positions,
            )
            cross_lines.append(cross_line)

        barrier_call += 1

    flat_cross = [c for sublist in cross_lines for c in sublist]

    g.extract_cross_lines(flat_cross)


def setup_homography():
    FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX = g.get_field_bounds()

    PIX_CORNERS = np.float32([
        [FIELD_X_MIN, FIELD_Y_MIN],   
        [FIELD_X_MAX, FIELD_Y_MIN],   
        [FIELD_X_MAX, FIELD_Y_MAX],   
        [FIELD_X_MIN, FIELD_Y_MAX]    
    ])

    FIELD_W, FIELD_H = 1800, 1200
    WORLD_CORNERS = np.float32([
        [0,        0],
        [FIELD_W,  0],
        [FIELD_W,  FIELD_H],
        [0,        FIELD_H]
    ])

    H, _ = cv2.findHomography(PIX_CORNERS, WORLD_CORNERS)

    return H

def setup_field_lines():
    g.set_field_lines_from_corners()
    g.set_field_bounds_by_corners()