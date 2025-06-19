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

    if flat_cross:
        cx_min, cx_max, cy_min, cy_max = inside_field(flat_cross)
        cross_center = ((cx_min + cx_max) // 2, (cy_min + cy_max) // 2)
        cross_bounds = {
            "x_min": cx_min,
            "x_max": cx_max,
            "y_min": cy_min,
            "y_max": cy_max,
        }
    else:
        cross_bounds = None
        cross_center = None
    g.set_cross_bounds(cross_bounds, cross_center)

    return flat_cross, cross_center, egg, last_robot_info


def setup_homography():
    FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX = g.get_field_bounds()

    PIX_CORNERS = np.float32([
        [FIELD_X_MIN, FIELD_Y_MIN],   # top-left
        [FIELD_X_MAX, FIELD_Y_MIN],   # top-right
        [FIELD_X_MAX, FIELD_Y_MAX],   # bottom-right
        [FIELD_X_MIN, FIELD_Y_MAX]    # bottom-left
    ])

    FIELD_W, FIELD_H = 1800, 1200
    WORLD_CORNERS = np.float32([
        [0,        0],
        [FIELD_W,  0],
        [FIELD_W,  FIELD_H],
        [0,        FIELD_H]
    ])

    H, _ = cv2.findHomography(PIX_CORNERS, WORLD_CORNERS)
    set_homography(H)  # Saves in pathfinding.py
    return H