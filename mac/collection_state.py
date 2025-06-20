from collections import deque
import socket
import time
import cv2
from robot_controller import RobotController
from robot_state import RobotState
from pathfinding import (bfs_path, determine_direction, get_cross_zones, get_simplified_path, get_simplified_target, get_zone_center, get_zone_for_position, sort_balls_by_distance,
    is_corner_ball, is_edge_ball, create_staging_point_corner, create_staging_point_edge)
import numpy as np
from config import EV3_IP, PORT
import time

import globals_config as g

def handle_collection(robot_info, ball_positions, egg, cross, controller: RobotController):
    front_marker, center_marker, back_marker, _ = robot_info
    cx, cy = center_marker

    corner_balls = [b for b in ball_positions if is_corner_ball(b)]
    print(f"balls: {ball_positions}")

    if(len(ball_positions) in [0, 4, 8]):
        return RobotState.DELIVERY
    else:
        if len(corner_balls) != len(ball_positions):
            filtered_balls = [b for b in ball_positions if not is_corner_ball(b)]
        else:
            filtered_balls = ball_positions

    pre_sorted_balls = sort_balls_by_distance(filtered_balls, front_marker)
    original_ball = pre_sorted_balls[0]
    ox, oy = original_ball[:2]

    original_dist = np.linalg.norm(np.array([cx, cy]) - np.array([ox, oy]))

    if(len(ball_positions) != controller.last_ball_count and original_dist < 20):
        controller.simplified_path = None
        controller.last_ball_count = len(ball_positions)

    if is_edge_ball(original_ball):
        target_ball = create_staging_point_edge(original_ball)
        controller.edge_alignment_active = True
    else:
        target_ball = original_ball
        controller.edge_alignment_active = False
    
    bx, by = target_ball[:2]

    recalculate = controller.simplified_path is None

    if controller.simplified_path and len(controller.simplified_path) > 1:
        zx, zy = controller.simplified_path[1][:2]
        dist = np.linalg.norm(np.array([cx, cy]) - np.array([zx, zy]))
        if dist < 20:
            controller.simplified_path.pop(0)
            recalculate = True

    if recalculate:
        robot_zone = get_zone_for_position(cx, cy)
        ball_zone = get_zone_for_position(bx, by)
        forbidden_zones = get_cross_zones()

        path = bfs_path(robot_zone, ball_zone, forbidden_zones)

        if path and len(path) > 1:
            controller.simplified_path = get_simplified_path(path, center_marker, target_ball, egg, cross)
        else:
            controller.simplified_path = None
            return
        
    if controller.simplified_path:
        # Brug eksisterende simplified_path indtil bolden nås
        if len(controller.simplified_path) >= 2:
            next_target = controller.simplified_path[1]
        else:
            next_target = controller.simplified_path[0]

        zx, zy = next_target[:2]

        dist = np.linalg.norm(np.array([cx, cy]) - np.array([zx, zy]))
        if dist < 20 and len(controller.simplified_path) > 1:
            controller.simplified_path.pop(0)
            # simplified_path opdateres IKKE her – behold indtil målet

        controller.current_target = next_target
        command = determine_direction(robot_info, next_target)
        controller.send_command(command)

    if controller.edge_alignment_active and controller.simplified_path == []:
        robot_zone = get_zone_for_position(cx, cy)
        ball_zone = get_zone_for_position(original_ball[0], original_ball[1])
        forbidden_zones = get_cross_zones()

        path = bfs_path(robot_zone, ball_zone, forbidden_zones)
        if path and len(path) > 1:
            controller.simplified_path = path[1:]
        controller.edge_alignment_active = False
    
    return RobotState.COLLECTION