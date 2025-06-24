from collections import deque
import socket
import time
import cv2
from robot_controller import RobotController
from robot_state import RobotState
from pathfinding import (barrier_blocks_path, bfs_path, determine_direction, get_cross_zones, get_simplified_path, get_zone_center, get_zone_for_position, is_ball_in_cross, sort_balls_by_distance,
    is_corner_ball, is_edge_ball, create_staging_point_corner, create_staging_point_edge, create_staging_point_cross)
import numpy as np
from config import EV3_IP, PORT
import time

import globals_config as g

def handle_collection(robot_info, ball_positions, egg, cross, controller: RobotController):
    front_marker, center_marker, back_marker, _ = robot_info
    cx, cy = center_marker


    if(len(ball_positions) < 1 and controller.delivery_counter > 20):
        return RobotState.DELIVERY

    pre_sorted_balls = sort_balls_by_distance(ball_positions, front_marker)

    if controller.path_counter >= len(pre_sorted_balls) and controller.delivery_counter > 20:
        return RobotState.DELIVERY
    
    original_ball = pre_sorted_balls[controller.path_counter]

    if((len(ball_positions) != controller.last_ball_count)):
        controller.simplified_path = None
        controller.last_ball_count = len(ball_positions)
        controller.edge_staging_reached = False
        controller.corner_staging_reached = False
        controller.cross_staging_reached = False
    
    if is_corner_ball(original_ball):
        if controller.corner_staging_reached:
            target_ball = original_ball
        else:
            target_ball = create_staging_point_corner(original_ball)
    elif is_edge_ball(original_ball):
        if controller.edge_staging_reached:
            target_ball = original_ball
        else:
            target_ball = create_staging_point_edge(original_ball)
    elif is_ball_in_cross(original_ball):
        if controller.cross_staging_reached:
            target_ball = original_ball
        else:
            target_ball = create_staging_point_cross(original_ball)
    else:
        target_ball = original_ball

    bx, by, _, _ = target_ball

    recalculate = controller.simplified_path is None

    if controller.simplified_path and len(controller.simplified_path) > 1:
        zx, zy = controller.simplified_path[0][:2]
        dist = np.linalg.norm(np.array([cx, cy]) - np.array([zx, zy]))
        if dist < 80:
            controller.simplified_path.pop(0)
            recalculate = True

    if recalculate:
        robot_zone = get_zone_for_position(cx, cy)
        ball_zone = get_zone_for_position(bx, by)

        path = bfs_path(robot_zone, ball_zone, egg, ball_position=target_ball[:2])

        if path:
            controller.path_counter = 0
            simplified = get_simplified_path(path, center_marker, target_ball, egg)

            if is_edge_ball(original_ball) and not controller.edge_staging_reached:
                simplified.append(original_ball[:2])
            if is_corner_ball(original_ball) and not controller.corner_staging_reached:
                simplified.append(original_ball[:2])
            if is_ball_in_cross(original_ball) and not controller.cross_staging_reached:
                simplified.append(original_ball[:2])

            controller.simplified_path = simplified
        else:
            print("no path")
            controller.simplified_path = None
            controller.path_counter += 1
            return

        
    if controller.simplified_path:
        next_target = controller.simplified_path[0]

        zx, zy = next_target[:2]

        dist = np.linalg.norm(np.array([cx, cy]) - np.array([zx, zy]))
        if dist < 50 and len(controller.simplified_path) > 1:
            controller.simplified_path.pop(0)

        controller.current_target = next_target
        command = determine_direction(robot_info, next_target, cross)
        controller.send_command(command)
    
    return RobotState.COLLECTION