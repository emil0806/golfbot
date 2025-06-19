from collections import deque
import socket
import time
import cv2
from mac.robot_controller import RobotController
from mac.robot_state import RobotState
from pathfinding import (bfs_path, determine_direction, find_best_ball, get_cross_zones, get_zone_center, get_zone_for_position, sort_balls_by_distance,
    is_corner_ball, is_edge_ball, create_staging_point_corner, create_staging_point_edge, 
    barrier_blocks_path, close_to_barrier, set_homography, determine_staging_point, is_ball_and_robot_on_line_with_cross, is_ball_in_cross, draw_lines, determine_staging_point_16, determine_zone)
import numpy as np
from config import EV3_IP, PORT
import time

import globals_config as g

def handle_collection(robot_info, ball_positions, egg, cross, controller: RobotController):
    front_marker, center_marker, back_marker, _ = robot_info
    cx, cy = center_marker

    field_bounds = g.get_field_bounds()
    corner_balls = [b for b in ball_positions if is_corner_ball(b, field_bounds)]

    if(len(ball_positions) in [0, 4, 8]):
        return RobotState.DELIVERY
    else:
        if len(corner_balls) != len(ball_positions):
            filtered_balls = [b for b in ball_positions if not is_corner_ball(b, field_bounds)]
        else:
            filtered_balls = ball_positions

    pre_sorted_balls = sort_balls_by_distance(filtered_balls, front_marker)
    original_ball = pre_sorted_balls[0]
    bx, by = target_ball[:2]

    if is_edge_ball(target_ball, field_bounds):
        target_ball = create_staging_point_edge(target_ball)
        controller.edge_alignment_active = True
    else:
        target_ball = original_ball
        controller.edge_alignment_active = False


    if controller.path_to_target is None or controller.reached_next_path_point(cx, cy):
        robot_zone = get_zone_for_position(cx, cy)
        ball_zone = get_zone_for_position(bx, by)
        forbidden_zones = get_cross_zones()

        path = bfs_path(robot_zone, ball_zone, forbidden_zones)

        if path and len(path) > 1:
            controller.path_to_target = path[1:] 
        else:
            controller.path_to_target = None
            return
        
    if controller.path_to_target:
        next_zone = controller.path_to_target[0]
        zx, zy = get_zone_center(next_zone)
        next_target = (zx, zy, 10, (255, 255, 255)) 

        dist = np.linalg.norm(np.array([cx, cy]) - np.array([zx, zy]))
        if dist < 20:
            controller.reached_path_point = True
            controller.path_to_target.pop(0)
        else:
            controller.reached_path_point = False
            controller.current_target = next_target
            command = determine_direction(robot_info, next_target)
            controller.send_command(command)

    if controller.edge_alignment_active and controller.path_to_target == []:
        robot_zone = get_zone_for_position(cx, cy)
        ball_zone = get_zone_for_position(original_ball[0], original_ball[1])
        forbidden_zones = get_cross_zones()

        path = bfs_path(robot_zone, ball_zone, forbidden_zones)
        if path and len(path) > 1:
            controller.path_to_target = path[1:]
        controller.edge_alignment_active = False
    
    return RobotState.COLLECTION
