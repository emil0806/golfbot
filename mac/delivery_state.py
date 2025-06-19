import globals_config as g
from mac.robot_controller import RobotController
import numpy as np
from mac.robot_state import RobotState
from pathfinding import (bfs_path, determine_direction, find_best_ball, get_cross_zones, get_zone_center, get_zone_for_position, sort_balls_by_distance,
    is_corner_ball, is_edge_ball, create_staging_point_corner, create_staging_point_edge,
    barrier_blocks_path, close_to_barrier, set_homography, determine_staging_point, is_ball_and_robot_on_line_with_cross, is_ball_in_cross, draw_lines, determine_staging_point_16, determine_zone)
from vision import detect_balls, detect_robot, detect_barriers, detect_egg, detect_cross, inside_field, filter_barriers_inside_field, stabilize_detections
import time
        
def handle_delivery(robot_info, egg, cross, ball_positions, controller: RobotController):
    front_marker, center_marker, back_marker, _ = robot_info
    cross_x, cross_y = g.get_cross_center

    if controller.delivery_stage == 0:
        controller.delivery_active = True
        controller.delivery_stage = 1
        return RobotState.DELIVERY

    elif controller.delivery_stage == 1:
        target_ball = controller.goal_first_target
        bx, by = target_ball[:2]

        cx = (front_marker[0] + back_marker[0]) / 2
        cy = (front_marker[1] + back_marker[1]) / 2
        dist_to_first_target = np.linalg.norm(
            np.array((cx, cy)) - np.array(controller.goal_first_target))
        print(f"[Stage 1] Distance to staging: {dist_to_first_target:.2f}")
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
                movement_command = determine_direction(robot_info, next_target)
                controller.send_command(movement_command)
            return RobotState.DELIVERY
        else:
            if dist_to_first_target < 20:
                controller.delivery_stage = 2
                controller.path_to_target = None
            return RobotState.DELIVERY

    elif controller.delivery_stage == 2:
        robot_vector = np.array(
            back_marker) - np.array(front_marker)
        desired_vector = np.array(
            controller.goal_back_alignment_target) - np.array(back_marker)

        dot = np.dot(robot_vector, desired_vector)
        mag_r = np.linalg.norm(robot_vector)
        mag_d = np.linalg.norm(desired_vector)
        cos_theta = max(-1, min(1, dot / (mag_r * mag_d + 1e-6)))
        angle_diff = np.degrees(np.arccos(cos_theta))

        print(f"[Stage 2] Angle to target: {angle_diff:.2f}")

        if angle_diff > 1.5:
            robot_3d = np.append(robot_vector, 0)
            desired_3d = np.append(desired_vector, 0)
            cross_product = np.cross(robot_3d, desired_3d)[
                2] 
            if angle_diff > 25:
                movement_command = "left"
            elif angle_diff > 15:
                movement_command = "medium_left"
            else:
                movement_command = "slow_left"
            controller.send_command(movement_command)
            return RobotState.DELIVERY
        else:
            controller.delivery_stage = 3
            return RobotState.DELIVERY

    elif controller.delivery_stage == 3:
        dist_back = np.linalg.norm(
            np.array(back_marker) - np.array(controller.goal_back_alignment_target))
        print(f"[Stage 3] Distance to back_alignment: {dist_back:.2f}")
        if dist_back > 95:
            movement_command = "slow_backward"
            controller.send_command(movement_command)
        else:
            controller.delivery_stage = 4
        return RobotState.DELIVERY

    elif controller.delivery_stage == 4:
        print("[Stage 4] Sending delivery command")
        movement_command = "delivery"
        controller.delivery_stage = 0 
        controller.delivery_active = False
        controller.last_delivery_count = len(ball_positions)
        controller.waiting_for_continue = True
        controller.send_command(movement_command)
        return RobotState.CORNER
    
    return RobotState.DELIVERY