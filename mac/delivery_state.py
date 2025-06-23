import globals_config as g
from robot_controller import RobotController
import numpy as np
from robot_state import RobotState
from pathfinding import (bfs_path, determine_direction, get_cross_zones, get_simplified_path, get_zone_center, get_zone_for_position)
import time
        
def handle_delivery(robot_info, ball_positions, egg, cross, controller: RobotController):
    front_marker, center_marker, back_marker, _ = robot_info

    if controller.last_delivery_count == 0:
        print("[Delivery] Ingen flere bolde tilbage - stopper.")
        controller.delivery_stage = 0
        controller.delivery_active = False
        controller.waiting_for_continue = False
        return RobotState.COMPLETE

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
        
        recalculate = controller.simplified_path is None

        if controller.simplified_path and len(controller.simplified_path) > 1:
            zx, zy = controller.simplified_path[0][:2]
            dist = np.linalg.norm(np.array([cx, cy]) - np.array([zx, zy]))
            if dist < 40:
                controller.simplified_path.pop(0)
                recalculate = True

        if recalculate:
            robot_zone = get_zone_for_position(cx, cy)
            ball_zone = get_zone_for_position(bx, by)

            path = bfs_path(robot_zone, ball_zone, egg, cross)

            if path:
                simplified = get_simplified_path(path, center_marker, target_ball, egg, cross)
                print(f"[Stage 1] Simplified path: {simplified}")

                controller.simplified_path = simplified
            else:
                controller.simplified_path = None
                return RobotState.DELIVERY
        
        if controller.simplified_path:
        
            next_target = controller.simplified_path[0]

            zx, zy = next_target[:2]

            dist = np.linalg.norm(np.array([cx, cy]) - np.array([zx, zy]))
            if dist < 50:
                if len(controller.simplified_path) > 1:
                    controller.simplified_path.pop(0)
                else:
                    print("[Stage 1] Reached final target, switching to stage 2")
                    controller.delivery_stage = 2
                    controller.simplified_path = None
                    return RobotState.DELIVERY

            controller.current_target = next_target
            command = determine_direction(robot_info, next_target, cross)
            controller.send_command(command)
            return RobotState.DELIVERY

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
        if dist_back > 115:
            movement_command = "slow_backward"
            controller.send_command(movement_command)
        else:
            controller.delivery_stage = 4
        return RobotState.DELIVERY

    elif controller.delivery_stage == 4:
        print("[Stage 4] Sending delivery command")
        movement_command = "delivery"
        controller.send_command(movement_command)
        controller.last_delivery_time = time.time()
        controller.delivery_stage = 5
        return RobotState.DELIVERY
    
    elif controller.delivery_stage == 5:
        if(time.time() - controller.last_delivery_time >= 5):
            movement_command = "continue"
            controller.send_command(movement_command)
            controller.delivery_stage = 6
            controller.last_delivery_time = time.time()
            return RobotState.DELIVERY
    elif controller.delivery_stage == 6:
        if(time.time() - controller.last_delivery_time >= 5):
            controller.delivery_stage = 0
            controller.delivery_active = False
            controller.waiting_for_continue = False
            controller.last_delivery_count = len(ball_positions)
            return RobotState.COLLECTION
     
    return RobotState.DELIVERY