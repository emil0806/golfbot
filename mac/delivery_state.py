import math
import globals_config as g
from robot_controller import RobotController
import numpy as np
from robot_state import RobotState
from pathfinding import (_correct_marker, bfs_path, determine_direction, get_cross_zones, get_simplified_path, get_zone_center, get_zone_for_position, pix2world)
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

        front_marker, center_marker, back_marker, _ = robot_info

        bx, by = pix2world(controller.goal_second_target[:2])
        (rx_p, ry_p), (fx_p, fy_p) = back_marker, front_marker
        rx_w, ry_w = pix2world((rx_p, ry_p))
        fx_w, fy_w = pix2world((fx_p, fy_p))
        rx, ry = _correct_marker((rx_w, ry_w))
        fx, fy = _correct_marker((fx_w, fy_w))

        vector_to_ball = (bx - rx, by - ry)
        vector_front = (fx - rx, fy - ry)

        dot = vector_front[0] * vector_to_ball[0] + vector_front[1] * vector_to_ball[1]
        mag_f = math.hypot(*vector_front)
        mag_b = math.hypot(*vector_to_ball)
        cos_theta = max(-1, min(1, dot / (mag_f * mag_b)))
        angle_diff = math.degrees(math.acos(cos_theta))

        cross_product = -(vector_front[0] * vector_to_ball[1] - vector_front[1] * vector_to_ball[0])
        center = ((fx + rx) / 2, (fy + ry) / 2)

        print(f"[Stage 2] Angle to target: {angle_diff:.2f}")

        if angle_diff < 1.5:
            controller.delivery_stage = 3
            return RobotState.DELIVERY
        elif cross_product < 0:       
            if angle_diff > 25:
                movement_command = "fast_right"
                controller.send_command(movement_command)
            elif angle_diff > 15:
                movement_command = "right"
                controller.send_command(movement_command)
            else:
                movement_command = "medium_right"
                controller.send_command(movement_command)
        else:
            if angle_diff > 25:
                movement_command = "fast_left"
                controller.send_command(movement_command)
            elif angle_diff > 15:
                movement_command = "left"
                controller.send_command(movement_command)
            else:
                movement_command = "medium_left"
                controller.send_command(movement_command)
            
    elif controller.delivery_stage == 3:
        dist_front = np.linalg.norm(
            np.array(front_marker) - np.array(controller.goal_second_target))
        print(f"[Stage 3] Distance to front_alignment: {dist_front:.2f}")
        if dist_front > 30:
            movement_command = "slow_forward"
            controller.send_command(movement_command)
        else:
            controller.delivery_stage = 4
        return RobotState.DELIVERY

    elif controller.delivery_stage == 4:
        print("[Stage 4] Sending delivery command")
        movement_command = "delivery"
        controller.send_command(movement_command)
        return RobotState.DELIVERY
         
    return RobotState.DELIVERY