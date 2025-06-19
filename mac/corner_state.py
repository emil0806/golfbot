from collections import deque
import time
from mac.collection_state import handle_collection
from mac.delivery_state import handle_delivery
from mac.robot_controller import RobotController
from mac.robot_state import RobotState
from pathfinding import (determine_direction, find_best_ball, sort_balls_by_distance,
    is_corner_ball, is_edge_ball, create_staging_point_corner, create_staging_point_edge, 
    barrier_blocks_path, close_to_barrier, set_homography, determine_staging_point, is_ball_and_robot_on_line_with_cross, is_ball_in_cross, draw_lines, determine_staging_point_16, determine_zone)
import numpy as np
import time

import globals_config as g

def handle_corner(robot_info, ball_positions, egg, cross, controller: RobotController):
    ### DEFINER BEST BALL KORREKT ###

    best_ball = ball_positions[0]
    front_marker, center_marker, back_marker, _ = robot_info
    fx, fy = front_marker
    rx, ry = back_marker

    if corner_stage == 1:
        print("Corner stage 1")
        # Naviger til staging

        if best_ball:
            if np.hypot(fx - best_ball[0], fy - best_ball[1]) < 50:
                movement_command = "delivery"
                controller.send_command(movement_command)
                corner_stage = 2
                print("Next corner stage")
            else:
                movement_command = determine_direction(robot_info, best_ball)
                controller.send_command(movement_command)

    elif corner_stage == 2:
        # Roter korrekt mod bolden (ligesom i delivery_stage 2)
        robot_vector = np.array(back_marker) - np.array(front_marker)
        desired_vector = np.array(best_ball[:2]) - np.array(back_marker)

        dot = np.dot(robot_vector, desired_vector)
        mag_r = np.linalg.norm(robot_vector)
        mag_d = np.linalg.norm(desired_vector)
        cos_theta = max(-1, min(1, dot / (mag_r * mag_d + 1e-6)))
        angle_diff = np.degrees(np.arccos(cos_theta))

        print(f"[Corner Stage 2] Angle to target: {angle_diff:.2f}")

        if angle_diff > 0.5:
            robot_3d = np.append(robot_vector, 0)
            desired_3d = np.append(desired_vector, 0)
            cross_product = np.cross(robot_3d, desired_3d)[2]
            if angle_diff > 30:
                movement_command = "left"
            elif angle_diff > 20:
                movement_command = "medium_left"
            elif angle_diff > 10:
                movement_command = "slow_left"
            else:
                movement_command = "very_slow_left"
            controller.send_command(movement_command)
        else:
            corner_stage = 3
    elif corner_stage == 3:
        print("Corner stage 3")
        if best_ball:
            bx, by, _, _ = best_ball
            if np.hypot(rx - bx, ry - by) < 120:
                robot_vector = np.array(back_marker) - np.array(front_marker)
                desired_vector = np.array(best_ball[:2]) - np.array(back_marker)

                dot = np.dot(robot_vector, desired_vector)
                mag_r = np.linalg.norm(robot_vector)
                mag_d = np.linalg.norm(desired_vector)
                cos_theta = max(-1, min(1, dot / (mag_r * mag_d + 1e-6)))
                angle_diff = np.degrees(np.arccos(cos_theta))

                print(f"[Corner Stage 3] Angle to target: {angle_diff:.2f}")

                if angle_diff > 0.5:
                    robot_3d = np.append(robot_vector, 0)
                    desired_3d = np.append(desired_vector, 0)
                    cross_product = np.cross(robot_3d, desired_3d)[2]
                    if angle_diff > 30:
                        movement_command = "left"
                    elif angle_diff > 20:
                        movement_command = "medium_left"
                    elif angle_diff > 10:
                        movement_command = "slow_left"
                    else:
                        movement_command = "very_slow_left"
                    controller.send_command(movement_command)
                else:
                    corner_stage = 4
            else:
                movement_command = "slow_backward"
                controller.send_command(movement_command)
    elif corner_stage == 4:
        print("Corner stage 4")
        if best_ball:
            bx, by, _, _ = best_ball
            if np.hypot(rx - bx, ry - by) < 60:
                stop_command = "stop"
                controller.send_command(stop_command)
                movement_command = "continue"
                controller.send_command(movement_command) 
                corner_stage = 0
                corner_ball = None
                controller.last_delivery_count -= 1
                print("continue")
                time.sleep(3)
                print("after sleep")
            else:
                movement_command = "slow_backward"
                controller.send_command(movement_command)
                print("back")
