import socket
import globals_config as g
from pathfinding import get_zone_center
from robot_state import RobotState
import numpy as np
import time

class RobotController:
    def __init__(self, conn: socket.socket=None):
        ### COMMUNICATION ###
        self.conn = conn

        ### STATES ###
        self.state = RobotState.COLLECTION
        self.next_state_candidate = None
        self.next_state_count = 0
        self.required_repeats = 3

        ### COMMAND ###
        self.last_command = None

        ### BALLS ###
        self.last_ball_count = 11

        ### DELIVERY ###
        self.delivery_stage = 0
        self.delivery_active = False
        self.last_delivery_count = 11
        self.delivery_candidate = None
        self.consecutive_delivery_frames = 0
        self.waiting_for_continue = False
        self.last_delivery_time = 0

        ### STAGING ###
        self.at_blocked_staging = False
        self.staged_balls = []
        self.edge_alignment_active = False
        
        ### PATH ###
        self.path_to_target = None
        self.reached_path_point = False
        self.simplified_path = None

        ### TARGETS ###
        self.goal_back_alignment_target = (g.FIELD_X_MAX - 20, (g.FIELD_Y_MIN + g.FIELD_Y_MAX) // 2)
        self.goal_first_target = (g.FIELD_X_MAX - 300, (g.FIELD_Y_MIN + g.FIELD_Y_MAX) // 2)

        self.current_target = None

    def send_command(self, command: str):
        if command and command != self.last_command:
            print(f"[RobotController] Sending command: {command}")
            if self.conn:
                try:
                    self.conn.sendall(command.encode())
                    self.last_command = command

                    if command == "delivery":
                        self.last_delivery_time = time.time()
                        self.waiting_for_continue = True

                except Exception as e:
                    print(f"[RobotController] Failed to send command: {e}")

    def reset_command(self):
        self.last_command = None

    def set_state(self, new_state: RobotState):
        if new_state != self.state:
            self.state = new_state

            # Reset values
            self.reset_command()
            self.path_to_target = None
            self.reached_path_point = False
            self.current_target = None
            self.delivery_stage = 0
            self.delivery_active = False
            self.waiting_for_continue = False
            self.at_blocked_staging = False

    def reached_next_path_point(self, cx, cy):
        if not self.path_to_target:
            return False
        zx, zy = get_zone_center(self.path_to_target[0])
        dist = np.linalg.norm(np.array([cx, cy]) - np.array([zx, zy]))
        return dist < 20
    
    def update_state(self, proposed_state: RobotState):
        if proposed_state == self.state:
            self.next_state_candidate = None
            self.next_state_count = 0
        else:
            if proposed_state == self.next_state_candidate:
                self.next_state_count += 1
                if self.next_state_count >= self.required_repeats:
                    self.set_state(proposed_state)
                    self.next_state_candidate = None
                    self.next_state_count = 0
            else:
                self.next_state_candidate = proposed_state
                self.next_state_count = 1



