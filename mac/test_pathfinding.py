import unittest
import pathfinding
from visualize_pathfinding import visualize_pathfinding

class TestPathfinding(unittest.TestCase):
    def setUp(self):
        pathfinding.previous_best_ball = None

    def test_find_best_ball_selects_closest(self):
        self.print_test_header("test_find_best_ball_selects_closest")

        robot_pos = ((600, 600), (700, 600), 0)  # Center field, facing right
        balls = [(900, 900), (800, 800), (300, 300)]

        best_ball = pathfinding.find_best_ball(balls, robot_pos[0], robot_pos[1])
        
        self.assertEqual(best_ball, (800, 800))
        
        visualize_pathfinding(robot_pos, balls)

        print("\n")


    def test_find_best_ball_sticks_with_previous(self):
        self.print_test_header("test_find_best_ball_sticks_with_previ")

        robot_pos = ((600, 600), (700, 600), 0)
        pathfinding.previous_best_ball = (800, 800)
        print("\n")

        # New "best" is only slightly closer
        balls = [(795, 795), (1000, 1000)]
        best_ball = pathfinding.find_best_ball(balls, robot_pos[0], robot_pos[1])
        
        self.assertEqual(best_ball, (800, 800))  # Should keep old
        
        visualize_pathfinding(robot_pos, balls)

        print("\n")

    def test_determine_direction_forward(self):
        self.print_test_header("test_determine_direction_forward")

        robot_pos = ((600, 600), (700, 600), 0)  # Facing right
        ball_pos = (1000, 600)  # Directly in front

        direction = pathfinding.determine_direction(robot_pos, ball_pos)

        self.assertEqual(direction, "forward")

        visualize_pathfinding(robot_pos, ball_pos)

        print("\n")


    def test_determine_direction_right_turn(self):
        self.print_test_header("test_determine_direction_right_turn")

        robot_pos = ((600, 600), (700, 600), 0)  # Facing right
        ball_pos = (600, 1000)  # Ball is directly "above" from robot's POV

        direction = pathfinding.determine_direction(robot_pos, ball_pos)

        self.assertEqual(direction, "right")

        visualize_pathfinding(robot_pos, ball_pos)

        print("\n")

    def test_determine_direction_various_angles(self):
        self.print_test_header("test_determine_direction_various_angles")

        robot_pos = ((600, 600), (700, 600), 0)  # Robot facing right

        test_cases = [
            ((1000, 600), "forward"),  # 0° — directly ahead
            ((900, 700), "right"),     # ~45° below-right
            ((700, 900), "right"),     # ~90° downward
            ((600, 1000), "right"),    # ~135° down-left
            ((200, 600), "left"),     # 180° behind
            ((600, 200), "left"),      # ~-90° up
            ((700, 400), "left"),      # ~-45° above right
            ((800, 600), "forward"),   # ~0°, far forward
            ((500, 500), "left"),      # ~135° up-left
        ]

        for i, (ball_pos, expected_direction) in enumerate(test_cases):
            with self.subTest(i=i, ball=ball_pos):
                direction = pathfinding.determine_direction(robot_pos, ball_pos)
                print(f"Test {i}: Ball at {ball_pos}, Expected: {expected_direction}, Got: {direction}")
                visualize_pathfinding(robot_pos, ball_pos)
                self.assertEqual(direction, expected_direction)

    def print_test_header(self, test_name):
        print(f"\n{'='*10} RUNNING: {test_name} {'='*10}")

if __name__ == "__main__":
    unittest.main()
