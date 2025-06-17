import unittest
import math
from mac.pathfinding import barrier_blocks_path

# kør test vha. python -m unittest tests.test_barrier -v i terminalen for nu


class TestBarrierBlocksPath(unittest.TestCase):
    def setUp(self):
        # En simpel robot-front og bold på x-aksen
        self.robot = (0, 0)
        # (x, y, r, o) — her r og o irrelevante
        self.ball = (100,  0,  0, None)

    def test_no_obstacles(self):
        # Ingen æg, ingen kryds, ALDRIG blokeret
        self.assertFalse(barrier_blocks_path(
            robot=self.robot,
            ball=self.ball,
            eggs=[],
            crosses=[],
            robot_radius=10,
            threshold=5
        ))

    def test_egg_blocking_center(self):
        # Æg ligger på midtlinjen tæt på (50,0) med radius 0
        eggs = [(50, 0, 0, None)]
        # threshold=0 så dist_to_line = 0 < er+threshold → True
        self.assertTrue(barrier_blocks_path(
            robot=self.robot,
            ball=self.ball,
            eggs=eggs,
            crosses=[],
            robot_radius=10,
            threshold=0
        ))

    def test_egg_blocking_side(self):
        # Æg ligger lige uden for midtlinjen, men rammer højre side-linje
        # Robot_radius=10 og threshold=0 → side-linje ligger ved y=±10
        eggs = [(50, 10, 0, None)]
        self.assertTrue(barrier_blocks_path(
            robot=self.robot,
            ball=self.ball,
            eggs=eggs,
            crosses=[],
            robot_radius=10,
            threshold=0
        ))

    def test_egg_not_blocking(self):
        # Æg ligger lige uden for rækkevidde af side-linjer
        eggs = [(50, 11, 0, None)]
        self.assertFalse(barrier_blocks_path(
            robot=self.robot,
            ball=self.ball,
            eggs=eggs,
            crosses=[],
            robot_radius=10,
            threshold=0
        ))

    def test_cross_blocking(self):
        # Kryds er et segment på midtlinjen fra (50,-5) til (50,5)
        crosses = [(50, -5, 50, 5)]
        self.assertTrue(barrier_blocks_path(
            robot=self.robot,
            ball=self.ball,
            eggs=[],
            crosses=crosses,
            robot_radius=10,
            threshold=0
        ))

    def test_cross_not_blocking(self):
        # Samme kryds, men threshold=-6 så det ikke fanges
        crosses = [(50, -5, 50, 5)]
        self.assertFalse(barrier_blocks_path(
            robot=self.robot,
            ball=self.ball,
            eggs=[],
            crosses=crosses,
            robot_radius=10,
            threshold=-6
        ))


if __name__ == "__main__":
    unittest.main()
