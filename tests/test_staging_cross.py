# tests/test_create_staging.py

import unittest
import math
from mac.pathfinding import create_staging_ball_cross


class TestCreateStagingBallCross(unittest.TestCase):
    # Definerer fælles parametre for alle tests
    def setUp(self):
        # Offset som staging-punktet skal skubbes med (mm)
        self.offset = 200
        # Radius og farve-id
        self.r, self.o = (3, 99)

    # Test 1: Bold i øvre højre kvadrant i et symmetrisk kryds
    def test_staging_quadrant_1_perfect(self):
        cross_bounds = (0, 20, 0, 20)
        # Boldposition i øvre højre hjørne (dx>0, dy>0)
        ball = (17, 17, self.r, self.o)

        sx, sy, rr, oo = create_staging_ball_cross(
            ball, cross_bounds, offset_distance=self.offset
        )

        # Beregn forventet staging-punkt: midt = (10,10), d = offset/√2
        cx = (0 + 20) / 2.0
        cy = (0 + 20) / 2.0
        d = self.offset / math.sqrt(2)
        exp_x = cx + d
        exp_y = cy + d

        # Tjek at radius og farve-id returneres uændret
        self.assertEqual(rr, self.r)
        self.assertEqual(oo, self.o)

        # Tjek at koordinater stemmer (afrundet til int)
        self.assertEqual(sx, int(exp_x))
        self.assertEqual(sy, int(exp_y))

    # Test 2: Bold i nederste venstre kvadrant med “roteret” boks
    def test_staging_quadrant_3_rotated_bounds(self):
        cross_bounds = (10, 30, 5, 25)
        # Boldposition i nederste venstre (dx<0, dy<0)
        ball = (12, 8, self.r, self.o)

        sx, sy, rr, oo = create_staging_ball_cross(
            ball, cross_bounds, offset_distance=self.offset
        )

        cx = (10 + 30) / 2.0
        cy = (5 + 25) / 2.0
        d = self.offset / math.sqrt(2)
        exp_x = cx - d
        exp_y = cy - d

        self.assertEqual(rr, self.r)
        self.assertEqual(oo, self.o)
        self.assertEqual(sx, int(exp_x))
        self.assertEqual(sy, int(exp_y))


if __name__ == "__main__":
    unittest.main()
