import unittest
import math
from mac.pathfinding import create_staging_ball_cross

# kør test vha. python -m unittest tests.test_staging_cross -v i termninal for nu


class TestCreateStagingBallCross(unittest.TestCase):
    def setUp(self):
        self.offset = 200  # offset
        self.r, self.o = (3, 99)  # radius og farve

    # Test 1: Bold i øvre højre kvadrant af et symmetrisk kryds
    def test_staging_quadrant_1_perfect(self):
        cross_bounds = (0, 20, 0, 20)
        ball = (17, 17, self.r, self.o)

        sx, sy, rr, oo = create_staging_ball_cross(
            ball, cross_bounds, offset_distance=self.offset)

        # Beregn centret
        cx = (0 + 20) / 2.0
        cy = (0 + 20) / 2.0
        # Retningsvektor fra centrum → bold
        dx = ball[0] - cx
        dy = ball[1] - cy
        mag = math.hypot(dx, dy) or 1.0
        ux, uy = dx/mag, dy/mag
        # Forventet staging = centrum + unit_vector * offset
        exp_x = cx + ux * self.offset
        exp_y = cy + uy * self.offset

        self.assertEqual(rr, self.r)
        self.assertEqual(oo, self.o)
        self.assertAlmostEqual(sx, int(exp_x))
        self.assertAlmostEqual(sy, int(exp_y))

    # Test 2: Bold i nederste venstre kvadrant af et “roteret” kryds
    def test_staging_quadrant_3_rotated_bounds(self):
        cross_bounds = (10, 30, 5, 25)
        ball = (12, 8, self.r, self.o)

        sx, sy, rr, oo = create_staging_ball_cross(
            ball, cross_bounds, offset_distance=self.offset)

        cx = (10 + 30) / 2.0
        cy = (5 + 25) / 2.0
        dx = ball[0] - cx
        dy = ball[1] - cy
        mag = math.hypot(dx, dy) or 1.0
        ux, uy = dx/mag, dy/mag
        exp_x = cx + ux * self.offset
        exp_y = cy + uy * self.offset

        self.assertEqual(rr, self.r)
        self.assertEqual(oo, self.o)
        self.assertAlmostEqual(sx, int(exp_x))
        self.assertAlmostEqual(sy, int(exp_y))


if __name__ == "__main__":
    unittest.main()
