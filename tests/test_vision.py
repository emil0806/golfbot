import unittest
import numpy as np
from mac.vision import inside_field, filter_barriers_inside_field

# kør test vha. python -m unittest tests.test_vision -v i terminal for nu


class TestVision(unittest.TestCase):
    # Test 1: inside_field med forskellige segment-typer (liste, tuple, ndarray)
    def test_inside_field_mixed_types(self):
        segs = [
            [0, 0, 5, 5],
            (1, 2, 3, 4),
            np.array([2, -1, 4, 6])
        ]
        xmin, xmax, ymin, ymax = inside_field(segs)
        self.assertEqual((xmin, xmax, ymin, ymax), (0, 5, -1, 6))

    # Test 2: filter_barriers_inside_field fjerner barrierer på margin‐grænsen
    def test_filter_barriers_with_margin_default(self):
        # Barrier på margin=20 fjernes, barrier midt i billedet bevares
        shape = (100, 100, 3)
        barriers = [
            ((10, 10, 20, 20), (20, 20)),  # præcis på margin fjernes
            ((50, 50, 60, 60), (50, 50)),  # midt i billedet bevares
        ]
        filtered = filter_barriers_inside_field(barriers, shape)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0][1], (50, 50))


if __name__ == "__main__":
    unittest.main()
