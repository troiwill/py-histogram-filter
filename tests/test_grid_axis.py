import unittest

import numpy as np
from pybhf.grid import GridAxis


class GridAxisTestCase(unittest.TestCase):
    def test_valid_initialization(self):
        # Test valid initialization of GridAxis
        axis = GridAxis(name="X", min=0.0, max=1.0, size=0.1)
        self.assertEqual(axis.name, "X")
        self.assertEqual(axis.min, 0.0)
        self.assertEqual(axis.max, 1.0)
        self.assertEqual(axis.size, 0.1)

    def test_empty_name(self):
        # Test initialization with an empty name (should raise ValueError)
        with self.assertRaises(ValueError):
            GridAxis(name="", min=0.0, max=1.0, size=0.1)

    def test_min_greater_than_max(self):
        # Test initialization with min greater than max (should raise ValueError)
        with self.assertRaises(ValueError):
            GridAxis(name="Y", min=1.0, max=0.0, size=0.1)

    def test_non_positive_size(self):
        # Test initialization with non-positive size (should raise ValueError)
        with self.assertRaises(ValueError):
            GridAxis(name="Z", min=0.0, max=1.0, size=0.0)

    def test_n_bins(self):
        # Test the n_bins property
        axis = GridAxis(name="X", min=0.0, max=1.0, size=0.1)
        self.assertEqual(axis.n_bins, 10)

    def test_valid_key(self):
        # Test _is_valid_key method with valid keys
        axis = GridAxis(name="X", min=0.0, max=1.0, size=0.1)
        self.assertTrue(axis._is_valid_key(0.5))
        self.assertTrue(axis._is_valid_key(0.0))
        self.assertTrue(axis._is_valid_key(1.0))

    def test_invalid_key(self):
        # Test _is_valid_key method with invalid keys
        axis = GridAxis(name="X", min=0.0, max=1.0, size=0.1)
        self.assertFalse(axis._is_valid_key(-0.1))
        self.assertFalse(axis._is_valid_key(1.1))
        self.assertFalse(axis._is_valid_key(2.0))

    def test_valid_index(self):
        # Test _is_valid_index method with valid indices
        axis = GridAxis(name="X", min=0.0, max=1.0, size=0.1)
        self.assertTrue(axis._is_valid_index(0))
        self.assertTrue(axis._is_valid_index(9))

    def test_invalid_index(self):
        # Test _is_valid_index method with invalid indices
        axis = GridAxis(name="X", min=0.0, max=1.0, size=0.1)
        self.assertFalse(axis._is_valid_index(-1))
        self.assertFalse(axis._is_valid_index(10))
        self.assertFalse(axis._is_valid_index(100))

    def test_copy(self):
        # Test the copy method
        axis = GridAxis(name="X", min=0.0, max=1.0, size=0.1)
        copied_axis = axis.copy()
        self.assertEqual(axis, copied_axis)

    def test_get_index(self):
        # Test the get_index method
        axis = GridAxis(name="X", min=0.0, max=1.0, size=0.1)
        self.assertEqual(axis.get_index(0.35), 3)
        self.assertEqual(axis.get_index(0.), 0)
        self.assertEqual(axis.get_index(1.), 9)

    def test_get_key(self):
        # Test the get_key method
        axis = GridAxis(name="X", min=-1.0, max=0.0, size=0.1)
        self.assertTrue(np.isclose(axis.get_key(3), -0.65))
        self.assertTrue(np.isclose(axis.get_key(0), -0.95))
        self.assertTrue(np.isclose(axis.get_key(9), -0.05))

    def test_eq(self):
        # Test the __eq__ method
        axis1 = GridAxis(name="X", min=0.0, max=1.0, size=0.1)
        axis2 = GridAxis(name="X", min=0.0, max=1.0, size=0.1)
        axis3 = GridAxis(name="Y", min=0.0, max=1.0, size=0.1)
        self.assertEqual(axis1, axis2)
        self.assertNotEqual(axis1, axis3)

    def test_repr(self):
        # Test the __repr__ method
        axis = GridAxis(name="X", min=0.0, max=1.0, size=0.1)
        self.assertEqual(repr(axis), "GridAxis(X, 0.0, 1.0, 0.1)")


if __name__ == "__main__":
    unittest.main()
