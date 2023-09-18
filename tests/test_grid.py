import unittest
from unittest.mock import Mock

import numpy as np
from pybhf.grid import Grid, GridAxis


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.zero_threshold = 0.01
        # Create GridAxis instances for testing
        self.x_axis = GridAxis(name="X", min=0.0, max=1.0, size=0.1)
        self.y_axis = GridAxis(name="Y", min=0.0, max=1.0, size=0.1)
        self.grid = Grid(self.x_axis, self.y_axis, self.zero_threshold)

    def test_valid_initialization(self):
        # Test valid initialization of Grid
        grid = Grid(self.x_axis, self.y_axis)
        self.assertIsInstance(grid, Grid)
        self.assertEqual(grid.x_axis, self.x_axis)
        self.assertEqual(grid.y_axis, self.y_axis)
        self.assertEqual(grid._zero_threshold, 0.00001)

    def test_invalid_axis_type(self):
        # Test initialization with invalid axis type (should raise TypeError)
        with self.assertRaises(TypeError):
            Grid(Mock(), Mock())

    def test_same_axis_name(self):
        # Test initialization with same axis names (should raise ValueError)
        with self.assertRaises(ValueError):
            Grid(self.x_axis, self.x_axis)

    def test_negative_zero_threshold(self):
        # Test initialization with a negative zero_threshold (should raise ValueError)
        with self.assertRaises(ValueError):
            Grid(self.x_axis, self.y_axis, zero_threshold=-0.1)

    def test_data_property(self):
        # Test the data property
        grid = Grid(self.x_axis, self.y_axis)
        self.assertIsInstance(grid.data, np.ndarray)
        self.assertEqual(grid.data.shape, (self.y_axis.n_bins, self.x_axis.n_bins))

    def test_volume_property(self):
        # Test the volume property
        grid = Grid(self.x_axis, self.y_axis)
        self.assertAlmostEqual(grid.volume, 0.01, places=8)

    def test_getitem(self):
        # Test the __getitem__ method
        grid = Grid(self.x_axis, self.y_axis)
        grid[(0.2, 0.3)] = 0.5
        self.assertTrue(np.allclose(grid[(0.2, 0.3)], np.array([0.5])))

    def test_setitem(self):
        # Test the __setitem__ method
        grid = Grid(self.x_axis, self.y_axis)
        grid[(0.2, 0.3)] = 0.5
        self.assertTrue(np.allclose(grid.get_nonzero_cells(), (3, 2)))
        self.assertAlmostEqual(grid.data[3, 2], 0.5)

    def test_copy(self):
        # Test the copy method
        grid = Grid(self.x_axis, self.y_axis)
        grid[(0.2, 0.3)] = 0.5
        copy = grid.copy()
        self.assertEqual(grid, copy)

    def test_get_cell_key(self):
        # Test the get_cell_key method
        grid = Grid(self.x_axis, self.y_axis)
        key = grid.get_cell_key((3, 2))
        self.assertTrue(np.allclose(key, np.array([0.25, 0.35])))

    def test_get_value_at_index(self):
        # Test the get_value_at_index method
        grid = Grid(self.x_axis, self.y_axis)
        grid[(0.2, 0.3)] = 0.5
        value = grid.data[3, 2]
        self.assertAlmostEqual(value, 0.5)

    def test_get_nonzero_cells(self):
        # Test the get_nonzero_cells method
        grid = Grid(self.x_axis, self.y_axis)
        grid[(0.2, 0.3)] = 0.5
        nonzero_cells = grid.get_nonzero_cells()
        self.assertTrue(np.allclose(nonzero_cells, (3, 2)))

    def test_get_nonzero_keys(self):
        # Test the get_nonzero_keys method
        grid = Grid(self.x_axis, self.y_axis)
        grid[(0.2, 0.3)] = 0.5
        nonzero_keys = grid.get_nonzero_keys()
        self.assertTrue(np.allclose(nonzero_keys, (0.25, 0.35)))


if __name__ == "__main__":
    unittest.main()
