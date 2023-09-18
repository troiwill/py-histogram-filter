import unittest

import numpy as np
from pybhf.filter import HistogramFilterBase
from pybhf.grid import GridAxis


class TestHistogramFilterBase(unittest.TestCase):
    def setUp(self):
        # Create a dummy GridAxis for testing
        self.x_axis = GridAxis("x", 0, 1, 0.2)
        self.y_axis = GridAxis("y", 0, 1, 0.2)

    def test_init(self):
        # Test the initialization of the HistogramFilterBase class
        histogram_filter = HistogramFilterBase(self.x_axis, self.y_axis)

        self.assertIsInstance(histogram_filter, HistogramFilterBase)
        self.assertEqual(histogram_filter.motion_noise.tolist(), [[1.0]])
        self.assertEqual(histogram_filter.observation_noise.tolist(), [[1.0]])

    def test_set_motion_noise(self):
        histogram_filter = HistogramFilterBase(self.x_axis, self.y_axis)

        # Test setting motion noise with valid input
        valid_noise = np.ones((3, 3)) / 9.0
        histogram_filter.set_motion_noise(valid_noise)
        self.assertTrue(np.array_equal(histogram_filter.motion_noise, valid_noise))

        # Test setting invalid motion noise
        with self.assertRaises(ValueError):
            invalid_noise = np.array([[0.2, 0.3], [0.4, 0.2]])
            histogram_filter.set_motion_noise(invalid_noise)

    def test_set_observation_noise(self):
        histogram_filter = HistogramFilterBase(self.x_axis, self.y_axis)

        # Test setting observation noise with valid input
        valid_noise = np.ones((5, 5)) / 25.0
        histogram_filter.set_observation_noise(valid_noise)
        self.assertTrue(np.array_equal(histogram_filter.observation_noise, valid_noise))

        # Test setting invalid observation noise
        with self.assertRaises(ValueError):
            invalid_noise = np.array([[0.2, 0.3], [0.4, 0.2]])
            histogram_filter.set_observation_noise(invalid_noise)

    def test_sample_shape(self):
        histogram_filter = HistogramFilterBase(self.x_axis, self.y_axis)
        with self.assertRaises(RuntimeError):
            _ = histogram_filter.sample(n_samples=5)

        histogram_filter[0.05, 0.05] = 0.7
        histogram_filter[0.2, 0.8] = 0.3
        samples = histogram_filter.sample(n_samples=5)
        self.assertEqual(samples.shape, (5, 2))  # Check the shape of samples
        self.assertTrue(
            all(
                (
                    np.allclose(s, (0.1, 0.1)) or np.allclose(s, (0.3, 0.9))
                    for s in samples
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
