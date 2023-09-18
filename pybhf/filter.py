from abc import abstractmethod
from typing import Optional, Tuple, Any, Iterable

import numpy as np
import numpy.typing as npt
from pybhf.grid import Grid, GridAxis, GridIndex, GridKey, GridValue


class HistogramFilterBase:
    def __init__(
        self,
        x_axis: GridAxis,
        y_axis: GridAxis,
        zero_threshold: float = 0.00001,
        unusable_cells: Iterable[GridIndex] = frozenset(),
        motion_noise: Optional[npt.ArrayLike] = None,
        observ_noise: Optional[npt.ArrayLike] = None,
    ) -> None:
        # Initialize the grid.
        self._init_grid(x_axis, y_axis, zero_threshold, unusable_cells)

        # Initialize the motion noise and observation noise if provided.
        self._motion_noise, self._observation_noise = None, None
        self.set_motion_noise(motion_noise)
        self.set_observation_noise(observ_noise)

        # Maintain a random sampler to prevent unnecessary recreation.
        self._rng = np.random.default_rng()

    def __getitem__(self, key: GridKey) -> GridValue:
        return self._grid[key]

    def __setitem__(self, key: GridKey, value: GridValue) -> None:
        self._grid[key] = value

    @property
    def belief(self) -> Grid:
        return self._grid

    @property
    def motion_noise(self) -> npt.NDArray:
        return self._motion_noise

    @property
    def observation_noise(self) -> npt.NDArray:
        return self._observation_noise

    def _init_grid(
        self,
        x_axis: GridAxis,
        y_axis: GridAxis,
        zero_threshold: float,
        unusable_cells: Iterable[GridIndex],
    ) -> None:
        self._grid: Grid = Grid(x_axis, y_axis, zero_threshold)
        self._grid.set_unusable_cells(unusable_cells)

    @staticmethod
    def _is_valid_noise(noise: npt.NDArray) -> bool:
        """Checks if the noise parameter is 'valid.'"""
        return (
            len(noise.shape) == 2
            and noise.shape[0] == noise.shape[1]
            and np.sum(noise) == 1.0
        )

    def sample(self, n_samples: int = 1) -> npt.NDArray[GridValue]:
        # Get the cells with positive probabilities.
        positive_prob_keys: Tuple[GridKey] = self._grid.get_nonzero_keys()
        if len(positive_prob_keys) == 0:
            raise RuntimeError("There are no cells with positive values!")

        # Randomly sample the cells with positive values.
        probs = tuple((self._grid[key] for key in positive_prob_keys))
        if (prob_sum := np.sum(probs)) != 1.0:
            raise RuntimeError(
                f"The probabilities for the nonzero keys sum to {prob_sum}."
            )
        samples: npt.NDArray = self._rng.choice(
            positive_prob_keys, size=n_samples, replace=True, p=probs, axis=0
        )
        return samples

    def set_motion_noise(self, noise: Optional[npt.ArrayLike] = None) -> None:
        noise = [[1.0]] if noise is None else noise
        noise = np.array(noise)
        if not self._is_valid_noise(noise):
            raise ValueError(f"motion noise must be a square matrix that sums to 1.")
        self._motion_noise = noise.copy()

    def set_observation_noise(self, noise: Optional[npt.ArrayLike] = None) -> None:
        noise = [[1.0]] if noise is None else noise
        noise = np.array(noise)
        if not self._is_valid_noise(noise):
            raise ValueError(
                f"observation noise must be a square matrix that sums to 1."
            )
        self._observation_noise = noise.copy()

    @abstractmethod
    def predict(self, command: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(
        self, observation: npt.ArrayLike, noise: Optional[npt.ArrayLike] = None
    ) -> None:
        raise NotImplementedError
