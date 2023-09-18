from functools import cached_property
from dataclasses import dataclass, field
from typing import Tuple, Set, Iterable

import numpy as np
import numpy.typing as npt

AxisIndex = int
"""The integer used to access a particular axis in the array."""

GridIndex = Tuple[AxisIndex, AxisIndex]
"""A pair of axis indices. Format: (row_index, column_index)."""

AxisKey = float
"""The float value (from a high-level application like the filter) used to access a particular axis in the array. 
The key is converted to an axis index before being used with the array."""

GridKey = Tuple[AxisKey, AxisKey]
"""A pair of axis keys. Format: (column_key, row_key)."""

GridValue = float
"""The type of each value in the array."""


@dataclass(frozen=True)
class GridAxis:
    name: str
    min: GridValue
    max: GridValue
    size: GridValue
    epsilon: GridValue = field(default=0.000001)

    def __post_init__(self) -> None:
        if self.name == "":
            raise ValueError("name cannot be empty.")
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be less than max ({self.max}).")
        if self.size <= 0:
            raise ValueError(f"size ({self.size}) must be greater than zero.")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon ({self.epsilon}) must be greater than zero.")

    @cached_property
    def _half_size(self) -> float:
        return self.size / 2.0

    @cached_property
    def n_bins(self) -> int:
        return int((self.max - self.min) / self.size)

    def _is_valid_key(self, key: AxisKey) -> bool:
        return self.min <= key <= self.max

    def _is_valid_index(self, index: AxisIndex) -> bool:
        return 0 <= index < self.n_bins

    def copy(self) -> "GridAxis":
        return GridAxis(self.name, self.min, self.max, self.size)

    def get_index(self, key: AxisKey) -> AxisIndex:
        # Sanity check.
        if not self._is_valid_key(key):
            raise IndexError(f"{self.name}-axis key ({key}) is out-of-bounds. Range: [{self.min}, {self.max}].")
        index = self.n_bins - 1
        if key < self.max:
            index = (key - self.min) / self.size
        return int(index + self.epsilon)  # Round to negate any effects from decimal rounding errors.

    def get_key(self, index: AxisIndex) -> AxisKey:
        if not self._is_valid_index(index):
            raise IndexError(
                f"{self.name}-axis index ({index}) is out-of-bounds. Range: [0, {self.n_bins})."
            )
        return (float(index) * self.size) + self.min + self._half_size

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, GridAxis)
            and self.name == other.name
            and self.min == other.min
            and self.max == other.max
            and self.size == other.size
        )

    def __repr__(self) -> str:
        return f"GridAxis({self.name}, {self.min}, {self.max}, {self.size})"


class Grid:
    def __init__(
        self, x_axis: GridAxis, y_axis: GridAxis, zero_threshold: float = 0.00001
    ) -> None:
        self._x_axis, self._y_axis = x_axis.copy(), y_axis.copy()
        self._zero_threshold = zero_threshold

        # Sanity checks.
        if any((not isinstance(a, GridAxis) for a in (self._x_axis, self._y_axis))):
            raise TypeError("axis0 and axis1 must be type AxisSpecs.")
        if self._x_axis.name == self._y_axis.name:
            raise ValueError(
                f"axis0.name ({self._x_axis.name}) == axis1.name ({self._y_axis.name})!"
            )
        if self._zero_threshold < 0.0:
            raise ValueError(f"zero threshold must be at least zero.")

        # Create the grid.
        self._grid = np.empty(
            (self._y_axis.n_bins, self._x_axis.n_bins), dtype=GridValue
        )
        self._grid[:] = 0.0
        self._sig_digits = 8
        self._nonzero_cells: Set[Tuple[int, int]] = set()

    @property
    def data(self) -> npt.NDArray[GridValue]:
        """Returns a reference to the grid."""
        return self._grid

    @cached_property
    def volume(self) -> float:
        return round(self._x_axis.size * self._y_axis.size, self._sig_digits)

    @property
    def x_axis(self) -> GridAxis:
        return self._x_axis

    @property
    def y_axis(self) -> GridAxis:
        return self._y_axis

    def __eq__(self, other: "Grid") -> bool:
        return (
            self._x_axis == other._x_axis
            and self._y_axis == other._y_axis
            and self._zero_threshold == other._zero_threshold
            and np.allclose(self._grid, other._grid)
            and self._sig_digits == other._sig_digits
            and self._nonzero_cells == self._nonzero_cells
        )

    def __getitem__(self, key: GridKey) -> GridValue:
        index = self._get_grid_index(key)
        return self._grid[index]

    def __setitem__(self, key: GridKey, value: GridValue) -> None:
        index = self._get_grid_index(key)
        if value >= self._zero_threshold:
            self._nonzero_cells.add(index)
        else:
            value = 0.0
            self._nonzero_cells.discard(index)
        self._grid[index] = value

    def _get_grid_index(self, key: GridKey) -> GridIndex:
        """Converts grid keys [x_key, y_key] to grid indices [y_idx, x_idx]."""
        x_index, y_index = self._x_axis.get_index(key[0]), self._y_axis.get_index(
            key[1]
        )
        return y_index, x_index

    def copy(self) -> "Grid":
        _copy = Grid(self._x_axis, self._y_axis, self._zero_threshold)
        _copy._sig_digits = self._sig_digits
        _copy._nonzero_cells = self._nonzero_cells.copy()
        _copy._grid[:] = self._grid
        return _copy

    def get_cell_key(self, index: GridIndex) -> GridKey:
        """Converts a grid index [y_idx, x_idx] to grid key [x_key, y_key]."""
        y_key, x_key = self._y_axis.get_key(index[0]), self._x_axis.get_key(
            index[1]
        )
        return x_key, y_key

    def get_nonzero_cells(self) -> Tuple[GridIndex]:
        return tuple(self._nonzero_cells)

    def get_nonzero_keys(self) -> Tuple[GridKey]:
        return tuple((self.get_cell_key(index) for index in self.get_nonzero_cells()))

    def set_unusable_cells(self, cells: Iterable[GridIndex]) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"Grid(\n\t{self._x_axis},\n\t{self._y_axis},\n\t{self._zero_threshold}\n)"
        )
