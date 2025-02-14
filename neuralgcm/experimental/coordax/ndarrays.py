# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Duck-typed JAX arrays for Coordax."""

from __future__ import annotations

import abc
from typing import Any, Callable, Self, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

try:
  from neuralgcm.experimental import jax_datetime  # pylint: disable=g-import-not-at-top
except ImportError:
  jax_datetime = None


class NDArray(abc.ABC):
  """Base class for non-JAX arrays that can be used with Coordax.

  To register a new NDArray, use `coordax.register_ndarray()`.
  """

  @property
  @abc.abstractmethod
  def shape(self) -> tuple[int, ...]:
    """Shape of this array."""

  @property
  @abc.abstractmethod
  def size(self) -> int:
    """Size of this array, typically `math.prod(self.shape)`."""

  @property
  @abc.abstractmethod
  def ndim(self) -> int:
    """Number of dimensions in this array, typcially `len(self.shape)`."""

  @abc.abstractmethod
  def transpose(self, axes: tuple[int, ...]) -> Self:
    """Transpose this array to this given axis order."""

  # Note: __getitem__ is not yet used by Coordax, but we will likely want it in
  # the near the future.
  # TODO(shoyer): Figure out the precise type for `value`.
  @abc.abstractmethod
  def __getitem__(self, value) -> Self:
    """Index this array, returning a new array."""


T = TypeVar('T')


_TO_NUMPY_FUNCS: list[
    tuple[type[NDArray], Callable[[NDArray], np.ndarray]]
] = []
_FROM_NUMPY_FUNCS: list[
    tuple[Callable[[Any], bool], Callable[[np.ndarray], NDArray]]
] = []


def register_ndarray(
    array_type: type[T],
    is_matching_numpy_array: Callable[[np.ndarray], bool],
    to_numpy: Callable[[T], np.ndarray],
    from_numpy: Callable[[np.ndarray], T],
) -> type[T]:
  """Registers a new duck-typed array type corresponding to a numpy dtype."""
  NDArray.register(array_type)
  _TO_NUMPY_FUNCS.append((array_type, to_numpy))
  _FROM_NUMPY_FUNCS.append((is_matching_numpy_array, from_numpy))
  return array_type


def to_ndarray(data: jax.typing.ArrayLike | NDArray) -> jax.Array | NDArray:
  """Returns an NDArray compatible with Coordax."""

  if isinstance(data, np.ndarray):
    for is_matching_numpy_array, from_numpy in _FROM_NUMPY_FUNCS:
      if is_matching_numpy_array(data):
        return from_numpy(data)

  if not isinstance(data, NDArray):
    try:
      data = jnp.asarray(data)
    except TypeError as e:
      raise TypeError(
          'data must be a jax.Array or a duck-typed array registered with '
          f'coordax.register_ndarray(), got {type(data).__name__}: {data}'
      ) from e

  return data


def to_numpy_array(data: jax.Array | NDArray) -> np.ndarray:
  """Returns a numpy-compatible version of this array."""
  if isinstance(data, NDArray):
    for array_type, to_numpy in _TO_NUMPY_FUNCS:
      if isinstance(data, array_type):
        return to_numpy(data)
  return np.asarray(data)


if jax_datetime is not None:
  register_ndarray(
      jax_datetime.Timedelta,
      lambda x: np.issubdtype(x.dtype, np.timedelta64),
      lambda x: x.to_timedelta64(),
      jax_datetime.Timedelta.from_timedelta64,
  )
  register_ndarray(
      jax_datetime.Datetime,
      lambda x: np.issubdtype(x.dtype, np.datetime64),
      lambda x: x.to_datetime64(),
      jax_datetime.Datetime.from_datetime64,
  )
