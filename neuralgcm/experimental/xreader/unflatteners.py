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
"""Unflattening interface."""

from __future__ import annotations

import abc
from collections.abc import Sequence
import dataclasses
import functools
from typing import Any, Callable, TYPE_CHECKING

import xarray

if TYPE_CHECKING:
  # pylint: disable=g-import-not-at-top,g-bad-import-order
  from neuralgcm.experimental import coordax


PyTree = Any


class Unflattener(abc.ABC):
  """Converts a list of arrays into a pytree of arrays."""

  @abc.abstractmethod
  def build(self, source: xarray.Dataset) -> Callable[[list[Any]], PyTree]:
    """Returns a function that unflattens a list of arrays into a pytree."""
    raise NotImplementedError


def _unflatten_arrays(names: list[str], arrays: list[Any]) -> dict[str, Any]:
  assert len(arrays) == len(names)
  return dict(zip(names, arrays))


class ArrayUnflattener(Unflattener):
  """Unflatten into a dict of arrays."""

  def build(self, source: xarray.Dataset) -> Callable[[list[Any]], PyTree]:
    names = list(source.keys())
    return functools.partial(_unflatten_arrays, names)


def _unflatten_fields(
    coords: dict[str, coordax.Coordinate], arrays: list[Any]
) -> dict[str, coordax.Field]:
  from neuralgcm.experimental import coordax  # pylint: disable=g-import-not-at-top

  return {
      name: coordax.Field(array).tag(None, coord)  # include leading sample dim
      for (name, coord), array in zip(coords.items(), arrays)
  }


@dataclasses.dataclass
class CoordaxUnflattener(Unflattener):
  """Unflatten into a dict of coordax.Field objects."""

  coord_types: Sequence[type[coordax.Coordinate]] | None = None

  def build(self, source: xarray.Dataset) -> Callable[[list[Any]], PyTree]:
    from neuralgcm.experimental import coordax  # pylint: disable=g-import-not-at-top

    # Coordax is an optional dependency for Xreader, so define default values
    # for coord_types here instead of on the dataclass field.
    coord_types = (
        (coordax.LabeledAxis, coordax.DummyAxis)
        if self.coord_types is None
        else self.coord_types
    )
    coords = {}
    for k, data_array in source.items():
      # sample_dim is moved to the front via transpose in _prepare_source.
      array_without_sample_dim = data_array[0, ...]
      coords[k] = coordax.coordinates_from_xarray(
          array_without_sample_dim, coord_types
      )

    return functools.partial(_unflatten_fields, coords)
