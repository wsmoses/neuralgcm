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
"""Functionality for converting between coordax.Field and Xarray objects."""
from collections import abc
import typing
from typing import Callable

from neuralgcm.experimental.coordax import core
import xarray


def field_to_data_array(field: core.Field) -> xarray.DataArray:
  """Converts a coordax.Field to an xarray.DataArray.

  Args:
    field: coordax.Field to convert into an xarray.DataArray.

  Returns:
    An xarray.DataArray object with the same data as the input coordax.Field.
    This DataArray will still be wrapping a jax.Array and have operations
    implemented on jax.Array objects using the Python Array API interface.
  """
  if not all(isinstance(dim, str) for dim in field.dims):
    raise ValueError(
        'can only convert Field objects with fully named dimensions to Xarray '
        f'objects, got dimensions {field.dims!r}'
    )

  coords = {}
  field_dims = set(field.dims)
  for coord in field.coords.values():
    for name, coord_field in coord.fields.items():
      if set(coord_field.dims) <= field_dims:
        # xarray.DataArray coordinate dimensions must be a subset of the
        # dimensions of the associated DataArray, which is not necessarily a
        # constraint for coordax.Field.
        variable = xarray.Variable(coord_field.dims, coord_field.data)
        if name in coords and not variable.identical(coords[name]):
          raise ValueError(
              f'inconsistent coordinate fields for {name!r}:\n'
              f'{variable}\nvs\n{coords[name]}'
          )
        coords[name] = variable

  return xarray.DataArray(data=field.data, dims=field.dims, coords=coords)


CoordMatcher = Callable[
    [tuple[str, ...], xarray.Coordinates], core.Coordinate | None
]


def _labeled_axis_matcher(dims: tuple[str, ...], coords: xarray.Coordinates):
  dim = dims[0]
  if dim in coords and coords[dim].ndim == 1:
    return core.LabeledAxis(dim, coords[dim].data)
  return None


def _named_axis_matcher(dims: tuple[str, ...], coords: xarray.Coordinates):
  dim = dims[0]
  return core.NamedAxis(dim, size=coords.sizes[dim])


DEFAULT_MATCHERS = (_labeled_axis_matcher, _named_axis_matcher)


def data_array_to_field(
    data_array: xarray.DataArray,
    coord_matchers: abc.Sequence[CoordMatcher] = DEFAULT_MATCHERS,
) -> core.Field:
  """Converts an xarray.DataArray to a coordax.Field.

  Args:
    data_array: xarray.DataArray to convert into a Field.
    coord_matchers: sequence of functions that take a tuple of dimensions and
      an xarray.Coordinates object and return a coordax.Coordinate object from
      the leading dimensions if possible, or None otherwise. The first matcher
      that returns a match will be used. By default, coordinates are constructed
      out of generic coordax.LabeledAxis and coordax.NamedAxis objects.

  Returns:
    A coordax.Field object with the same data as the input xarray.DataArray.
  """
  field = core.wrap(data_array.data)
  dims = data_array.dims
  coords = []

  if not all(isinstance(dim, str) for dim in dims):
    raise TypeError(
        'can only convert DataArray objects with string dimensions to Field'
    )
  dims = typing.cast(tuple[str, ...], dims)

  def get_next_match():
    for matcher in coord_matchers:
      coord = matcher(dims, data_array.coords)
      if coord is not None:
        return coord
    raise ValueError(f'no match found for dimensions starting with {dims}')

  while dims:
    coord = get_next_match()
    coords.append(coord)
    assert coord.ndim > 0  # dimensions will shrink by at least one
    dims = dims[coord.ndim :]

  return field.tag(*coords)
