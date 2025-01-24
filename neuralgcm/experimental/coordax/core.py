# Copyright 2024 Google LLC
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
"""Defines ``Field`` and base ``Coordinate`` classes that define coordax API.

``Coordinate`` objects define a discretization schema, dimension names and
provide methods & coordinate field values to facilitate computations.
``Field`` objects keep track of positional and named dimensions of an array.
Named dimensions of a ``Field`` are associated with coordinates that describe
their discretization.
"""

from __future__ import annotations

import abc
import collections
import dataclasses
import functools
import operator
import textwrap
import typing
from typing import Any, Callable, Self, TypeAlias, TypeGuard, TypeVar

import jax
import jax.numpy as jnp
from neuralgcm.experimental.coordax import named_axes
import numpy as np
# TODO(shoyer): consider making Xarray an optional dependency of core Coordax
import xarray


Pytree: TypeAlias = Any
Sequence = collections.abc.Sequence

T = TypeVar('T')


# TODO(shoyer): remove this once Pytype understands T -> T preserves type.
@typing.overload
def export(func: type[T]) -> type[T]:
  ...


@typing.overload
def export(func: T) -> T:
  ...


def export(func):
  func.__module__ = 'neuralgcm.experimental.coordax'
  return func


@export
@dataclasses.dataclass(frozen=True)
class NoCoordinateMatch:
  """For use when a coordinate does not match an xarray.Coordinates object."""

  reason: str


@export
class Coordinate(abc.ABC):
  """Abstract class for coordinate objects.

  Coordinate subclasses are expected to obey several invariants:
  1. Dimension names may not be repeated: `len(set(dims)) == len(dims)`
  2. All dimensions must be named: `len(shape) == len(dims)`

  Every non-abstract Coordinate subclass must be registered as a "static"
  pytree node, e.g., by decorating the class with
  `@jax.tree_util.register_static`. Static pytrees nodes must implement
  `__hash__` and `__eq__` according to the requirements of keys in Python
  dictionaries. This is easiest to acheive with frozen dataclasses, but care
  must be taken when working with np.ndarray attributes.

  TODO(shoyer): add documentation examples, including a version using ArrayKey
  to wrap np.ndarray attributions.
  """

  @property
  @abc.abstractmethod
  def dims(self) -> tuple[str, ...]:
    """Dimension names of the coordinate."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def shape(self) -> tuple[int, ...]:
    """Shape of the coordinate."""
    raise NotImplementedError()

  @property
  @abc.abstractmethod
  def fields(self) -> dict[str, Field]:
    """A maps from field names to their values."""

  @property
  def sizes(self) -> dict[str, int]:
    """Sizes of all dimensions on this coordinate."""
    return dict(zip(self.dims, self.shape))

  @property
  def ndim(self) -> int:
    """Dimensionality of the coordinate."""
    return len(self.dims)

  @property
  def axes(self) -> tuple[Coordinate, ...]:
    """Tuple of one-dimensional Coordinate objects for each dimension."""
    if self.ndim == 1:
      return (self,)
    else:
      return tuple(SelectedAxis(self, i) for i in range(self.ndim))

  def to_xarray(self) -> dict[str, xarray.Variable]:
    """Convert this coordinate into xarray variables."""
    variables = {}
    dims_set = set(self.dims)
    for name, coord_field in self.fields.items():
      if set(coord_field.dims) <= dims_set:
        # xarray.DataArray coordinate dimensions must be a subset of the
        # dimensions of the associated DataArray, which is not necessarily a
        # constraint for coordax.Field.
        variables[name] = xarray.Variable(coord_field.dims, coord_field.data)
    return variables

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | NoCoordinateMatch:
    """Construct a matching Coordax coordinate from xarray, if possible.

    Args:
      dims: tuple of dimension names. Only the leading dimensions should be
        checks for a match.
      coords: xarray.Coordinates object providing dimension sizes and coordinate
        values.

    Returns:
      A matching instance of this coordinate, or `NoCoordinateMatch` if the
      coordinate does not match the xarray dimensions and coordinates.
    """
    raise NotImplementedError('from_xarray not implemented')


@dataclasses.dataclass(frozen=True)
class ArrayKey:
  """Wrapper for a numpy array to make it hashable."""

  value: np.ndarray

  def __eq__(self, other):
    return (
        isinstance(self, ArrayKey)
        and self.value.dtype == other.value.dtype
        and self.value.shape == other.value.shape
        and (self.value == other.value).all()
    )

  def __hash__(self) -> int:
    return hash((self.value.shape, self.value.tobytes()))


@export
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class SelectedAxis(Coordinate):
  """Coordinate that exposes one dimension of a multidimensional coordinate."""

  coordinate: Coordinate
  axis: int

  def __post_init__(self):
    if self.axis >= self.coordinate.ndim:
      raise ValueError(
          f'Dimension {self.axis=} of {self.coordinate=} is out of bounds'
      )

  @property
  def dims(self) -> tuple[str, ...]:
    """Dimension names of the coordinate."""
    return (self.coordinate.dims[self.axis],)

  @property
  def shape(self) -> tuple[int, ...]:
    """Shape of the coordinate."""
    return (self.coordinate.shape[self.axis],)

  @property
  def fields(self) -> dict[str, Field]:
    """A maps from field names to their values."""
    return self.coordinate.fields

  def __repr__(self):
    return f'coordax.SelectedAxis({self.coordinate!r}, axis={self.axis})'

  def to_xarray(self) -> dict[str, xarray.Variable]:
    """Convert this coordinate into xarray variables."""
    # Override the default method to avoid restricting variables to only those
    # along the selected axis.
    return self.coordinate.to_xarray()


def consolidate_coordinates(*coordinates: Coordinate) -> tuple[Coordinate, ...]:
  """Consolidates coordinates without SelectedAxis objects, if possible."""
  axes = []
  result = []

  def reset_axes():
    result.extend(axes)
    axes[:] = []

  def append_axis(c):
    axes.append(c)
    if len(axes) == c.coordinate.ndim:
      # sucessful consolidation
      result.append(c.coordinate)
      axes[:] = []

  for c in coordinates:
    if isinstance(c, SelectedAxis) and c.axis == 0:
      # new SelectedAxis to consider consolidating
      reset_axes()
      append_axis(c)
    elif (
        isinstance(c, SelectedAxis)
        and axes
        and c.axis == len(axes)
        and c.coordinate == axes[-1].coordinate
    ):
      # continued SelectedAxis to consolidate
      append_axis(c)
    else:
      # coordinate cannot be consolidated
      reset_axes()
      result.append(c)

  reset_axes()

  return tuple(result)


@export
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class CartesianProduct(Coordinate):
  """Coordinate defined as the outer product of independent coordinates."""

  coordinates: tuple[Coordinate, ...]

  def __post_init__(self):
    new_coordinates = []
    for c in self.coordinates:
      new_coordinates.extend(c.axes)
    combined_coordinates = consolidate_coordinates(*new_coordinates)
    if len(combined_coordinates) <= 1:
      raise ValueError('CartesianProduct must contain more than 1 component')
    existing_dims = collections.Counter()
    for c in new_coordinates:
      existing_dims.update(c.dims)
    repeated_dims = [dim for dim, count in existing_dims.items() if count > 1]
    if repeated_dims:
      raise ValueError(f'CartesianProduct components contain {repeated_dims=}')
    object.__setattr__(self, 'coordinates', tuple(new_coordinates))

  def __eq__(self, other):
    # TODO(shoyer): require exact equality of coordinate types?
    if not isinstance(other, CartesianProduct):
      return len(self.coordinates) == 1 and self.coordinates[0] == other
    return isinstance(other, CartesianProduct) and all(
        self.coordinates[i] == other.coordinates[i]
        for i in range(len(self.coordinates))
    )

  @property
  def dims(self):
    return sum([c.dims for c in self.coordinates], start=tuple())

  @property
  def shape(self) -> tuple[int, ...]:
    """Returns the shape of the coordinate axes."""
    return sum([c.shape for c in self.coordinates], start=tuple())

  @property
  def fields(self) -> dict[str, Field]:
    """Returns a mapping from field names to their values."""
    return functools.reduce(
        operator.or_, [c.fields for c in self.coordinates], {}
    )

  @property
  def axes(self) -> tuple[Coordinate, ...]:
    """Returns a tuple of Axis objects for each dimension."""
    return self.coordinates


@export
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class NamedAxis(Coordinate):
  """One dimensional coordinate that only dimension size."""

  name: str
  size: int

  @property
  def dims(self) -> tuple[str, ...]:
    return (self.name,)

  @property
  def shape(self) -> tuple[int, ...]:
    return (self.size,)

  @property
  def fields(self) -> dict[str, Field]:
    return {}

  def __repr__(self):
    return f'coordax.NamedAxis({self.name!r}, size={self.size})'

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | NoCoordinateMatch:
    dim = dims[0]
    if dim in coords:
      return NoCoordinateMatch(
          'can only reconstruct NamedAxis objects from xarray dimensions'
          ' without associated coordinate variables, but found a coordinate'
          f' variable for dimension {dim!r}'
      )
    return cls(dim, size=coords.sizes[dim])


# TODO(dkochkov): consider storing tuple values instead of np.ndarray (which
# could be exposed as a property).
@export
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class LabeledAxis(Coordinate):
  """One dimensional coordinate with custom coordinate values."""

  name: str
  ticks: np.ndarray

  def __post_init__(self):
    object.__setattr__(self, 'ticks', np.asarray(self.ticks))
    if self.ticks.ndim != 1:
      raise ValueError(f'ticks must be a 1D array, got {self.ticks.shape=}')

  @property
  def dims(self) -> tuple[str, ...]:
    return (self.name,)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.ticks.shape

  @property
  def fields(self) -> dict[str, Field]:
    return {self.name: wrap(self.ticks, self)}

  def _components(self):
    return (self.name, ArrayKey(self.ticks))

  def __eq__(self, other):
    return (
        isinstance(other, LabeledAxis)
        and self._components() == other._components()
    )

  def __hash__(self) -> int:
    return hash(self._components())

  def __repr__(self):
    return f'coordax.LabeledAxis({self.name!r}, ticks={self.ticks!r})'

  @classmethod
  def from_xarray(
      cls, dims: tuple[str, ...], coords: xarray.Coordinates
  ) -> Self | NoCoordinateMatch:
    dim = dims[0]
    if dim not in coords:
      return NoCoordinateMatch(
          f'no associated coordinate for dimension {dim!r}'
      )
    if coords[dim].ndim != 1:
      return NoCoordinateMatch(f'coordinate for dimension {dim!r} is not 1D')
    return cls(dim, coords[dim].data)


@export
def compose_coordinates(*coordinates: Coordinate) -> Coordinate:
  """Composes `coords` into a single coordinate system by cartesian product."""
  if not coordinates:
    raise ValueError('No coordinates provided.')
  coordinate_axes = []
  for c in coordinates:
    if isinstance(c, CartesianProduct):
      coordinate_axes.extend(c.coordinates)
    else:
      coordinate_axes.append(c)
  coordinates = consolidate_coordinates(*coordinate_axes)
  if len(coordinates) == 1:
    return coordinates[0]
  return CartesianProduct(coordinates)


def _dimension_names(*names: str | Coordinate) -> tuple[str, ...]:
  """Returns a tuple of dimension names from a list of names or coordinates."""
  dims_or_name_tuple = lambda x: x.dims if isinstance(x, Coordinate) else (x,)
  return sum([dims_or_name_tuple(c) for c in names], start=tuple())


def _validate_matching_coords(
    *axis_order: str | Coordinate,
    coords: dict[str, Coordinate],
):
  """Validates `axis_order` entries match with coordinates in `coords`."""
  coordinates = []
  for axis in axis_order:
    if isinstance(axis, Coordinate):
      if isinstance(axis, CartesianProduct):
        for c in axis.coordinates:
          coordinates.append(c)
      elif axis.ndim > 1:
        for i in range(axis.ndim):
          coordinates.append(SelectedAxis(axis, i))
      else:
        coordinates.append(axis)
  for c in coordinates:
    for dim in c.dims:
      if dim not in coords:
        raise ValueError(f'{dim=} of {c} not found in {coords.keys()=}')
      if coords[dim] != c:
        raise ValueError(f'Coordinate {c} does not match {coords[dim]=}')


@export
def is_field(value) -> TypeGuard[Field]:
  """Returns True if `value` is of type `Field`."""
  return isinstance(value, Field)


@export
def cmap(
    fun: Callable[..., Any], out_axes: dict[str, int] | None = None
) -> Callable[..., Any]:
  """Vectorizes `fun` over coordinate dimensions of ``Field`` inputs.

  Args:
    fun: Function to vectorize over coordinate dimensions.
    out_axes: Optional dictionary from dimension names to axis positions in the
      output. By default, dimension names appear as the trailing dimensions of
      every output, in order of their appearance on the inputs.

  Returns:
    A vectorized version of `fun` that applies original `fun` to locally
    positional dimensions in inputs, while vectorizing over all coordinate
    dimensions. All dimensions over which `fun` is vectorized will be present in
    every output.
  """
  if hasattr(fun, '__name__'):
    fun_name = fun.__name__
  else:
    fun_name = repr(fun)
  if hasattr(fun, '__doc__'):
    fun_doc = fun.__doc__
  else:
    fun_doc = None
  return _cmap_with_doc(fun, fun_name, fun_doc, out_axes)


def _cmap_with_doc(
    fun: Callable[..., Any],
    fun_name: str,
    fun_doc: str | None = None,
    out_axes: dict[str, int] | None = None,
) -> Callable[..., Any]:
  """Builds a coordinate-vectorized wrapped function with a docstring."""

  @functools.wraps(fun)
  def wrapped_fun(*args, **kwargs):
    leaves, treedef = jax.tree.flatten((args, kwargs), is_leaf=is_field)
    field_leaves = [leaf for leaf in leaves if is_field(leaf)]
    all_coords = {}
    for field in field_leaves:
      for dim_name, c in field.coords.items():
        if dim_name in all_coords and all_coords[dim_name] != c:
          other = all_coords[dim_name]
          raise ValueError(f'Coordinates {c=} != {other=} use same {dim_name=}')
        else:
          all_coords[dim_name] = c
    named_array_leaves = [x.named_array if is_field(x) else x for x in leaves]
    fun_on_named_arrays = named_axes.nmap(fun, out_axes=out_axes)
    na_args, na_kwargs = jax.tree.unflatten(treedef, named_array_leaves)
    result = fun_on_named_arrays(*na_args, **na_kwargs)

    def _wrap_field(leaf):
      return Field.from_namedarray(
          named_array=leaf,
          coords={k: all_coords[k] for k in leaf.dims if k is not None},
      )

    return jax.tree.map(_wrap_field, result, is_leaf=named_axes.is_namedarray)

  docstr = (
      f'Dimension-vectorized version of `{fun_name}`. Takes similar arguments'
      f' as `{fun_name}` but accepts and returns Fields in place of arrays.'
  )
  if fun_doc:
    docstr += f'\n\nOriginal documentation:\n\n{fun_doc}'
  wrapped_fun.__doc__ = docstr
  return wrapped_fun


def _swapped_binop(binop):
  """Swaps the order of operations for a binary operation."""

  def swapped(x, y):
    return binop(y, x)

  return swapped


def _wrap_scalar_conversion(scalar_conversion):
  """Wraps a scalar conversion operator on a Field."""

  def wrapped_scalar_conversion(self: Field):
    if self.shape:
      raise ValueError(
          f'Cannot convert a non-scalar Field with {scalar_conversion}'
      )
    return scalar_conversion(self.data)

  return wrapped_scalar_conversion


def _wrap_array_method(name):
  """Wraps an array method on a Field."""

  def func(array, *args, **kwargs):
    return getattr(array, name)(*args, **kwargs)

  array_method = getattr(jax.Array, name)
  wrapped_func = cmap(func)
  functools.update_wrapper(
      wrapped_func,
      array_method,
      assigned=('__name__', '__qualname__', '__annotations__'),
      updated=(),
  )
  wrapped_func.__module__ = __name__
  wrapped_func.__doc__ = (
      'Name-vectorized version of array method'
      f' `{name} <numpy.ndarray.{name}>`. Takes similar arguments as'
      f' `{name} <numpy.ndarray.{name}>` but accepts and returns Fields'
      ' in place of regular arrays.'
  )
  return wrapped_func


@export
@jax.tree_util.register_pytree_node_class
class Field:
  """An array with optional named dimensions and associated coordinates."""

  _named_array: named_axes.NamedArray
  _coords: dict[str, Coordinate]

  def __init__(
      self,
      data: jax.typing.ArrayLike,
      dims: tuple[str | None, ...] | None = None,
      coords: dict[str, Coordinate] | None = None,
  ):
    """Construct a Field.

    Args:
      data: the underlying data array.
      dims: optional tuple of dimension names, with the same length as
        `data.ndim`. Strings indicate named axes, and may not be repeated.
        `None` indicates positional axes. If `dims` is not provided, all axes
        are positional.
      coords: optional mapping from dimension names to associated
        `coordax.Coordinate` objects.
    """
    self._named_array = named_axes.NamedArray(data, dims)
    self._coords = coords or {}
    self._check_valid()

  def _check_valid(self) -> None:
    """Checks that the field coordinates and dimension names are consistent."""
    data_dims = set(self.named_array.named_shape.keys())
    keys_dims = set(self.coords.keys())
    coord_dims = set(sum([c.dims for c in self.coords.values()], start=()))
    if (data_dims != keys_dims) or (data_dims != coord_dims):
      raise ValueError(
          'Field dimension names must be the same across keys and coordinates'
          f' , got {data_dims=}, {keys_dims=} and {keys_dims=}.'
      )

  @classmethod
  def from_namedarray(
      cls,
      named_array: named_axes.NamedArray,
      coords: dict[str, Coordinate] | None = None,
  ) -> Self:
    """Creates a Field from a named array."""
    return cls(named_array.data, named_array.dims, coords)

  @classmethod
  def from_xarray(
      cls,
      data_array: xarray.DataArray,
      coord_types: Sequence[type[Coordinate]] = (LabeledAxis, NamedAxis),
  ) -> Self:
    """Converts an xarray.DataArray into a coordax.Field.

    Args:
      data_array: xarray.DataArray to convert into a Field.
      coord_types: sequence of coordax.Coordinate subclasses with `from_xarray`
        methods defined. The first coordinate class that returns a coordinate
        object (indicating a match) will be used. By default, coordinates are
        constructed out of generic coordax.LabeledAxis and coordax.NamedAxis
        objects.

    Returns:
      A coordax.Field object with the same data as the input xarray.DataArray.
    """
    field = cls(data_array.data)
    dims = data_array.dims
    coords = []

    if not all(isinstance(dim, str) for dim in dims):
      raise TypeError(
          'can only convert DataArray objects with string dimensions to Field'
      )
    dims = typing.cast(tuple[str, ...], dims)

    if not coord_types:
      raise ValueError('coord_types must be non-empty')

    def get_next_match():
      reasons = []
      for coord_type in coord_types:
        result = coord_type.from_xarray(dims, data_array.coords)
        if isinstance(result, Coordinate):
          return result
        assert isinstance(result, NoCoordinateMatch)
        coord_name = coord_type.__module__ + '.' + coord_type.__name__
        reasons.append(f'{coord_name}: {result.reason}')

      reasons_str = '\n'.join(reasons)
      raise ValueError(
          'failed to convert xarray.DataArray to coordax.Field, because no '
          f'coordinate type matched the dimensions starting with {dims}:\n'
          f'{data_array}\n\n'
          f'Reasons why coordinate matching failed:\n{reasons_str}'
      )

    while dims:
      coord = get_next_match()
      coords.append(coord)
      assert coord.ndim > 0  # dimensions will shrink by at least one
      dims = dims[coord.ndim :]

    return field.tag(*coords)

  def to_xarray(self) -> xarray.DataArray:
    """Convert this Field to an xarray.DataArray.

    Returns:
      An xarray.DataArray object with the same data as the input coordax.Field.
      This DataArray will still be wrapping a jax.Array, and have operations
      implemented on jax.Array objects using the Python Array API interface.
    """
    if not all(isinstance(dim, str) for dim in self.dims):
      raise ValueError(
          'can only convert Field objects with fully named dimensions to '
          f'xarray.DataArray objects, got dimensions {self.dims!r}'
      )

    coords = {}
    for coord in self.coords.values():
      for name, variable in coord.to_xarray().items():
        if name in coords and not variable.identical(coords[name]):
          raise ValueError(
              f'inconsistent coordinate fields for {name!r}:\n'
              f'{variable}\nvs\n{coords[name]}'
          )
        coords[name] = variable

    return xarray.DataArray(data=self.data, dims=self.dims, coords=coords)

  @property
  def named_array(self) -> named_axes.NamedArray:
    """The value of the underlying named array."""
    return self._named_array

  @property
  def coords(self) -> dict[str, Coordinate]:
    """The coordinate objects associated with this field."""
    return self._coords

  @property
  def data(self) -> jax.Array:
    """The value of the underlying data array."""
    return self.named_array.data

  @property
  def dtype(self) -> np.dtype:
    """The dtype of the field."""
    return self.named_array.dtype

  @property
  def named_shape(self) -> dict[str, int]:
    """A mapping of axis names to their sizes."""
    return self.named_array.named_shape

  @property
  def positional_shape(self) -> tuple[int, ...]:
    """A tuple of axis sizes for any anonymous axes."""
    return self.named_array.positional_shape

  @property
  def shape(self) -> tuple[int, ...]:
    """A tuple of axis sizes of the underlying data array."""
    return self.named_array.shape

  @property
  def ndim(self) -> int:
    return len(self.dims)

  @property
  def dims(self) -> tuple[str | None, ...]:
    """Named and unnamed dimensions of this array."""
    return self.named_array.dims

  @property
  def named_dims(self) -> tuple[str, ...]:
    """Namd dimensions of this array."""
    return self.named_array.named_dims

  @property
  def named_axes(self) -> dict[str, int]:
    """Mapping from dimension names to axis positions."""
    return self.named_array.named_axes

  @property
  def coord_fields(self) -> dict[str, Field]:
    """A mapping from coordinate field names to their values."""
    return functools.reduce(
        operator.or_, [c.fields for c in self.coords.values()], {}
    )

  def tree_flatten(self):
    """Flatten this object for JAX pytree operations."""
    return [self.named_array], tuple(self.coords.items())

  @classmethod
  def tree_unflatten(cls, coords, leaves) -> Self:
    """Unflatten this object for JAX pytree operations."""
    [named_array] = leaves
    result = object.__new__(cls)
    result._named_array = named_array
    result._coords = dict(coords)
    if isinstance(named_array.data, jnp.ndarray):
      result._check_valid()
    return result

  def unwrap(self, *names: str | Coordinate) -> jax.Array:
    """Extracts underlying data from a field without named dimensions."""
    names = _dimension_names(*names)
    if names != self.named_dims:
      raise ValueError(
          f'Field has {self.named_dims=} but {names=} were requested.'
      )
    return self.data

  def untag(self, *axis_order: str | Coordinate) -> Field:
    """Returns a view of the field with the requested axes made positional."""
    _validate_matching_coords(*axis_order, coords=self.coords)
    untag_dims = _dimension_names(*axis_order)
    named_array = self.named_array.untag(*untag_dims)
    coords = {k: v for k, v in self.coords.items() if k not in untag_dims}
    result = Field.from_namedarray(named_array=named_array, coords=coords)
    return result

  def tag(self, *names: str | Coordinate | ellipsis | None) -> Field:
    """Returns a Field with attached coordinates to the positional axes."""
    tag_dims = _dimension_names(*names)
    tagged_array = self.named_array.tag(*tag_dims)
    coords = {}
    coords.update(self.coords)
    for c in names:
      if isinstance(c, Coordinate):
        if isinstance(c, CartesianProduct):
          for sub_c in c.coordinates:
            [dim] = sub_c.dims  # all sub-coordinates are 1d
            coords[dim] = sub_c
        else:
          for i, dim in enumerate(c.dims):
            coords[dim] = c if c.ndim == 1 else SelectedAxis(c, i)
      elif isinstance(c, str):
        coords[c] = NamedAxis(c, size=tagged_array.named_shape[c])
    result = Field.from_namedarray(tagged_array, coords)
    return result

  # Note: Can't call this "transpose" like Xarray, to avoid conflicting with the
  # positional only ndarray method.
  def order_as(self, *axis_order: str | Coordinate) -> Field:
    """Returns a field with the axes in the given order."""
    _validate_matching_coords(*axis_order, coords=self.coords)
    ordered_dims = _dimension_names(*axis_order)
    ordered_array = self.named_array.order_as(*ordered_dims)
    result = Field.from_namedarray(ordered_array, self.coords)
    return result

  def __repr__(self):
    data_repr = textwrap.indent(repr(self.data), prefix=' ' * 13)[13:]
    if self.coords:
      coords_repr = (
          '{\n'
          + '\n'.join(
              textwrap.indent(f'{k!r}: {v},', prefix=' ' * 12)
              for k, v in self.coords.items()
          )
          + '\n        }'
      )
    else:
      coords_repr = '{}'
    return textwrap.dedent(f"""\
    {type(self).__name__}(
        data={data_repr},
        dims={self.dims},
        coords={coords_repr},
    )""")

  # Convenience wrappers: Elementwise infix operators.
  __lt__ = _cmap_with_doc(operator.lt, 'jax.Array.__lt__')
  __le__ = _cmap_with_doc(operator.le, 'jax.Array.__le__')
  __eq__ = _cmap_with_doc(operator.eq, 'jax.Array.__eq__')
  __ne__ = _cmap_with_doc(operator.ne, 'jax.Array.__ne__')
  __ge__ = _cmap_with_doc(operator.ge, 'jax.Array.__ge__')
  __gt__ = _cmap_with_doc(operator.gt, 'jax.Array.__gt__')

  __add__ = _cmap_with_doc(operator.add, 'jax.Array.__add__')
  __sub__ = _cmap_with_doc(operator.sub, 'jax.Array.__sub__')
  __mul__ = _cmap_with_doc(operator.mul, 'jax.Array.__mul__')
  __truediv__ = _cmap_with_doc(operator.truediv, 'jax.Array.__truediv__')
  __floordiv__ = _cmap_with_doc(operator.floordiv, 'jax.Array.__floordiv__')
  __mod__ = _cmap_with_doc(operator.mod, 'jax.Array.__mod__')
  __divmod__ = _cmap_with_doc(divmod, 'jax.Array.__divmod__')
  __pow__ = _cmap_with_doc(operator.pow, 'jax.Array.__pow__')
  __lshift__ = _cmap_with_doc(operator.lshift, 'jax.Array.__lshift__')
  __rshift__ = _cmap_with_doc(operator.rshift, 'jax.Array.__rshift__')
  __and__ = _cmap_with_doc(operator.and_, 'jax.Array.__and__')
  __or__ = _cmap_with_doc(operator.or_, 'jax.Array.__or__')
  __xor__ = _cmap_with_doc(operator.xor, 'jax.Array.__xor__')

  __radd__ = _cmap_with_doc(_swapped_binop(operator.add), 'jax.Array.__radd__')
  __rsub__ = _cmap_with_doc(_swapped_binop(operator.sub), 'jax.Array.__rsub__')
  __rmul__ = _cmap_with_doc(_swapped_binop(operator.mul), 'jax.Array.__rmul__')
  __rtruediv__ = _cmap_with_doc(
      _swapped_binop(operator.truediv), 'jax.Array.__rtruediv__'
  )
  __rfloordiv__ = _cmap_with_doc(
      _swapped_binop(operator.floordiv), 'jax.Array.__rfloordiv__'
  )
  __rmod__ = _cmap_with_doc(_swapped_binop(operator.mod), 'jax.Array.__rmod__')
  __rdivmod__ = _cmap_with_doc(_swapped_binop(divmod), 'jax.Array.__rdivmod__')
  __rpow__ = _cmap_with_doc(_swapped_binop(operator.pow), 'jax.Array.__rpow__')
  __rlshift__ = _cmap_with_doc(
      _swapped_binop(operator.lshift), 'jax.Array.__rlshift__'
  )
  __rrshift__ = _cmap_with_doc(
      _swapped_binop(operator.rshift), 'jax.Array.__rrshift__'
  )
  __rand__ = _cmap_with_doc(_swapped_binop(operator.and_), 'jax.Array.__rand__')
  __ror__ = _cmap_with_doc(_swapped_binop(operator.or_), 'jax.Array.__ror__')
  __rxor__ = _cmap_with_doc(_swapped_binop(operator.xor), 'jax.Array.__rxor__')

  __abs__ = _cmap_with_doc(operator.abs, 'jax.Array.__abs__')
  __neg__ = _cmap_with_doc(operator.neg, 'jax.Array.__neg__')
  __pos__ = _cmap_with_doc(operator.pos, 'jax.Array.__pos__')
  __invert__ = _cmap_with_doc(operator.inv, 'jax.Array.__invert__')

  # Convenience wrappers: Scalar conversions.
  __bool__ = _wrap_scalar_conversion(bool)
  __complex__ = _wrap_scalar_conversion(complex)
  __int__ = _wrap_scalar_conversion(int)
  __float__ = _wrap_scalar_conversion(float)
  __index__ = _wrap_scalar_conversion(operator.index)

  # elementwise operations
  astype = _wrap_array_method('astype')
  clip = _wrap_array_method('clip')
  conj = _wrap_array_method('conj')
  conjugate = _wrap_array_method('conjugate')
  imag = _wrap_array_method('imag')
  real = _wrap_array_method('real')
  round = _wrap_array_method('round')
  view = _wrap_array_method('view')

  # Intentionally not included: anything that acts on a subset of axes or takes
  # an axis as an argument (e.g., mean). It is ambiguous whether these should
  # act over positional or named axes.
  # TODO(shoyer): re-write some of these with explicit APIs similar to xarray.

  # maybe include some of below with names that signify positional nature?
  # reshape = _wrap_array_method('reshape')
  # squeeze = _wrap_array_method('squeeze')
  # transpose = _wrap_array_method('transpose')
  # T = _wrap_array_method('T')
  # mT = _wrap_array_method('mT')  # pylint: disable=invalid-name


@export
def wrap(array: jax.typing.ArrayLike, *names: str | Coordinate | None) -> Field:
  """Wraps a positional array as a ``Field``."""
  field = Field(array)
  if names:
    field = field.tag(*names)
  return field


@export
def wrap_like(array: jax.typing.ArrayLike, other: Field) -> Field:
  """Wraps `array` with the same coordinates as `other`."""
  array = jnp.asarray(array)
  if array.shape != other.shape:
    raise ValueError(f'{array.shape=} and {other.shape=} must be equal')
  return Field(array, other.dims, other.coords)
