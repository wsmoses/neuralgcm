# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for converting between xarray and DataObservation objects."""

from typing import Hashable

from dinosaur import spherical_harmonic
from dinosaur import xarray_utils as dino_xarray_utils
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import coordinates
from neuralgcm.experimental import data_specs
from neuralgcm.experimental import scales
from neuralgcm.experimental import typing
from neuralgcm.experimental import units
import neuralgcm.experimental.jax_datetime as jdt
import xarray


AxisName = Hashable
XR_TIME_NAME = 'time'
XR_TIME_IN_MINUTES_SINCE_EPOCH_NAME = 'minutes_since_epoch'
XR_SHORT_LON_NAME = 'lon'
XR_SHORT_LAT_NAME = 'lat'
XR_LON_NAME = 'longitude'
XR_LAT_NAME = 'latitude'


def _wrap_suffix(array: typing.Array, *names: AxisName | cx.Coordinate):
  return cx.wrap(array).tag(..., *names)


verify_grid_consistency = dino_xarray_utils.verify_grid_consistency


def xarray_nondimensionalize(
    ds: xarray.Dataset | xarray.DataArray,
    sim_units: units.SimUnits,
) -> xarray.Dataset:
  return xarray.apply_ufunc(sim_units.nondimensionalize, ds)


def coordinates_from_dataset(
    ds: xarray.Dataset,
    spherical_harmonics_impl=spherical_harmonic.FastSphericalHarmonics,
) -> coordinates.DinosaurCoordinates:
  """Infers coordinates from `ds`."""
  # TODO(dkochkov): Generalize this to take a role of composing coordinates
  # from a collection of candidate coordax.Coordinate instances.
  dino_coords = dino_xarray_utils.coordinate_system_from_dataset(
      ds,
      truncation=dino_xarray_utils.LINEAR,
      spherical_harmonics_impl=spherical_harmonics_impl,
  )
  return coordinates.DinosaurCoordinates.from_dinosaur_coords(dino_coords)


def get_longitude_latitude_names(ds: xarray.Dataset) -> tuple[str, str]:
  """Infers names used for longitude and latitude in the dataset `ds`."""
  if XR_SHORT_LON_NAME in ds.dims and XR_SHORT_LAT_NAME in ds.dims:
    return (XR_SHORT_LON_NAME, XR_SHORT_LAT_NAME)
  if XR_LON_NAME in ds.dims and XR_LAT_NAME in ds.dims:
    return (XR_LON_NAME, XR_LAT_NAME)
  raise ValueError(f'No `lon/lat`|`longitude/latitude` in {ds.coords.keys()=}')


def attach_data_units(
    array: xarray.DataArray,
    default_units: typing.Quantity = typing.units.dimensionless,
) -> xarray.DataArray:
  """Attaches units to `array` based on `attrs.units` or `default_units`."""
  attrs = dict(array.attrs)
  unit = attrs.pop('units', None)
  if unit is not None:
    data = units.parse_units(unit) * array.data
  else:
    data = default_units * array.data
  return xarray.DataArray(data, array.coords, array.dims, attrs=attrs)


def nodal_orography_from_ds(ds: xarray.Dataset) -> xarray.DataArray:
  """Returns orography in nodal representation from `ds`."""
  orography_key = dino_xarray_utils.OROGRAPHY
  if orography_key not in ds:
    ds[orography_key] = (
        ds[dino_xarray_utils.GEOPOTENTIAL_AT_SURFACE_KEY]
        / scales.GRAVITY_ACCELERATION.magnitude
    )
  lon_lat_order = get_longitude_latitude_names(ds)
  orography = attach_data_units(ds[orography_key], typing.units.meter)
  return orography.transpose(*lon_lat_order)


def _xarray_to_data_dict(dataset, *, values: str = 'values'):
  """Extracts `values` from `dataset` and returns as a dictionary."""
  expected_dims_order = (
      dino_xarray_utils.XR_TIME_NAME,
      dino_xarray_utils.XR_LEVEL_NAME,
      dino_xarray_utils.XR_LON_NAME,
      dino_xarray_utils.XR_LAT_NAME,
  )
  dims = []
  for dim in expected_dims_order:
    if dim in dataset.dims:
      dims.append(dim)

  dataset = dataset.transpose(..., *dims)
  data = {}
  for k in dataset:
    assert isinstance(k, str)  # satisfy pytype
    v = getattr(dataset[k], values)
    data[k] = v
  return data


def xarray_to_data_dict(ds: xarray.Dataset, *, values: str = 'values'):
  lon_name, lat_name = get_longitude_latitude_names(ds)
  if lon_name != XR_SHORT_LON_NAME:
    ds = ds.rename({lon_name: XR_SHORT_LON_NAME})
  if lat_name != XR_SHORT_LAT_NAME:
    ds = ds.rename({lat_name: XR_SHORT_LAT_NAME})
  return _xarray_to_data_dict(ds, values=values)


def xarray_to_timed_fields_dict(
    ds: xarray.Dataset,
    *,
    values: str = 'values',
    coords: cx.Coordinate | None = None,
):
  """Extracts data from `ds` to a dictionary of `StaticField`s."""
  if coords is None:
    coords = coordinates_from_dataset(ds)
  data_dict = xarray_to_data_dict(ds, values=values)
  data_dict.pop(XR_TIME_NAME, None)
  time = jdt.to_datetime(ds[XR_TIME_NAME].data)
  observations = {
      k: data_specs.TimedField(_wrap_suffix(v, coords), time)
      for k, v in data_dict.items()
  }
  return observations


def xarray_to_timed_fieldset(
    ds: xarray.Dataset,
    *,
    values: str = 'values',
    coords: cx.Coordinate | None = None,
):
  """Extracts data from `ds` to a dictionary of `StaticField`s."""
  if coords is None:
    coords = coordinates_from_dataset(ds)  # use default unnamed dims for now.
  data_dict = xarray_to_data_dict(ds, values=values)
  data_dict.pop(XR_TIME_NAME, None)
  time = jdt.to_datetime(ds[XR_TIME_NAME].data)
  return data_specs.TimedObservations(
      fields={k: _wrap_suffix(v, coords) for k, v in data_dict.items()},
      timestamp=time,
  )


def timed_field_to_xarray(
    fields_dict: dict[str, data_specs.TimedField[cx.Field]],
    *,
    additional_coords: dict[str, typing.Array] | None = None,
    serialize_coords_to_attrs: bool = False,
):
  """Converts `fields_dict` to an xarray dataset."""
  data_dict = {k: v.field.data for k, v in fields_dict.items()}
  sample_obs = next(iter(fields_dict.values()))
  if sample_obs.timestamp is None:
    raise ValueError(f'observations do not have timestamps: {sample_obs}')
  time = sample_obs.timestamp.to_datetime64()
  level_options = set(['sigma', 'pressure', 'level', 'layer_index'])
  data_dims = sample_obs.field.dims
  level_name = level_options.intersection(set(data_dims))
  if len(level_name) != 1:
    raise ValueError(f'Need exactly 1 match {level_options=} & {data_dims=}')
  (level_name,) = level_name
  vertical = sample_obs.field.coords[level_name]
  horizontal_options = set(
      ['longitude', 'latitude', 'longitudinal_wavenumber', 'total_wavenumber']
  )
  horizontal_names = horizontal_options.intersection(set(data_dims))
  if horizontal_names == set(['longitude', 'latitude']):
    horizontal_names = ('longitude', 'latitude')
  elif horizontal_names == set(['longitudinal_wavenumber', 'total_wavenumber']):
    horizontal_names = ('longitudinal_wavenumber', 'total_wavenumber')
  else:
    raise ValueError(f'not recognized {horizontal_names=}')
  horizontal = cx.compose_coordinates(
      *[sample_obs.field.coords[k] for k in horizontal_names]
  )
  dino_coords = coordinates.DinosaurCoordinates(
      horizontal=horizontal, vertical=vertical
  )
  return dino_xarray_utils.data_to_xarray(
      data={k: v for k, v in data_dict.items()},
      coords=dino_coords.dinosaur_coords,
      times=time,
      sample_ids=None,
      additional_coords=additional_coords,
      serialize_coords_to_attrs=serialize_coords_to_attrs,
  )
