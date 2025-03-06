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
"""Xarray data reader for NeuralGCM training."""

from __future__ import annotations

import abc
from collections.abc import Sequence
import concurrent.futures
import dataclasses
import functools
import logging
import math
from typing import Any, Callable, TYPE_CHECKING, TypeVar

import grain.python as grain
from neuralgcm.experimental.xreader import samplers
import numpy as np
import xarray

if TYPE_CHECKING:
  # pylint: disable=g-import-not-at-top,g-bad-import-order
  from neuralgcm.experimental import coordax
  import neuralgcm.experimental.jax_datetime as jdt

# pylint: disable=logging-fstring-interpolation


# Old versions of Grain (namely, 0.2.3) didn't expose WindowShuffleIterDataset
# as a public API.
if not hasattr(grain.experimental, 'WindowShuffleIterDataset'):
  # pylint: disable=g-import-not-at-top,g-bad-import-order
  from grain._src.python.dataset.transformations import shuffle  # pylint: disable=protected-access

  grain.experimental.WindowShuffleIterDataset = shuffle.WindowShuffleIterDataset


T = TypeVar('T')


def _xarray_bytes_per_element(
    source: xarray.Dataset, exclude_dims: set[str]
) -> int:
  bytes_per_element = 0
  for variable in source.values():
    items_per_element = math.prod(
        size for dim, size in variable.sizes.items() if dim not in exclude_dims
    )
    bytes_per_element += variable.dtype.itemsize * items_per_element
  return bytes_per_element


def _calculate_block_size(
    source: xarray.Dataset,
    sample_dims: list[str],
    bytes_per_request: float,
    min_elements_per_request: int = 1,
) -> int:
  """Calculate the size of blocks to read simultaneously from disk."""
  bytes_per_element = _xarray_bytes_per_element(source, set(sample_dims))
  elements_per_request = round(bytes_per_request / bytes_per_element)
  max_elements = math.prod(source.sizes[dim] for dim in sample_dims)
  elements_per_request = min(
      max(elements_per_request, min_elements_per_request), max_elements
  )
  return elements_per_request


def _drop_static_vars(
    dataset: xarray.Dataset, sample_dim: str
) -> xarray.Dataset:
  """Drop fields that are static and do not vary across the sample dimension."""
  vars_to_drop = [k for k, v in dataset.items() if sample_dim not in v.dims]
  return dataset.drop_vars(vars_to_drop)


PyTree = Any


def _thread_pool_loader(max_workers: int = 100):
  """Dataset loader using a large thread pool for concurrency."""
  # We use a separate thread for reading each data variable in each block. This
  # should suffice for maximum concurrency when reading from Zarr, as long as
  # the Zarr store uses internal concurrency for loading each variable.
  executor = concurrent.futures.ThreadPoolExecutor(max_workers)

  def load(dataset: xarray.Dataset) -> xarray.Dataset:
    arrays = executor.map(lambda var: var.values, dataset.values())
    return dataset.copy(data={k: v for k, v in zip(dataset, arrays)})

  return load


@functools.cache
def _array_converter(dtype_kind: str) -> Callable[[np.ndarray], PyTree]:
  # NumPy dtype kinds corresponding to datetime64 and timedelta64:
  # https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
  if dtype_kind in 'mM':
    import neuralgcm.experimental.jax_datetime as jdt  # pylint: disable=g-import-not-at-top

    to_jdt = {'M': jdt.to_datetime, 'm': jdt.to_timedelta}[dtype_kind]
    return to_jdt
  else:
    return lambda x: x


def to_cpu_array(data: np.ndarray) -> np.ndarray | jdt.Datetime | jdt.Timedelta:
  converter = _array_converter(data.dtype.kind)
  return converter(data)


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
    sample_dim: str, coords: dict[str, coordax.Coordinate], arrays: list[Any]
) -> dict[str, coordax.Field]:
  from neuralgcm.experimental import coordax  # pylint: disable=g-import-not-at-top

  # TODO(shoyer): consider removing sample_dim from coords, rather than tagging
  # followed by untagging.
  return {
      name: coordax.Field(array).tag(coord).untag(sample_dim)
      for (name, coord), array in zip(coords.items(), arrays)
  }


@dataclasses.dataclass
class CoordaxUnflattener(Unflattener):
  """Unflatten into a dict of coordax.Field objects."""

  coord_types: Sequence[type[coordax.Coordinate]] | None = None

  def build(self, source: xarray.Dataset) -> Callable[[list[Any]], PyTree]:
    from neuralgcm.experimental import coordax  # pylint: disable=g-import-not-at-top

    # sample_dim is moved to the front via transpose in _prepare_source.
    sample_dim = next(iter(source.values())).dims[0]

    # Coordax is an optional depdenncy for Xreader, so define default values for
    # coord_types here instead of on the dataclass field.
    coord_types = (
        (coordax.LabeledAxis, coordax.DummyAxis)
        if self.coord_types is None
        else self.coord_types
    )
    coords = {}
    for k, data_array in source.items():
      coords[k] = coordax.coordinates_from_xarray(data_array, coord_types)

    return functools.partial(_unflatten_fields, sample_dim, coords)


@dataclasses.dataclass(repr=False, eq=False)
class _XarrayBlockSource(grain.RandomAccessDataSource[T]):
  """Grain data source for reading blocks from an xarray.Dataset."""

  source: xarray.Dataset
  blocks: list[slice]
  unflatten: Callable[[list[Any]], T]

  @functools.cached_property
  def loader(self) -> Callable[[xarray.Dataset], xarray.Dataset]:
    # In principle, it could make sense to support passing alternative
    # loaders, such as xarray_tensorstore.read() or a dask loader that calls
    # .compute(). We don't yet have any use cases where this seems to make a
    # difference, though. (The thread pool loader works as well as
    # xarray_tensorstore.read.)
    return _thread_pool_loader()

  def __getitem__(self, index):
    selection = self.source.isel(time=self.blocks[index])
    loaded = self.loader(selection)
    arrays = [to_cpu_array(x.values) for x in loaded.values()]
    return self.unflatten(arrays)

  def __len__(self):
    return len(self.blocks)

  def __getstate__(self):
    # Cannot pickle the loader because it includes a thread pool.
    return {
        'source': self.source,
        'blocks': self.blocks,
        'unflatten': self.unflatten,
    }


def _prepare_source(source: xarray.Dataset, sample_dim: str) -> xarray.Dataset:
  """Prepare an xarray.Dataset for reading."""
  # TODO(shoyer): support multiple sample dimensions
  if sample_dim not in source.dims:
    raise ValueError(
        'source does not include variables with a'
        f' {sample_dim!r} dimension:\n{source}'
    )
  if sample_dim in source.indexes:
    source = source.reset_index(sample_dim)
  if sample_dim in source.coords:
    source = source.reset_coords(sample_dim)
  source = _drop_static_vars(source, sample_dim)
  source = source.transpose(sample_dim, ...)
  return source


@dataclasses.dataclass
class _UnbatchTransform(grain.experimental.FlatMapTransform):
  max_fan_out: int

  def flat_map(self, element):
    return element


class _Reader:
  """Shared logic for loading an xarray.Dataset."""

  # TODO(shoyer): explore this as a public API?

  def __init__(
      self,
      source: xarray.Dataset,
      sampler: samplers.Sampler,
      *,
      sample_dim: str = 'time',
      block_size_in_bytes: float = 1e8,
      unflattener: Unflattener = ArrayUnflattener(),
  ):
    source = _prepare_source(source, sample_dim)
    unflatten = unflattener.build(source)

    block_size = _calculate_block_size(
        source,
        sample_dims=[sample_dim],
        bytes_per_request=block_size_in_bytes,
        min_elements_per_request=sampler.example_size,
    )
    block_slices = sampler.get_block_slices(
        block_size=block_size, total_size=source.sizes[sample_dim]
    )

    bytes_per_element = _xarray_bytes_per_element(source, {sample_dim})
    bytes_per_example = sampler.example_size * bytes_per_element
    examples_per_block = sampler.examples_per_block(block_size)
    sample_bytes_per_block = bytes_per_example * examples_per_block
    expansion = sample_bytes_per_block / block_size_in_bytes
    logging.info(
        f'picked block_size={block_size}, corresponding to {len(block_slices)} '
        f'blocks with examples_per_block={examples_per_block}, based on '
        f'sampler={sampler} and {block_size_in_bytes=:g}. '
        f'{sample_bytes_per_block=:g} is a {expansion:1.2f}x expansion.'
    )
    self.block_size = block_size
    self.bytes_per_block = bytes_per_example * examples_per_block
    self.bytes_per_example = bytes_per_example
    self.examples_per_block = examples_per_block

    self.block_source = _XarrayBlockSource(
        source=source,
        blocks=block_slices,
        unflatten=unflatten,
    )


def read_timeseries(
    source: xarray.Dataset,
    sampler: samplers.Sampler,
    *,
    sample_dim: str = 'time',
    num_epochs: int | None = 1,
    unflattener: Unflattener = ArrayUnflattener(),
    block_size_in_bytes: float = 1e8,
    read_options: grain.ReadOptions | None = None,
) -> grain.IterDataset:
  """Read a time-series xarray.Dataset into an iterator of samples.

  Args:
    source: lazy xarray.Dataset, e.g., opened from a Zarr file with
      `open_zarr(..., chunks=None)`. All data variables with a 'time' dimension
      will be sampled. Note: setting `chunks=None` to avoid using Dask is
      preferred for optimal performance.
    sampler: specification of what time-series samples of this dataset should
      look like. Currently the only supported sampler is Windower.
    sample_dim: name of the dimension to sample along.
    num_epochs: number of epoch for which to read the dataset, or `None` to read
      indefinitely.
    unflattener: defines how to convert a list of arrays into a pytree.
    block_size_in_bytes: number of bytes to use for each reading a "block" of
      data from the source data. Larger block sizes are more efficient.
    read_options: options to use for reading the Grain dataset. Buffer size here
      indicates the number of blocks to prefetch.

  Returns:
    grain.IterDataset where each element is a pytree of arrays.
  """

  if read_options is None:
    read_options = grain.ReadOptions(num_threads=16, prefetch_buffer_size=32)

  reader = _Reader(
      source=source,
      sampler=sampler,
      sample_dim=sample_dim,
      block_size_in_bytes=block_size_in_bytes,
      unflattener=unflattener,
  )
  # Our strategy of emitting multiple examples per block means that we need to
  # drop into Grain's lower-level Dataset API.
  # We use IterDataset instead of MapDataset because the flat map transform on
  # MapDataset will re-read the entire block for each example, rather than
  # fetching it all at once.
  dataset = grain.MapDataset.source(reader.block_source)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.map(sampler.sample_block)
  dataset = dataset.to_iter_dataset(read_options=read_options)
  dataset = grain.experimental.FlatMapIterDataset(
      dataset, _UnbatchTransform(reader.examples_per_block)
  )
  return dataset


def read_shuffled_shard(
    source: xarray.Dataset,
    sampler: samplers.Sampler,
    *,
    sample_dim: str = 'time',
    num_epochs: int | None = 1,
    unflattener: Unflattener = ArrayUnflattener(),
    block_size_in_bytes: float = 1e8,
    buffer_size_in_bytes: float = 1e10,
    min_buffer_blocks: float = 10,
    read_options: grain.ReadOptions | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
    seed: int = 0,
) -> grain.IterDataset:
  """Read a time-series into an iterator of samples in randomly shuffled order.

  Args:
    source: lazy xarray.Dataset, e.g., opened from a Zarr file with
      `open_zarr(..., chunks=None)`. All data variables with a 'time' dimension
      will be sampled. Note: setting `chunks=None` to avoid using Dask is
      preferred for optimal performance.
    sampler: specification of what time-series samples of this dataset should
      look like.
    sample_dim: name of the dimension to sample along.
    num_epochs: number of epoch for which to read the dataset, or `None` to read
      indefinitely.
    unflattener: defines how to convert a list of arrays into a pytree.
    block_size_in_bytes: number of bytes to use for each reading a "block" of
      data from the source data. Larger block sizes are more efficient.
    buffer_size_in_bytes: number of bytes to use in the shuffle window.
    min_buffer_blocks: minimum number of blocks that must be represented in the
      shuffle buffer, if more than one sample is taken from each block.
      Typically this should be at least as large as the batch size.
    read_options: options to use for reading the Grain dataset. Buffer size here
      indicates the number of blocks to prefetch.
    shard_index: integer index for this shard of the data, in the range `[0,
      shard_count)`. In a multi-host JAX training setup, this should equal
      `jax.process_index()`.
    shard_count: total number of data shards. In a multi-host JAX training
      setup, this should equal `jax.process_count()`.
    seed: seed to use for random number generation.

  Returns:
    grain.IterDataset where each element is a pytree of arrays.
  """
  if shard_index is None and shard_count is None:
    shard_index = 0
    shard_count = 1

  if shard_index is None or shard_count is None:
    raise ValueError('must set both or neither of shard_index and shard_count')

  def _make_reader(block_size_in_bytes):
    reader = _Reader(
        source=source,
        sampler=sampler,
        sample_dim=sample_dim,
        block_size_in_bytes=block_size_in_bytes,
        unflattener=unflattener,
    )
    shuffle_window = round(buffer_size_in_bytes / reader.bytes_per_example)
    logging.info(
        f'picked shuffle window size of {shuffle_window} based on '
        f'{buffer_size_in_bytes=:g}'
    )
    return reader, shuffle_window

  reader, shuffle_window = _make_reader(block_size_in_bytes)

  if shuffle_window:
    examples_per_block = reader.examples_per_block
    buffer_blocks = shuffle_window / examples_per_block
    if examples_per_block > 1 and buffer_blocks < min_buffer_blocks:
      block_size_in_bytes = reader.bytes_per_example
      logging.warning(
          'insufficient diversity in proposed shuffle buffer: '
          f'{examples_per_block=} and {shuffle_window=} means that on average '
          f'only {buffer_blocks:g} blocks will be represented in the shuffle '
          f'buffer, which is less than {min_buffer_blocks=}. Falling back to '
          f'one example per block ({block_size_in_bytes=:g}).'
      )
      reader, shuffle_window = _make_reader(block_size_in_bytes)
      assert reader.examples_per_block == 1

  if read_options is None:
    # We already have multiple threads inside the source, so we don't need to
    # use Grain's internal concurrency.
    buffer_blocks = 2 * round(buffer_size_in_bytes / reader.bytes_per_block)
    read_options = grain.ReadOptions(
        num_threads=1, prefetch_buffer_size=buffer_blocks
    )

  rng = np.random.default_rng(seed)
  global_seed = rng.integers(2**32)
  window_seed = rng.integers(2**32)

  dataset = grain.MapDataset.source(reader.block_source)
  dataset = dataset[shard_index::shard_count]
  dataset = dataset.shuffle(global_seed)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.to_iter_dataset(read_options=read_options)
  dataset = dataset.map(sampler.sample_block)
  dataset = grain.experimental.FlatMapIterDataset(
      dataset, _UnbatchTransform(reader.examples_per_block)
  )
  if shuffle_window:
    dataset = grain.experimental.WindowShuffleIterDataset(
        dataset,
        window_size=shuffle_window,
        seed=window_seed,
    )

  return dataset
