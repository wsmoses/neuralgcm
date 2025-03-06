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
"""Samplers for producing examples from blocks of data."""

from __future__ import annotations

import abc
import dataclasses
from typing import Any

import jax


PyTree = Any


class Sampler(abc.ABC):
  """Base class for sampling from blocks.

  Attributes:
    example_size: size of each example along the sampled dimension.
  """

  example_size: int

  @abc.abstractmethod
  def get_block_slices(self, block_size: int, total_size: int) -> list[slice]:
    """Returns a list of slices bounding blocks to sample from.

    Args:
      block_size: desired size of blocks to read from disk.
      total_size: total size of the dimension being sampled along.

    Returns:
      Slices for each block to sample from.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def sample_block(self, block: PyTree) -> list[PyTree]:
    """Returns sample pytrees from block pytrees.

    Args:
      block: pytree of arrays to sample from.

    Returns:
      List of pytrees of arrays sampled from this block.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def examples_per_block(self, block_size: int) -> int:
    """Number of examples per block."""
    raise NotImplementedError


def _leading_axis_size(block: PyTree) -> int:
  return jax.tree.leaves(block)[0].shape[0]


@dataclasses.dataclass(init=False)
class Windower(Sampler):
  """Sample rolling windows along the first axis.

  Attributes:
    example_size: size of output windows.
    output_window_stride: separation between between observations within a
      window.
    stride_between_windows: offset between starting sequential windows. By
      default, matches output_window_stride.
    first_window_offset: offset of starting the first window.
  """

  example_size: int
  output_window_stride: int
  stride_between_windows: int
  first_window_offset: int

  def __init__(
      self,
      example_size: int,
      *,
      output_window_stride: int = 1,
      stride_between_windows: int | None = None,
      first_window_offset: int = 0,
  ):
    if stride_between_windows is None:
      stride_between_windows = output_window_stride
    self.example_size = example_size
    self.output_window_stride = output_window_stride
    self.stride_between_windows = stride_between_windows
    self.first_window_offset = first_window_offset

  def get_block_slices(self, block_size: int, total_size: int) -> list[slice]:
    example_size = self.example_size
    output_window_stride = self.output_window_stride
    stride_between_windows = self.stride_between_windows
    first_sample_offset = self.first_window_offset

    assert stride_between_windows >= 1
    assert output_window_stride >= 1
    assert first_sample_offset >= 0

    sample_input_size = (
        range(0, example_size * output_window_stride, output_window_stride)[-1]
        + 1
    )
    assert 0 < sample_input_size <= block_size <= total_size

    sample_stop = 0  # unused

    # first block
    block_start = first_sample_offset
    block_stop = first_sample_offset + block_size

    slices = []

    # iterate through all slices, in order
    for start in range(
        first_sample_offset,
        total_size - sample_input_size + 1,
        stride_between_windows,
    ):
      prev_sample_stop = sample_stop
      sample_stop = start + sample_input_size

      if sample_stop > block_stop:
        # append the previous block
        assert prev_sample_stop > 0
        slices.append(slice(block_start, prev_sample_stop))

        # begin new block
        block_start = start
        block_stop = start + block_size

    if sample_stop > block_start:
      # append the final block
      slices.append(slice(block_start, sample_stop))

    return slices

  def _all_slice_starts(self, block_size: int) -> range:
    stop = block_size - self.output_window_stride * (self.example_size - 1)
    return range(0, stop, self.stride_between_windows)

  def sample_block(self, block: PyTree) -> list[PyTree]:
    block_size = _leading_axis_size(block)
    starts = self._all_slice_starts(block_size)
    size = self.example_size * self.output_window_stride
    stride = self.output_window_stride
    return [
        jax.tree.map(lambda x: x[start : start + size : stride], block)  # pylint: disable=cell-var-from-loop
        for start in starts
    ]

  def examples_per_block(self, block_size: int) -> int:
    return len(self._all_slice_starts(block_size))


@dataclasses.dataclass
class WindowerAtOffsets(Sampler):
  """Sample rolling windows along the first axis at specified offsets."""

  example_size: int
  window_offsets: list[int]
  output_window_stride: int = 1

  def get_block_slices(self, block_size: int, total_size: int) -> list[slice]:
    stride = self.output_window_stride
    # TODO(shoyer): optimize this Sampler to read more than 1 example per block.
    # This suffices for now because generally we cache evaluation data.
    sample_input_size = range(0, self.example_size * stride, stride)[-1] + 1
    assert 0 < sample_input_size <= block_size <= total_size
    slices = []
    for start in self.window_offsets:
      stop = start + sample_input_size
      if stop > total_size:
        raise ValueError(
            f'offset at {start} needs data through {stop=}, which is beyond'
            f' {total_size=}'
        )
      slices.append(slice(start, stop))
    return slices

  def sample_block(self, block: PyTree) -> list[PyTree]:
    return [jax.tree.map(lambda x: x[:: self.output_window_stride], block)]

  def examples_per_block(self, block_size: int) -> int:
    del block_size  # unused
    return 1
