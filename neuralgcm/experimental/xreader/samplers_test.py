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

import chex
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized
from neuralgcm.experimental import xreader



class SamplersTest(parameterized.TestCase):


  @parameterized.parameters(
      {
          'example_size': 5,
          'total_size': 5,
          'block_size': 5,
          'expected': [slice(0, 5)],
      },
      {
          'example_size': 5,
          'total_size': 10,
          'block_size': 10,
          'expected': [slice(0, 10)],
      },
      {
          'example_size': 5,
          'total_size': 8,
          'block_size': 5,
          'expected': [slice(0, 5), slice(1, 6), slice(2, 7), slice(3, 8)],
      },
      {
          'example_size': 5,
          'total_size': 10,
          'block_size': 5,
          'stride_between_windows': 2,
          'expected': [slice(0, 5), slice(2, 7), slice(4, 9)],
      },
      {
          'example_size': 5,
          'total_size': 8,
          'block_size': 5,
          'first_window_offset': 1,
          'expected': [slice(1, 6), slice(2, 7), slice(3, 8)],
      },
      {
          'example_size': 5,
          'total_size': 8,
          'block_size': 5,
          'stride_between_windows': 2,
          'first_window_offset': 1,
          'expected': [slice(1, 6), slice(3, 8)],
      },
      {
          'example_size': 3,
          'total_size': 10,
          'block_size': 6,
          'expected': [slice(0, 6), slice(4, 10)],
      },
      {
          'example_size': 3,
          'total_size': 10,
          'block_size': 6,
          'stride_between_windows': 2,
          # [0 1 2] [2 3 4] | [4 5 6] [6 7 8]
          'expected': [slice(0, 5), slice(4, 9)],
      },
      {
          'example_size': 3,
          'total_size': 10,
          'block_size': 6,
          'output_window_stride': 2,
          'stride_between_windows': 1,
          # [0 2 4] [1 3 5] | [2 4 6] [3 5 7] | [4 6 8] [5 7 9]
          'expected': [slice(0, 6), slice(2, 8), slice(4, 10)],
      },
      {
          'example_size': 2,
          'total_size': 10,
          'block_size': 6,
          'output_window_stride': 2,
          'stride_between_windows': 2,
          # [0 2] [2 4] | [4 6] [6 8]
          'expected': [slice(0, 5), slice(4, 9)],
      },
  )
  def test_windower_get_block_slices(
      self, block_size, total_size, expected, **kwargs
  ):
    actual = xreader.Windower(**kwargs).get_block_slices(block_size, total_size)
    self.assertEqual(actual, expected)

  @parameterized.parameters(
      {
          'inputs': np.arange(5),
          'example_size': 5,
          'stride_between_windows': 1,
          'expected': [np.arange(5)],
      },
      {
          'inputs': np.arange(5),
          'example_size': 3,
          'stride_between_windows': 1,
          'output_window_stride': 2,
          'expected': [np.array([0, 2, 4])],
      },
      {
          'inputs': np.arange(10),
          'example_size': 6,
          'stride_between_windows': 2,
          'expected': [
              np.array([0, 1, 2, 3, 4, 5]),
              np.array([2, 3, 4, 5, 6, 7]),
              np.array([4, 5, 6, 7, 8, 9]),
          ],
      },
      {
          'inputs': np.arange(10),
          'example_size': 4,
          'stride_between_windows': 1,
          'output_window_stride': 2,
          'expected': [
              np.array([0, 2, 4, 6]),
              np.array([1, 3, 5, 7]),
              np.array([2, 4, 6, 8]),
              np.array([3, 5, 7, 9]),
          ],
      },
      {
          'inputs': np.arange(9),
          'example_size': 3,
          'stride_between_windows': 2,
          'output_window_stride': 2,
          'expected': [
              np.array([0, 2, 4]),
              np.array([2, 4, 6]),
              np.array([4, 6, 8]),
          ],
      },
      {
          'inputs': np.arange(10),
          'example_size': 3,
          'stride_between_windows': 1,
          'output_window_stride': 3,
          'expected': [
              np.array([0, 3, 6]),
              np.array([1, 4, 7]),
              np.array([2, 5, 8]),
              np.array([3, 6, 9]),
          ],
      },
      {
          'inputs': {
              'x': np.array([1, 2, 3]),
              'y': np.array([[1, 2], [3, 4], [5, 6]]),
          },
          'example_size': 2,
          'stride_between_windows': 1,
          'expected': [
              {'x': np.array([1, 2]), 'y': np.array([[1, 2], [3, 4]])},
              {'x': np.array([2, 3]), 'y': np.array([[3, 4], [5, 6]])},
          ],
      },
  )
  def test_windower_sample_block(self, inputs, expected, **kwargs):
    actual = xreader.Windower(**kwargs).sample_block(inputs)
    chex.assert_trees_all_equal(actual, expected)

if __name__ == "__main__":
  absltest.main()
