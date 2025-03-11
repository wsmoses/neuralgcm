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

import pickle

from absl.testing import absltest
from absl.testing import parameterized
import chex
from neuralgcm.experimental import coordax
from neuralgcm.experimental import xreader
import neuralgcm.experimental.jax_datetime as jdt
import numpy as np
import xarray


class ReaderTest(parameterized.TestCase):

  def test_evaluation_iterator(self):
    source = xarray.Dataset(
        {'x': ('time', np.arange(10))}, coords={'time': np.arange(10)}
    )
    stencil = xreader.Stencil(start=-2, stop=4, step=2)
    sample_origins = np.array([2, 3, 7])
    data = xreader.evaluation_iterator(source, stencil, sample_origins)
    expected = [
        {'x': np.array([0, 2, 4]), 'time': np.array([0, 2, 4])},
        {'x': np.array([1, 3, 5]), 'time': np.array([1, 3, 5])},
        {'x': np.array([5, 7, 9]), 'time': np.array([5, 7, 9])},
    ]
    actual = [item for item in data]
    chex.assert_trees_all_equal(actual, expected)

  def test_training_iterator_basic(self):
    source = xarray.Dataset(coords={'time': np.arange(100)})
    stencil = xreader.Stencil(start=-2, stop=3, step=1)
    sample_origins = np.arange(2, 100, 5)
    data = xreader.training_iterator(
        source,
        stencil,
        sample_origins,
        buffer_size_in_bytes=100 * 8,
        seed=0,
    )
    expected = np.arange(100)
    actual = np.sort(np.concatenate([x['time'] for x in data]))
    np.testing.assert_equal(actual, expected)

  @parameterized.named_parameters(
      {'testcase_name': 'only_window_shuffle', 'buffer_size_in_bytes': 100 * 8},
      {'testcase_name': 'only_global_shuffle', 'buffer_size_in_bytes': 0},
  )
  def test_training_iterator_shuffling(self, **kwargs):
    source = xarray.Dataset(coords={'time': np.arange(100)})
    stencil = xreader.Stencil(start=-2, stop=3, step=1)
    sample_origins = np.arange(2, 100, 5)
    # same values repeated twice, but in a different order
    data = xreader.training_iterator(
        source, stencil, sample_origins, num_epochs=2, **kwargs
    )
    actual_2x = np.concatenate([x['time'] for x in data])
    expected = np.arange(100)
    np.testing.assert_equal(np.sort(actual_2x[:100]), expected)
    np.testing.assert_equal(np.sort(actual_2x[100:]), expected)
    self.assertFalse((actual_2x[:100] == actual_2x[100:]).all())

  def test_training_iterator_pickle(self):
    source = xarray.Dataset(coords={'time': np.arange(100)})
    stencil = xreader.Stencil(start=-2, stop=3, step=1)
    sample_origins = np.arange(2, 100, 5)
    data = xreader.training_iterator(source, stencil, sample_origins)
    restored = pickle.loads(pickle.dumps(data))
    expected = np.arange(100)
    actual = np.sort(np.concatenate([x['time'] for x in restored]))
    np.testing.assert_equal(actual, expected)

  def test_iterator_with_datetime(self):
    times = np.arange(
        np.datetime64('2000-01-01'),
        np.datetime64('2000-01-04'),
        np.timedelta64(1, 'D'),
    )
    source = xarray.Dataset(coords={'time': ('time', times)})
    stencil = xreader.TimeStencil(start='0h', stop='24h', step='24h')
    sample_origins = times
    data = xreader.evaluation_iterator(source, stencil, sample_origins)
    actual = [item for item in data]
    expected = [
        {'time': jdt.to_datetime(times[0:1])},
        {'time': jdt.to_datetime(times[1:2])},
        {'time': jdt.to_datetime(times[2:3])},
    ]
    chex.assert_trees_all_equal(actual, expected)

  def test_iterator_with_timedelta(self):
    times = np.arange(
        np.timedelta64(1, 'D'),
        np.timedelta64(4, 'D'),
        np.timedelta64(1, 'D'),
    )
    source = xarray.Dataset(coords={'time': ('time', times)})
    stencil = xreader.TimeStencil(
        start='0h', stop='24h', step='24h', closed='both'
    )
    sample_origins = times[:2]
    data = xreader.evaluation_iterator(source, stencil, sample_origins)
    actual = [item for item in data]
    expected = [
        {'time': jdt.to_timedelta(times[0:2])},
        {'time': jdt.to_timedelta(times[1:3])},
    ]
    chex.assert_trees_all_equal(actual, expected)

  def test_coordax_unflattener(self):
    source = xarray.Dataset(
        {
            'foo': (('time', 'x'), np.array([[1, 2], [3, 4], [5, 6]])),
        },
        coords={'time': np.array([0, 10, 20])},
    )
    stencil = xreader.Stencil(start=0, stop=10, step=10, closed='both')
    sample_origins = np.array([0, 10])
    unflattener = xreader.CoordaxUnflattener()
    data = xreader.evaluation_iterator(
        source, stencil, sample_origins, unflattener=unflattener
    )
    actual = [item for item in data]
    expected = [
        {
            'foo': coordax.Field(np.array([[1, 2], [3, 4]]), dims=(None, 'x')),
            'time': coordax.Field(np.array([0, 10])),
        },
        {
            'foo': coordax.Field(np.array([[3, 4], [5, 6]]), dims=(None, 'x')),
            'time': coordax.Field(np.array([10, 20])),
        },
    ]
    chex.assert_trees_all_equal(actual, expected)


if __name__ == '__main__':
  absltest.main()
