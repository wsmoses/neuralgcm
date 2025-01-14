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
import functools
import operator
import textwrap
from typing import Any, Callable

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from neuralgcm.experimental import coordax
from neuralgcm.experimental.coordax import core
from neuralgcm.experimental.coordax import testing
import numpy as np


class CoreTest(parameterized.TestCase):

  PRODUCT_XY = coordax.CartesianProduct(
      (coordax.NamedAxis('x', 2), coordax.NamedAxis('y', 3))
  )

  @parameterized.named_parameters(
      dict(
          testcase_name='empty',
          coordinates=(),
          expected=(),
      ),
      dict(
          testcase_name='single_other_axis',
          coordinates=(coordax.NamedAxis('x', 2),),
          expected=(coordax.NamedAxis('x', 2),),
      ),
      dict(
          testcase_name='single_selected_axis',
          coordinates=(
              coordax.SelectedAxis(coordax.NamedAxis('x', 2), axis=0),
          ),
          expected=(coordax.NamedAxis('x', 2),),
      ),
      dict(
          testcase_name='pair_of_other_axes',
          coordinates=(
              coordax.NamedAxis('x', 2),
              coordax.LabeledAxis('y', np.arange(3)),
          ),
          expected=(
              coordax.NamedAxis('x', 2),
              coordax.LabeledAxis('y', np.arange(3)),
          ),
      ),
      dict(
          testcase_name='pair_of_selections_correct',
          coordinates=(
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.SelectedAxis(PRODUCT_XY, axis=1),
          ),
          expected=(PRODUCT_XY,),
      ),
      dict(
          testcase_name='pair_of_selections_wrong_order',
          coordinates=(
              coordax.SelectedAxis(PRODUCT_XY, axis=1),
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
          ),
          expected=(
              coordax.SelectedAxis(PRODUCT_XY, axis=1),
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
          ),
      ),
      dict(
          testcase_name='selection_incomplete',
          coordinates=(coordax.SelectedAxis(PRODUCT_XY, axis=0),),
          expected=(coordax.SelectedAxis(PRODUCT_XY, axis=0),),
      ),
      dict(
          testcase_name='selections_with_following',
          coordinates=(
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.SelectedAxis(PRODUCT_XY, axis=1),
              coordax.NamedAxis('z', 4),
          ),
          expected=(
              PRODUCT_XY,
              coordax.NamedAxis('z', 4),
          ),
      ),
      dict(
          testcase_name='selections_with_preceeding',
          coordinates=(
              coordax.NamedAxis('z', 4),
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.SelectedAxis(PRODUCT_XY, axis=1),
          ),
          expected=(
              coordax.NamedAxis('z', 4),
              PRODUCT_XY,
          ),
      ),
      dict(
          testcase_name='selections_split',
          coordinates=(
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.NamedAxis('z', 4),
              coordax.SelectedAxis(PRODUCT_XY, axis=1),
          ),
          expected=(
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.NamedAxis('z', 4),
              coordax.SelectedAxis(PRODUCT_XY, axis=1),
          ),
      ),
      dict(
          testcase_name='two_selected_axes_consolidate_after',
          coordinates=(
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.SelectedAxis(coordax.NamedAxis('x', 4), axis=0),
          ),
          expected=(
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.NamedAxis('x', 4),
          ),
      ),
      dict(
          testcase_name='two_selected_axes_consolidate_before',
          coordinates=(
              coordax.SelectedAxis(coordax.NamedAxis('x', 4), axis=0),
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
          ),
          expected=(
              coordax.NamedAxis('x', 4),
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
          ),
      ),
  )
  def test_consolidate_coordinates(self, coordinates, expected):
    actual = core.consolidate_coordinates(*coordinates)
    self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='selected_axes_compoents_merge',
          coordinates=(
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.SelectedAxis(PRODUCT_XY, axis=1),
          ),
          expected=PRODUCT_XY,
      ),
      dict(
          testcase_name='selected_axis_simplified',
          coordinates=(
              coordax.SelectedAxis(coordax.NamedAxis('x', 4), axis=0),
              coordax.NamedAxis('z', 7),
          ),
          expected=coordax.CartesianProduct(
              (coordax.NamedAxis('x', 4), coordax.NamedAxis('z', 7))
          ),
      ),
      dict(
          testcase_name='cartesian_product_unraveled',
          coordinates=(
              coordax.NamedAxis('x', 7),
              coordax.CartesianProduct(
                  (coordax.NamedAxis('y', 7), coordax.NamedAxis('z', 4))
              ),
          ),
          expected=coordax.CartesianProduct((
              coordax.NamedAxis('x', 7),
              coordax.NamedAxis('y', 7),
              coordax.NamedAxis('z', 4),
          )),
      ),
      dict(
          testcase_name='consolidate_over_parts',
          coordinates=(
              coordax.SelectedAxis(PRODUCT_XY, axis=0),
              coordax.CartesianProduct((
                  coordax.SelectedAxis(PRODUCT_XY, axis=1),
                  coordax.NamedAxis('z', 4),
              )),
          ),
          expected=coordax.CartesianProduct((
              coordax.NamedAxis('x', 2),
              coordax.NamedAxis('y', 3),
              coordax.NamedAxis('z', 4),
          )),
      ),
  )
  def test_compose_coordinates(self, coordinates, expected):
    actual = core.compose_coordinates(*coordinates)
    self.assertEqual(actual, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='view_with_name',
          array=np.arange(5 * 3).reshape((5, 3)),
          tags=('i', 'j'),
          untags=('i',),
          expected_dims=(None, 'j'),
          expected_named_shape={'j': 3},
          expected_positional_shape=(5,),
          expected_coord_field_keys=set(),
      ),
      dict(
          testcase_name='view_with_name_and_coord',
          array=np.arange(5 * 3).reshape((5, 3, 1)),
          tags=('i', 'j', coordax.LabeledAxis('k', np.arange(1))),
          untags=('j',),
          expected_dims=('i', None, 'k'),
          expected_named_shape={'i': 5, 'k': 1},
          expected_positional_shape=(3,),
          expected_coord_field_keys=set(['k']),
      ),
  )
  def test_field_properties(
      self,
      array: np.ndarray,
      tags: tuple[str | coordax.Coordinate, ...],
      untags: tuple[str | coordax.Coordinate, ...],
      expected_dims: tuple[str | int, ...],
      expected_named_shape: dict[str, int],
      expected_positional_shape: tuple[int, ...],
      expected_coord_field_keys: set[str],
  ):
    """Tests that field properties are correctly set."""
    field = coordax.Field(array).tag(*tags)
    if untags:
      field = field.untag(*untags)
    testing.assert_field_properties(
        actual=field,
        data=array,
        dims=expected_dims,
        named_shape=expected_named_shape,
        positional_shape=expected_positional_shape,
        coord_field_keys=expected_coord_field_keys,
    )

  def test_field_binary_op_sum_simple(self):
    field_a = coordax.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3)))
    field_b = coordax.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3)))
    actual = operator.add(field_a, field_b)
    expected_result = coordax.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3)) * 2)
    testing.assert_fields_allclose(actual=actual, desired=expected_result)

  def test_field_binary_op_sum_aligned(self):
    field_a = coordax.wrap(np.arange(2 * 3).reshape((2, 3)), 'x', 'y')
    field_b = coordax.wrap(np.arange(2 * 3)[::-1].reshape((3, 2)), 'y', 'x')
    actual = operator.add(field_a, field_b)
    expected_result = coordax.wrap(np.array([[5, 4, 3], [7, 6, 5]]), 'x', 'y')
    testing.assert_fields_allclose(actual=actual, desired=expected_result)

  def test_field_binary_op_product_aligned(self):
    field_a = coordax.wrap(np.arange(2 * 3).reshape((2, 3))).tag('x', 'y')
    field_b = coordax.wrap(np.arange(2), 'x')
    actual = operator.mul(field_a, field_b)
    expected_result = coordax.wrap(
        np.arange(2 * 3).reshape((2, 3)) * np.array([[0], [1]])
    ).tag('x', 'y')
    testing.assert_fields_allclose(actual=actual, desired=expected_result)

  def test_field_repr(self):
    expected = textwrap.dedent("""\
        Field(
            data=Array([[1, 2, 3],
                        [4, 5, 6]], dtype=int32),
            dims=('x', 'y'),
            coords={
                'x': coordax.NamedAxis('x', size=2),
                'y': coordax.LabeledAxis('y', ticks=array([7, 8, 9])),
            },
        )""")
    actual = coordax.wrap(
        np.array([[1, 2, 3], [4, 5, 6]]),
        'x',
        coordax.LabeledAxis('y', np.array([7, 8, 9])),
    )
    self.assertEqual(repr(actual), expected)

  @parameterized.named_parameters(
      dict(
          testcase_name='_name_&_name',
          array=np.arange(4),
          tags=('idx',),
          untags=('idx',),
      ),
      dict(
          testcase_name='coord_&_name',
          array=np.arange(4),
          tags=(coordax.NamedAxis('idx', 4),),
          untags=('idx',),
      ),
      dict(
          testcase_name='coord_&_coord',
          array=np.arange(4),
          tags=(coordax.NamedAxis('idx', 4),),
          untags=(coordax.NamedAxis('idx', 4),),
      ),
      dict(
          testcase_name='names_&_partial_name',
          array=np.arange(2 * 3).reshape((2, 3)),
          tags=('x', 'y'),
          untags=('x',),
          full_unwrap=False,
      ),
      dict(
          testcase_name='names_&_partial_coord',
          array=np.arange(2 * 3).reshape((2, 3)),
          tags=('x', 'y'),
          untags=(coordax.NamedAxis('y', 3),),
          full_unwrap=False,
      ),
      dict(
          testcase_name='names_&_coords',
          array=np.arange(2 * 3).reshape((2, 3)),
          tags=('x', 'y'),
          untags=(
              coordax.NamedAxis('x', 2),
              coordax.NamedAxis('y', 3),
          ),
          full_unwrap=True,
      ),
      dict(
          testcase_name='names_&_product_coord',
          array=np.arange(2 * 3).reshape((2, 3)),
          tags=('x', 'y'),
          untags=(
              coordax.compose_coordinates(
                  coordax.NamedAxis('x', 2),
                  coordax.NamedAxis('y', 3),
              ),
          ),
          full_unwrap=True,
      ),
      dict(
          testcase_name='product_coord_&_product_coord',
          array=np.arange(2 * 3).reshape((2, 3)),
          tags=(
              coordax.compose_coordinates(
                  coordax.NamedAxis('x', 2),
                  coordax.NamedAxis('y', 3),
              ),
          ),
          untags=(
              coordax.compose_coordinates(
                  coordax.NamedAxis('x', 2),
                  coordax.NamedAxis('y', 3),
              ),
          ),
          full_unwrap=True,
      ),
      dict(
          testcase_name='mixed_&_names',
          array=np.arange(2 * 3 * 4).reshape((2, 4, 3)),
          tags=('x', coordax.NamedAxis('y', 4), 'z'),
          untags=('y', 'z'),
          full_unwrap=False,
      ),
      dict(
          testcase_name='mixed_&_wrong_names',
          array=np.arange(2 * 3 * 4).reshape((2, 4, 3)),
          tags=('x', coordax.NamedAxis('y_prime', 4), 'z'),
          untags=('y', 'z'),
          full_unwrap=False,
          should_raise_on_untag=True,
      ),
      dict(
          testcase_name='coord_&_wrong_coord_value',
          array=np.arange(9),
          tags=(
              coordax.LabeledAxis(
                  'z',
                  np.arange(9),
              ),
          ),
          untags=(coordax.LabeledAxis('z', np.arange(9) + 1),),
          full_unwrap=False,
          should_raise_on_untag=True,
      ),
  )
  def test_tag_then_untag_by(
      self,
      array: np.ndarray,
      tags: tuple[str | coordax.Coordinate, ...],
      untags: tuple[str | coordax.Coordinate, ...],
      should_raise_on_untag: bool = False,
      full_unwrap: bool = True,
  ):
    """Tests that tag and untag on Field work as expected."""
    with self.subTest('tag'):
      field = coordax.Field(array).tag(*tags)
      expected_dims = sum(
          [
              tag.dims if isinstance(tag, coordax.Coordinate) else (tag,)
              for tag in tags
          ],
          start=tuple(),
      )
      chex.assert_trees_all_equal(field.dims, expected_dims)

    with self.subTest('untag'):
      if should_raise_on_untag:
        with self.assertRaises(ValueError):
          field.untag(*untags)
      else:
        untagged = field.untag(*untags)
        if full_unwrap:
          unwrapped = untagged.unwrap()
          np.testing.assert_array_equal(unwrapped, array)

  def test_cmap_cos(self):
    """Tests that cmap works as expected."""
    inputs = (
        coordax.wrap(np.arange(2 * 3 * 4).reshape((2, 4, 3)))
        .tag('x', 'y', 'z')
        .untag('x')
    )
    actual = coordax.cmap(jnp.cos)(inputs)
    expected_values = jnp.cos(inputs.data)
    testing.assert_field_properties(
        actual=actual,
        data=expected_values,
        dims=(None, 'y', 'z'),
        shape=expected_values.shape,
        coord_field_keys=set(),
    )

  def test_cmap_norm(self):
    """Tests that cmap works as expected."""
    inputs = (
        coordax.wrap(np.arange(2 * 3 * 5).reshape((2, 3, 5)))
        .tag('x', coordax.LabeledAxis('y', np.arange(3)), 'z')
        .untag('x', 'z')
    )
    actual = coordax.cmap(jnp.linalg.norm)(inputs)
    expected_values = jnp.linalg.norm(inputs.data, axis=(0, 2))
    testing.assert_field_properties(
        actual=actual,
        data=expected_values,
        dims=('y',),
        shape=expected_values.shape,
        coord_field_keys=set(['y']),
    )

  def test_jax_transforms(self):
    """Tests that vmap/scan work with Field with leading positional axes."""
    x_coord = coordax.LabeledAxis('x', np.array([2, 3, 7]))
    batch, length = 4, 10
    vmap_axis = coordax.NamedAxis('vmap', batch)
    scan_axis = coordax.LabeledAxis('scan', np.arange(length))

    def initialize(data):
      return coordax.wrap(data, x_coord)

    def body_fn(c, _):
      return (c + 1, c)

    with self.subTest('scan'):
      data = np.zeros(x_coord.shape)
      init = initialize(data)
      _, scanned = jax.lax.scan(body_fn, init, length=length)
      scanned = scanned.tag(scan_axis)
      testing.assert_field_properties(
          actual=scanned,
          dims=('scan', 'x'),
          shape=(length,) + x_coord.shape,
      )

    with self.subTest('vmap'):
      batch_data = np.zeros(vmap_axis.shape + x_coord.shape)
      batch_init = jax.vmap(initialize)(batch_data)
      batch_init = batch_init.tag(vmap_axis)
      testing.assert_field_properties(
          batch_init, dims=('vmap', 'x'), shape=batch_data.shape
      )

    with self.subTest('vmap_of_scan'):
      batch_data = np.zeros(vmap_axis.shape + x_coord.shape)
      batch_init = jax.vmap(initialize)(batch_data)
      scan_fn = functools.partial(jax.lax.scan, body_fn, length=length)
      _, scanned = jax.vmap(scan_fn, in_axes=0)(batch_init)
      scanned = scanned.tag(vmap_axis, scan_axis)
      testing.assert_field_properties(
          actual=scanned,
          dims=('vmap', 'scan', 'x'),
          shape=(batch, length) + x_coord.shape,
      )

    with self.subTest('scan_of_vmap'):
      batch_data = np.zeros(vmap_axis.shape + x_coord.shape)
      batch_init = jax.vmap(initialize)(batch_data)
      vmapped_body_fn = jax.vmap(body_fn)
      scan_fn = functools.partial(jax.lax.scan, vmapped_body_fn, length=length)
      _, scanned = scan_fn(batch_init)
      scanned = scanned.tag(scan_axis, vmap_axis)
      testing.assert_field_properties(
          actual=scanned,
          dims=('scan', 'vmap', 'x'),
          shape=(length, batch) + x_coord.shape,
      )


if __name__ == '__main__':
  absltest.main()
