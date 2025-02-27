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

"""Defines observation operator API and sample operators for NeuralGCM."""

import abc
import dataclasses
from flax import nnx
import jax
import jax.numpy as jnp
from neuralgcm.experimental import coordax as cx
from neuralgcm.experimental import pytree_mappings


class ObservationOperator(nnx.Module, abc.ABC):
  """Base class for observation operators."""

  @abc.abstractmethod
  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    """Returns observations for `query`."""
    ...


@dataclasses.dataclass
class DataObservationOperator(ObservationOperator):
  """Operator that returns pre-computed fields for matching coordinate queries.

  This observation operator matches keys and coordinates in the pre-computed
  dictionary of `coordax.Field`s and the query to the observation operator. This
  operator requires that all `query` entries are of `coordax.Coordinate` type.

  Attributes:
    fields: A dictionary of `coordax.Field`s to return in the observation.
  """

  fields: dict[str, cx.Field]

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    """Returns observations for `query` matched against `self.fields`."""
    del inputs  # unused.
    observations = {}
    is_coordinate = lambda x: isinstance(x, cx.Coordinate)
    valid_keys = list(self.fields.keys())
    for k, v in query.items():
      if k not in valid_keys:
        raise ValueError(f'Query contains {k=} not in {valid_keys}')
      if not is_coordinate(v):
        raise ValueError(
            f'DataObservationOperator only supports coordinate queries, got {v}'
        )
      coord = cx.get_coordinate(self.fields[k])
      if v != coord:
        raise ValueError(f'Query ({k}, {v}) does not match field.{coord=}')
      observations[k] = self.fields[k]
    return observations


@dataclasses.dataclass
class FixedLearnedObservationOperator(ObservationOperator):
  """Operator that computes fixed set of observations using state mapping."""

  coordinate_mapping: pytree_mappings.CoordsStateMapping

  def observe(
      self,
      inputs: dict[str, cx.Field],
      query: dict[str, cx.Field | cx.Coordinate],
  ) -> dict[str, cx.Field]:
    """Returns predicted observations matching `query`."""
    inputs = jax.tree.map(lambda x: x.data, inputs, is_leaf=cx.is_field)
    # TODO(dkochkov): make dummy axis expansion part of the mapping specs.
    maybe_add_dummy = lambda x: jnp.expand_dims(x, 0) if x.ndim == 2 else x
    inputs = jax.tree.map(maybe_add_dummy, inputs)
    predictions = self.coordinate_mapping(inputs)
    # TODO(dkochkov): move squeezing to the mapping.
    predictions = {
        k: jnp.squeeze(v, axis=0) if v.shape[0] == 1 else v
        for k, v in predictions.items()
    }
    is_coordinate = lambda x: isinstance(x, cx.Coordinate)
    prediction_keys = list(predictions.keys())
    observations = {}
    coord = self.coordinate_mapping.coords
    for k, v in query.items():
      if k not in prediction_keys:
        raise ValueError(f'Query contains {k=} not in {prediction_keys}')
      if not is_coordinate(v):
        raise ValueError(
            'FixedLearnedObservationOperator only supports coordinate queries,'
            f' got {v}'
        )
      if (coord == v) or (coord.horizontal == v):
        observations[k] = cx.wrap(predictions[k], v)
      else:
        raise ValueError(f'Query ({k}, {v}) is not compatible with {coord=}')
    return observations
