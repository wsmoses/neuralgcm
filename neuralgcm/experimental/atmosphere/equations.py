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

"""Modules parameterizing PDEs describing atmospheric processes."""

import dataclasses
from typing import Callable, Sequence

from dinosaur import primitive_equations
from dinosaur import sigma_coordinates
from neuralgcm.experimental.core import coordinates
from neuralgcm.experimental.core import orographies
from neuralgcm.experimental.core import parallelism
from neuralgcm.experimental.core import time_integrators
from neuralgcm.experimental.core import typing
from neuralgcm.experimental.core import units
import numpy as np


class PrimitiveEquations(time_integrators.ImplicitExplicitODE):
  """Equation module for moist primitive equations ."""

  def __init__(
      self,
      coords: coordinates.DinosaurCoordinates,
      sim_units: units.SimUnits,
      reference_temperatures: Sequence[float],
      orography_module: orographies.ModalOrography,
      vertical_advection: Callable[..., typing.Array] = (
          sigma_coordinates.centered_vertical_advection
      ),
      equation_cls=primitive_equations.MoistPrimitiveEquationsWithCloudMoisture,
      include_vertical_advection: bool = True,
      *,
      mesh: parallelism.Mesh,
  ):
    self.coords = coords
    self.orography_module = orography_module
    self.sim_units = sim_units
    self.orography = orography_module
    self.reference_temperatures = reference_temperatures
    self.vertical_advection = vertical_advection
    self.include_vertical_advection = include_vertical_advection
    self.equation_cls = equation_cls
    self.mesh = mesh

  @property
  def primitive_equation(self):
    dinosaur_coords = self.coords.dinosaur_coords
    dinosaur_coords = dataclasses.replace(
        dinosaur_coords, spmd_mesh=self.mesh.spmd_mesh
    )
    return self.equation_cls(
        coords=dinosaur_coords,
        physics_specs=self.sim_units,
        reference_temperature=np.asarray(self.reference_temperatures),
        orography=self.orography_module.modal_orography,
        vertical_advection=self.vertical_advection,
        include_vertical_advection=self.include_vertical_advection,
    )

  @property
  def T_ref(self) -> typing.Array:
    return self.primitive_equation.T_ref

  def explicit_terms(
      self, state: primitive_equations.StateWithTime
  ) -> primitive_equations.StateWithTime:
    return self.primitive_equation.explicit_terms(state)

  def implicit_terms(
      self, state: primitive_equations.StateWithTime
  ) -> primitive_equations.StateWithTime:
    return self.primitive_equation.implicit_terms(state)

  def implicit_inverse(
      self, state: primitive_equations.StateWithTime, step_size: float
  ) -> primitive_equations.StateWithTime:
    return self.primitive_equation.implicit_inverse(state, step_size)
