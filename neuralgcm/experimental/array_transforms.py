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


"""Modules that implement transforms for JAX arrays."""

from __future__ import annotations

from typing import Protocol, Sequence

import jax


class ArrayTransform(Protocol):
  """Abstract base class for JAX array transforms."""

  def __call__(self, x: jax.Array) -> jax.Array:
    ...


class Identity():
  """Identity transform."""

  def __call__(self, x: jax.Array) -> jax.Array:
    return x


class CombinedArrayTransform():
  """Combines a sequence of transforms.

  Order matters in non-linear transforms.
  """

  def __init__(self, transforms: Sequence[ArrayTransform]):
    self.transforms = transforms

  def __call__(self, x: jax.Array) -> jax.Array:
    for transform in self.transforms:
      x = transform(x)
    return x
