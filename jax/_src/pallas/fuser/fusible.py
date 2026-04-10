# Copyright 2025 The JAX Authors.
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

"""Fusible primitive."""
from functools import partial
from typing import Any

import jax
from jax._src import api_util
from jax._src import core as jax_core
from jax._src import hijax
from jax._src.interpreters import batching
from jax._src import linear_util as lu
from jax._src.traceback_util import api_boundary
from jax._src import tree_util
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas.fuser import fusion as fusion_lib


class Fusible(hijax.VJPHiPrimitive):

  def __init__(self, jaxpr, consts, in_avals, out_aval, params):
    self.jaxpr: jax_core.Jaxpr = jaxpr
    self.consts: list = consts
    self.in_avals = tuple(in_avals)
    self.out_aval = out_aval
    self.params = params
    self.effects = self.jaxpr.effects
    super().__init__()

  def is_high(self, *avals, **params) -> bool:
    return self.jaxpr.is_high

  def expand(self, *args):
    flat_args = tree_util.tree_leaves(args)
    out_flat = jax_core.eval_jaxpr(self.jaxpr, self.consts, *flat_args)
    return tree_util.tree_unflatten(self.out_tree, out_flat)

  def vjp_fwd(self, in_nzs, *args):
    out, vjp_fun = jax.vjp(self.expand, *args)
    return out, vjp_fun

  def vjp_bwd_retval(self, vjp_fun, outgrad):
    return vjp_fun(outgrad)

  def batch(self, axis_data, args, dims):
    if axis_data.size != 1:
      raise NotImplementedError("Fusible does not support non-trivial batching")

    def unbatch_leaf(a, d):
      if d is batching.not_mapped or d is None:
        return a
      return a[d]

    unbatched_args = tree_util.tree_map(unbatch_leaf, args, dims)
    out_unbatched = self(*unbatched_args)

    def batch_leaf(o):
      return o[None]

    out = tree_util.tree_map(batch_leaf, out_unbatched)
    out_dims = tree_util.tree_map(lambda _: 0, self.out_aval)

    return out, out_dims


def _make_trivial_fusion(x: jax.Array) -> fusion_lib.Fusion:
  return fusion_lib.Fusion(
      func=lambda: x,
      in_type=((), {}),
      out_type=jax.typeof(x),
  )


@partial(api_boundary, repro_api_name="fuser.fusible")
def fusible(f=None, *, output_fusion_prefix: Any = True):
  def decorator(f):
    def wrapper(*args):
      def wrapped(*args):
        in_fusions = tree_util.tree_map(_make_trivial_fusion, args)
        output_fusions = tree_util.tree_unflatten(
            tree_util.tree_structure(output_fusion_prefix),
            [None] * len(tree_util.tree_leaves(output_fusion_prefix)),
        )
        return f(*in_fusions, output_fusions)

      flat_args, in_tree = tree_util.tree_flatten(args)
      debug_info = api_util.debug_info('fusible', wrapped, args, {})
      flat_fun, out_tree_thunk = api_util.flatten_fun_nokwargs(
          lu.wrap_init(wrapped, debug_info=debug_info), in_tree
      )
      flat_avals = [jax_core.typeof(x) for x in flat_args]
      jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, flat_avals)
      out_tree = out_tree_thunk()
      out_avals_flat = [v.aval for v in jaxpr.outvars]
      out_aval = tree_util.tree_unflatten(out_tree, out_avals_flat)
      fusible_prim = Fusible(
          jaxpr=jaxpr,
          consts=consts,
          in_avals=tree_util.tree_map(jax_core.typeof, args),
          out_aval=out_aval,
          params=dict(
              output_fusion_prefix=output_fusion_prefix,
              func=f,
          )
      )
      return fusible_prim(*args)

    return wrapper

  if f is not None:
    return decorator(f)
  return decorator
