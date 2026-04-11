/* Copyright 2021 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "nanobind/nanobind.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace {

namespace nb = nanobind;
namespace ffi = xla::ffi;

ffi::Error DceSinkImpl(ffi::AnyBuffer x) { return ffi::Error::Success(); }

XLA_FFI_DEFINE_HANDLER_SYMBOL(dce_sink_ffi, DceSinkImpl,
                              ffi::Ffi::Bind().Arg<ffi::AnyBuffer>());

nb::dict Registrations() {
  nb::dict dict;
  dict["dce_sink_ffi"] = EncapsulateFunction(dce_sink_ffi);
  return dict;
}

NB_MODULE(_dce_sink, m) { m.def("registrations", &Registrations); }

}  // namespace
}  // namespace jax
