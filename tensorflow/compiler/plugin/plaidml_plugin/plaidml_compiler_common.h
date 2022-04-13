/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_COMPILER_COMMON_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_COMPILER_COMMON_H_

#include <cstdint>
#include <unordered_map>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
  class Operation;
}

using llvm::SmallVector;
using llvm::Expected;
namespace xla {
namespace plaidml {

using MultiBuffer = std::variant<   //
    std::vector<half_float::half>,  //
    std::vector<float>,             //
    std::vector<double>,            //
    std::vector<int8_t>,            //
    std::vector<int16_t>,           //
    std::vector<int32_t>,           //
    std::vector<int64_t>,           //
    std::vector<uint8_t>,           //
    std::vector<uint16_t>,          //
    std::vector<uint32_t>,          //
    std::vector<uint64_t>>;

using TensorBuffers = std::vector<MultiBuffer>;

struct RecipeInfo {
  RecipeInfo(const RecipeInfo& r) { *this = r; }
  RecipeInfo() {}
  std::unique_ptr<plaidml::Program > plaidml_program = nullptr;
  std::unique_ptr<plaidml::exec::Executable> plaidml_exe = nullptr;
//  TensorBuffers inputs;
//  TensorBuffers outputs;
};

#define ELEMENTS_ATTR_TO_LITERAL(xla_type, plaidml_type) \
  case xla_type:                                     \
    ret = plaidml_type;                                  \
    break;

plaidml::DType XlaTypeToPlaidmlType(PrimitiveType xt) {
  plaidml::DType ret;
  switch (xt) {
    ELEMENTS_ATTR_TO_LITERAL(xla::PRED, PLAIDML_DATA_BOOLEAN)
    ELEMENTS_ATTR_TO_LITERAL(xla::F32, PLAIDML_DATA_FLOAT32)
    ELEMENTS_ATTR_TO_LITERAL(xla::F64, PLAIDML_DATA_FLOAT64)
    ELEMENTS_ATTR_TO_LITERAL(xla::S8, PLAIDML_DATA_INT8)
    ELEMENTS_ATTR_TO_LITERAL(xla::S16, PLAIDML_DATA_INT16)
    ELEMENTS_ATTR_TO_LITERAL(xla::S32, PLAIDML_DATA_INT32)
    ELEMENTS_ATTR_TO_LITERAL(xla::S64, PLAIDML_DATA_INT64)
    ELEMENTS_ATTR_TO_LITERAL(xla::U8, PLAIDML_DATA_UINT8)
    ELEMENTS_ATTR_TO_LITERAL(xla::U16, PLAIDML_DATA_UINT16)
    ELEMENTS_ATTR_TO_LITERAL(xla::U32, PLAIDML_DATA_UINT32)
    ELEMENTS_ATTR_TO_LITERAL(xla::U64, PLAIDML_DATA_UINT64)
    ELEMENTS_ATTR_TO_LITERAL(xla::C64, PLAIDML_DATA_INVALID)
    ELEMENTS_ATTR_TO_LITERAL(xla::C128, PLAIDML_DATA_INVALID)
    ELEMENTS_ATTR_TO_LITERAL(xla::F16, PLAIDML_DATA_FLOAT16)
    ELEMENTS_ATTR_TO_LITERAL(xla::BF16, PLAIDML_DATA_INVALID)
    default:
      VLOG(0) << "Unknown type in conversion from xla to synapse!";
      ret = syn_type_na;
      break;
  }
  return ret;
}

} // namespace plaidml
} // namespace xla
