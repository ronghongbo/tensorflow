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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_CPU_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_CPU_EXECUTABLE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"
#include "tensorflow/compiler/xla/service/custom_call_status_internal.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "plaidml/pmlc/compiler/program.h"

namespace xla {
namespace cpu {

class PlaidmlCpuExecutable : public Executable {
 public:
  explicit PlaidmlCpuExecutable(std::unique_ptr<HloModule> hlo_module);

  StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile) override;
  static int64_t ShapeSizeBytes(const Shape& shape);
  RecipeInfo& getRecipeInfo(){return recipe_info_;}
  void setRecipeInfo(RecipeInfo& recipe_info) { recipe_info_ = recipe_info; }

 private:
      RecipeInfo recipe_info_;

 TF_DISALLOW_COPY_AND_ASSIGN(PlaidmlCpuExecutable);
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_CPU_EXECUTABLE_H_
