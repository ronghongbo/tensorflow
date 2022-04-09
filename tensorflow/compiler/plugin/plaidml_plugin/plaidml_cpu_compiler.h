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

#ifndef TENSORFLOW_COMPILER_PLUGIN_PLAIDMLCPU_COMPILER_H_
#define TENSORFLOW_COMPILER_PLUGIN_PLAIDMLCPU_COMPILER_H_

#include <memory>

#include "absl/types/span.h"
#include "llvm/Target/TargetMachine.h"
#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/target_machine_features.h"
#include "tensorflow/compiler/xla/service/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace cpu {
struct RecipeInfo {
  RecipeInfo(const RecipeInfo& r) { *this = r; }
  RecipeInfo() {}
  plaidml::exec::Executable plaidml_exe;
  llvm::SmallVector<mlir::Type, 4> min_inputs;
  llvm::SmallVector<mlir::Type, 4> max_inputs;
  llvm::SmallVector<RecipeTensorInfo, 4> inputs;
  llvm::SmallVector<RecipeTensorInfo, 4> outputs;
  bool IsDynamicRecipe() { return min_inputs != max_inputs; }
  bool IsDynamicTensor(unsigned i) {
    assert(i < min_inputs.size() && "Index is out of range");
    return min_inputs[i] != max_inputs[i];
  }
};

// The compiler inherits the behavior of a CPU compiler. The major difference
// is that after HLO optimization, in running the backend, the compiler would
// emit MLIR Linalg dialect, and invoke the PlaidML compiler, which translates
// the emitted Linalg diact into LLVM IR and uses LLVM's JIT infrastructure to
// create an executable "blob" that can then be returned wrapped in
// CpuExecutable and actually invoked.
class PlaidmlCpuCompiler : public CpuCompiler {
 public:
  PlaidmlCpuCompiler() { VLOG(1) << "PlaidML CPU Compiler constructed "; }
  ~PlaidmlCpuCompiler() override {
    VLOG(1) << "PlaidML CPU Compiler destructed ";
  }

  StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  se::Platform::Id PlatformId() const override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

 private:
  Status LowerMLIRModuleToLinalg(
      const HloModule& hlo_module, mlir::ModuleOp mlir_module,
      mlir::MLIRContext& mlir_context);

  TF_DISALLOW_COPY_AND_ASSIGN(PlaidmlCpuCompiler);
};

}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_PLAIDMLCPU_COMPILER_H_
