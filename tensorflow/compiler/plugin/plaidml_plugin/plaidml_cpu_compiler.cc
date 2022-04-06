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

#include "tensorflow/compiler/plugin/plaidml_plugin/plaidml_cpu_compiler.h"

#include <stddef.h>
#include <string.h>

#include <functional>
#include <map>
#include <memory>
#include <stack>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// IWYU pragma: no_include "llvm/Config/Disassemblers.def.inc"
// IWYU pragma: no_include "llvm/Config/Targets.def.inc"
#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"  // from @llvm-project
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"  // from @llvm-project
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"      // from @llvm-project
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"  // from @llvm-project
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"       // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h"              // from @llvm-project
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"         // from @llvm-project
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"     // from @llvm-project
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"                  // from @llvm-project
#include "mlir/Dialect/Func/Transforms/Passes.h"           // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"               // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"                 // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"                 // from @llvm-project
#include "mlir/Dialect/Linalg/Passes.h"                    // from @llvm-project
#include "mlir/Dialect/MemRef/Transforms/Passes.h"         // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"                          // from @llvm-project
#include "mlir/Dialect/Shape/Transforms/Passes.h"          // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"                 // from @llvm-project
#include "mlir/Dialect/Vector/IR/VectorOps.h"              // from @llvm-project
#include "mlir/IR/Builders.h"                              // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"                     // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"                          // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"                  // from @llvm-project
#include "mlir/InitAllDialects.h"                          // from @llvm-project
#include "mlir/Pass/PassManager.h"                         // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Export.h"                    // from @llvm-project
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"                       // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/hlo_to_mlir_hlo.h"
#include "tensorflow/compiler/mlir/xla/ir/xla_framework.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_passes.h"
#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/all_gather_decomposer.h"
#include "tensorflow/compiler/xla/service/all_to_all_decomposer.h"
#include "tensorflow/compiler/xla/service/batch_dot_simplification.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
#include "tensorflow/compiler/xla/service/bfloat16_normalization.h"
#include "tensorflow/compiler/xla/service/bitcast_dtypes_expander.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/cholesky_expander.h"
#include "tensorflow/compiler/xla/service/comparison_expander.h"
#include "tensorflow/compiler/xla/service/conditional_canonicalizer.h"
#include "tensorflow/compiler/xla/service/conditional_simplifier.h"
#include "tensorflow/compiler/xla/service/conditional_to_select.h"
#include "tensorflow/compiler/xla/service/convolution_group_converter.h"
#include "tensorflow/compiler/xla/service/copy_insertion.h"
#include "tensorflow/compiler/xla/service/cpu/buffer_info_util.h"
#include "tensorflow/compiler/xla/service/cpu/compiler_functor.h"
#include "tensorflow/compiler/xla/service/cpu/conv_canonicalization.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_instruction_fusion.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_layout_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_options.h"
#include "tensorflow/compiler/xla/service/cpu/dot_op_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/cpu/ir_emitter.h"
#include "tensorflow/compiler/xla/service/cpu/parallel_task_assignment.h"
#include "tensorflow/compiler/xla/service/cpu/simple_orc_jit.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/dump.h"
#include "tensorflow/compiler/xla/service/dynamic_dimension_simplifier.h"
#include "tensorflow/compiler/xla/service/dynamic_index_splitter.h"
#include "tensorflow/compiler/xla/service/dynamic_padder.h"
#include "tensorflow/compiler/xla/service/eigh_expander.h"
#include "tensorflow/compiler/xla/service/flatten_call_graph.h"
#include "tensorflow/compiler/xla/service/gather_expander.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/indexed_array_analysis.h"
#include "tensorflow/compiler/xla/service/llvm_compiler.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_command_line_options.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/logistic_expander.h"
#include "tensorflow/compiler/xla/service/map_inliner.h"
#include "tensorflow/compiler/xla/service/operand_upcaster.h"
#include "tensorflow/compiler/xla/service/optimization_barrier_expander.h"
#include "tensorflow/compiler/xla/service/qr_expander.h"
#include "tensorflow/compiler/xla/service/reduce_scatter_decomposer.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/result_caster.h"
#include "tensorflow/compiler/xla/service/rng_bit_generator_expander.h"
#include "tensorflow/compiler/xla/service/rng_expander.h"
#include "tensorflow/compiler/xla/service/scatter_expander.h"
#include "tensorflow/compiler/xla/service/sharding_propagation.h"
#include "tensorflow/compiler/xla/service/sharding_remover.h"
#include "tensorflow/compiler/xla/service/slice_sinker.h"
#include "tensorflow/compiler/xla/service/slow_operation_alarm.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
#include "tensorflow/compiler/xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "tensorflow/compiler/xla/service/topk_rewriter.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/service/tree_reduction_rewriter.h"
#include "tensorflow/compiler/xla/service/triangular_solve_expander.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/while_loop_invariant_code_motion.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace xla {
namespace cpu {

void DumpMlirToFile(const HloModule& hlo_module, string suffix,
                    ModuleOp mlir_module) {
  DumpToFileInDirOrStdout(hlo_module, TimestampFor(hlo_module),
                          suffix + ".mlir", llvm_ir::DumpToString(mlir_module));
}

Status PlaidmlCpuCompiler::LowerMLIRModuleToLinalg(
    const HloModule& hlo_module, mlir::ModuleOp mlir_module,
    mlir::MLIRContext& mlir_context) {
  TF_TRACE_SCOPE_ACTIVITY("Lower MHLO To Linalg")

  mlir::PassManager pm(&mlir_context);

  // Resolve all shape constraints (e.g. broadcast constraints that can be
  // proved statically and changed to const witness) early to allow more
  // efficient broadcast operations moving.
  // Move up broadcasting operations to allow for more fusion opportunities.
  pm.addPass(mlir::mhlo::CreateExpandHloTuplesPass("main"));
  pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createLegalizeGeneralDotPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createBroadcastPropagationPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Transform HLO operations to Linalg.
  pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createLegalizeControlFlowPass());
  pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createLegalizeHloToLinalgPass());

  // Lower index cast on tensors to tensor.generate.
  // pm.addNestedPass<mlir::FuncOp>(
  //    mlir::kernel_gen::transforms::CreateLowerIndexCastPass());

  // Lower shape dialect to standard to enable linalg canonicalizations (e.g.
  // use linalg inputs instead of outputs for memref.dim operations).
  // pm.addNestedPass<mlir::FuncOp>(
  //    mlir::kernel_gen::transforms::CreateShapeSimplification());
  // pm.addNestedPass<mlir::FuncOp>(mlir::createShapeToShapeLowering());
  // pm.addPass(mlir::createConvertShapeToStandardPass());
  // pm.addNestedPass<mlir::FuncOp>(mlir::createConvertShapeConstraintsPass());

  // Fuse Linalg on tensors operations.
  // pm.addPass(mlir::createCSEPass());
  // pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
  // pm.addPass(mlir::createCanonicalizerPass());
  // pm.addNestedPass<mlir::FuncOp>(mlir::createLinalgElementwiseOpFusionPass());
  // pm.addNestedPass<mlir::FuncOp>(mlir::createLinalgBufferizePass());
  // pm.addNestedPass<mlir::FuncOp>(mlir::createConvertLinalgToLoopsPass());
  // pm.addPass(mlir::createInlinerPass());

  // Bufferize Linalg on tensors program.
  // Always run canonicalizer (which does dead code removal) before
  // bufferizing anything.
  // pm.addPass(mlir::createCanonicalizerPass());
  // Now bufferize all the compute operations (hlo + linalg) and func
  // signature.
  // pm.addPass(
  //    mlir::kernel_gen::transforms::CreateComputeOpAndFuncBufferizePass());
  // pm.addNestedPass<mlir::FuncOp>(
  //    mlir::kernel_gen::transforms::CreateTiledLoopBufferizePass());
  // Turn tensor constants into global memrefs.
  // TODO(kramerb): Expose the patterns and add them to the bufferize passes.
  // pm.addPass(mlir::createTensorConstantBufferizePass());
  // Always run canonicalizer (which does dead code removal) before
  // bufferizing anything.
  // pm.addPass(mlir::createCanonicalizerPass());
  // pm.addPass(mlir::kernel_gen::transforms::CreateFinalBufferizePass(
  //    /*alignment=*/xla::cpu_function_runtime::Align()));
  // pm.addPass(mlir::createCSEPass());
  // pm.addPass(mlir::createCanonicalizerPass());
  // pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass());
  // pm.addPass(mlir::mhlo::CreateOutlineWithXLAFrameworkPass());

  // Deallocate all temporary buffers.
  // pm.addNestedPass<mlir::FuncOp>(
  //    mlir::bufferization::createBufferDeallocationPass());

  // pm.addPass(mlir::createBufferizationToMemRefPass());

  // Specilize linalg.matmul to linalg.dot, linalg.matvec or linalg.vecmat,
  // and immediately canonicalize to clean up not taken branches.
  // pm.addNestedPass<mlir::FuncOp>(CreateLinalgMatmulSpecializationPass());
  // pm.addPass(mlir::createCanonicalizerPass());

  // Tile and vectorize linalg operation using Linalg Codegen Strategy.
  // pm.addNestedPass<mlir::FuncOp>(CreateCodegenStrategyForMatMulPass());

  // pm.addPass(mlir::createCSEPass());
  // pm.addPass(mlir::createCanonicalizerPass());

  // mlir::VectorTransferToSCFOptions vec_to_scf_options;
  // vec_to_scf_options.unroll = true;
  // pm.addNestedPass<mlir::FuncOp>(
  //     mlir::createConvertVectorToSCFPass(vec_to_scf_options));
  // pm.addNestedPass<mlir::FuncOp>(mlir::arith::createArithmeticExpandOpsPass());
  // pm.addNestedPass<mlir::FuncOp>(mlir::memref::createExpandOpsPass());
  // pm.addPass(mlir::mhlo::CreateLegalizeXLAFrameworkToLLVMPass());
  // pm.addPass(mlir::createMemRefToLLVMPass());
  // pm.addPass(mlir::createConvertSCFToCFPass());
  // pm.addNestedPass<mlir::FuncOp>(mlir::createConvertMathToLLVMPass());
  // pm.addNestedPass<mlir::FuncOp>(
  //     mlir::arith::createConvertArithmeticToLLVMPass());
  // pm.addPass(mlir::createLowerToLLVMPass());
  // pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (pm.run(mlir_module).failed()) {
    DumpMlirToFile(*hlo_module, "linalg", module.get());
    return tensorflow::errors::Internal("Failed to lower MHLO into Linalg");
  }

  // Make @main private so it doesn't clash with other modules.
  mlir_module->walk([&](mlir::LLVM::LLVMFuncOp f) {
    if (f.getName() == "main") {
      f.setLinkageAttr(mlir::LLVM::LinkageAttr::get(
          f.getContext(), mlir::LLVM::Linkage::Private));
    }
  });

  return Status::OK();
}

StatusOr<std::unique_ptr<Executable>> PlaidmlCpuCompiler::RunBackend(
    std::unique_ptr<HloModule> hlo_module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  TF_TRACE_SCOPE
  TF_RET_CHECK(stream_exec != nullptr);

  VLOG(1) << "PlaidmlCpuCompiler RunBackend: " << module->name();
  XLA_SCOPED_LOGGING_TIMER(
      absl::StrFormat("Compiling [%s] for CPU using JIT", module->name()));

  // Prepare MLIR context
  mlir::MLIRContext* context = new mlir::MLIRContext();
  mlir::OwningModuleRef mlir_module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  context->loadAllAvailableDialects();

  // Convert HLO into MLIR HLO
  HloModule* clone = hlo_module->Clone().release();
  auto status = ConvertHloToMlirHlo(mlir_module.get(), clone, true);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to convert HLO to MLIR HLO: " << status;
  }
  free(clone);
  DumpMlirToFile(*hlo_module, "mhlo", mlir_module.get());

  // Convert MLIR HLO into MLIR Linalg
  LowerMLIRModuleToLinalg(hlo_module, mlir_module, compileroptions);

  // Invoke PlaidML compiler to compile MLIR Linalg into an executable 
  VLOG(2) << "Generating executable with PlaidML compiler";
  xla::PmlCpuCompile(mlir_module.release(), hlo_module->unique_id());
  std::unique_ptr<Executable> executable =
      absl::make_unique<PlaidmlCpuExecutable>(std::move(hlo_module));
  return std::move(executable);
}

se::Platform::Id PlaidmlCpuCompiler::PlatformId() const {
  return se::host::kPlaidmlCpuPlatformId;
}

HloCostAnalysis::ShapeSizeFunction CpuCompiler::ShapeSizeBytesFunction() const {
  return PlaidmlCpuExecutable::ShapeSizeBytes;
}

}  // namespace cpu
}  // namespace xla

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      stream_executor::host::kPlaidmlCpuPlatformId,
      []() { return absl::make_unique<xla::cpu::PlaidmlCpuCompiler>(); });
  return true;
}
static bool module_initialized = InitModule();
