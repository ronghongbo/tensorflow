/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
#include "tensorflow/compiler/plugin/plaidml_plugin/plaidml_cpu_executable.h"

#include <cstdint>
#include <type_traits>
#include <vector>

#include "plaidml_cpu_executor.h"
#include "plaidml_cpu_stream.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/logical_buffer.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/service/xla_debug_info_manager.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/host/host_stream.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"
#include "runtime_tracing.h"
namespace xla {
namespace cpu {
PlaidmlCpuExecutable::PlaidmlCpuExecutable(std::unique_ptr<HloModule> hlo_module)
    : Executable(std::move(hlo_module), /*hlo_profile_printer_data=*/nullptr,
                 /*hlo_profile_index_map=*/nullptr) {}

StatusOr<ExecutionOutput> PlaidmlCpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
  TF_TRACE_SCOPE

  if (GetRootValueSet().IsAmbiguous()) {
    return Unimplemented("Points-to set of root instruction is ambiguous");
  }

  if (hlo_module_) {
    const HloComputation* entry_comp = hlo_module_->entry_computation();
    CHECK_EQ(entry_comp->num_parameters(), arguments.size())
        << "Wrong number of arguments passed when running executable";
    for (int64_t i = 0; i < entry_comp->num_parameters(); ++i) {
      const Shape& expected_shape =
          entry_comp->parameter_instruction(i)->shape();
      const Shape& actual_shape = arguments[i].Buffers().shape();
      TF_RET_CHECK(
          ShapeUtil::DynamicShapeIsCompatible(actual_shape, expected_shape))
          << "Shape mismatch on argument " << i << ", "
          << expected_shape.ToString(/*print_layout=*/true) << " vs. "
          << actual_shape.ToString(/*print_layout=*/true);
    }
  }

  auto* host_stream = dynamic_cast<se::host::HostStream*>(
      run_options->stream()->implementation());
  se::Stream* stream = run_options->stream();
  se::DeviceMemoryAllocator* memory_allocator = run_options->allocator();
  RecipeInfo& recipe_info = this->getRecipeInfo();
  std::unique_ptr<plaidml::compiler::Program> plaidml_program = recipe_info.plaidml_program;
  std::unique_ptr<plaidml::exec::Executable> plaidml_exe = recipe_info.plaidml_exe;
  
  // Prepare input and output buffers in the formats required by PlaidML
  VLOG(2) << "recieved " << arguments.size() << " arguments (execution inputs)";
  std::vector<plaidml::Buffer> input_plaidml_buffers;
  input_plaidml_buffers.reserve(arguments.size());
  for (auto& argument : arguments) {
      const ShapeTree<MaybeOwningDeviceMemory>& buffers = argument.Buffers();
      xla::Shape = buffers.shape();
      plaidml::DType plaidml_type = xlaTypeToPlaidmlType(shape.element_type());
      auto in_it = buffers.begin();
      for (; in_it != buffers.end(); ++in_it) {  // do we execpt to see only a single buffer per
                                // argument?...
        se::DeviceMemoryBase device_memopry_base = in_it->second.AsDeviceMemoryBase();
        char* data = (char *)device_memopry_base.opaque();
        size_t size = (size_t) device_memopry_base.size();
        plaidml::TensorShape plaidml_shape(plaidml_type, std::vector<int64_t>().push_back(size));
        plaidml::Buffer plaidml_buffer(data, size, plaidml_shape);
        input_plaidml_buffers.push_back(plaidml_buffer);
      }
  }
  std::vector<plaidml::Buffer> output_plaidml_buffers;
    auto inputShapes = program.inputs();
    ASSERT_EQ(inputs.size(), inputShapes.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      std::visit(
          [&](auto&& vec) {
            Buffer buffer{vec, inputShapes[i]};
            inputBuffers.emplace_back(buffer);
          },
          inputs[i]);
    }

    for (auto shape : program.outputs()) {
      outputBuffers.emplace_back(shape);
    }
    auto executable = exec::Executable(program);
    executable.run(inputBuffers, outputBuffers);
     input_buffers.push_back(BuildTensorBuffers(buffers));
+      auto in_it = buffers.begin();
+      auto out_it = argument_buffers.back().buffers().begin();
+      for (; in_it != buffers.end();
+          ++in_it, ++out_it) {  // do we execpt to see only a single buffer per
+                                // argument?...
+        out_it->second = in_it->second.AsDeviceMemoryBase();
+
+        persistentTensorInfo[pt].pTensorAddress =
+            (uint64_t)out_it->second.opaque();
+        persistentTensorInfo[pt].tensorType = DATA_TENSOR;
+
+        persistentTensorInfo[pt].tensorName =
+            getRecipeInfo().inputs[pt].name.c_str();
+        for (int i = 0; i < argument.shape().rank(); i++) {
+          persistentTensorInfo[pt].tensorSize[i] = argument.shape().dimensions(i);
+        }
+        VLOG(1) << "buffer " << pt << " located at " << std::hex
+                << (uint64_t)out_it->second.opaque() << " for buffer "
+                << persistentTensorInfo[pt].tensorName;
+        pt++;
+      }
+    }

  const ShapeTree<MaybeOwningDeviceMemory>& Buffers() const { return buffers_; }


  std::vector<Buffer> outputBuffers;

+  int device_ordinal = run_options->device_ordinal();
+  if (device_ordinal < 0) {
+    device_ordinal = stream->parent()->device_ordinal();
+  }


    TF_ASSIGN_OR_RETURN(
      std::vector<MaybeOwningDeviceMemory> buffers,
      CreateBufferTable(memory_allocator, stream->parent()->device_ordinal(),
                        arguments));

  TF_ASSIGN_OR_RETURN(
      ExecutionOutput result,
      CreateResultShapedBuffer(run_options, absl::MakeSpan(buffers),
                               absl::MakeSpan(arguments)));

  auto inputShapes = plaidml_program.inputs();
  ASSERT_EQ(inputs.size(), inputShapes.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      std::visit(
          [&](auto&& vec) {
            Buffer buffer{vec, inputShapes[i]};
            inputBuffers.emplace_back(buffer);
          },
          inputs[i]);
    }

    for (auto shape : program.outputs()) {
      outputBuffers.emplace_back(shape);
    }

  struct AsyncRunTask {
    PlaidmlCpuExecutable* executable;
    ServiceExecutableRunOptions run_options;

    Status operator()() {
      return plaidml_exe->run(inputBuffers, outputBuffers);
    }
  };
  host_stream->EnqueueTaskWithStatus(
      AsyncRunTask{plaidml_exe, *run_options});

  MarkToBeReleasedArguments(absl::MakeSpan(arguments), result);
  return std::move(result);



  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor = stream->parent();
  const se::Platform* platform = executor->platform();
  se::hpu::XlaHpuExecutor* ex =
      dynamic_cast<se::hpu::XlaHpuExecutor*>(executor->implementation());
  synStreamHandle s = ex->getComputeStream();
  se::hpu::HpuStream* hs =
      dynamic_cast<se::hpu::HpuStream*>(stream->implementation());
  hs->setStream(s, se::hpu::hs_compute);
  synDeviceId devId = ex->getDevId();

  VLOG(2) << "recieved " << arguments.size() << " arguments";
  std::vector<ShapedBuffer> argument_buffers;
  argument_buffers.reserve(arguments.size());
  int device_ordinal = run_options->device_ordinal();
  if (device_ordinal < 0) {
    device_ordinal = stream->parent()->device_ordinal();
  }

  const char* env = getenv("HPU_NO_COMPILE");
  if (env != nullptr) {
    se::DeviceMemoryAllocator* memory_allocator = run_options->allocator();
    TF_ASSIGN_OR_RETURN(se::OwningDeviceMemory out,
                        memory_allocator->Allocate(
                            device_ordinal, ShapeSizeBytes(result_shape())));

    ScopedShapedBuffer scopedBuffer(result_shape(), run_options->allocator(),
                                    device_ordinal);
    scopedBuffer.set_buffer(std::move(out), {0});
    ExecutionOutput result(std::move(scopedBuffer));
    MarkToBeReleasedArguments(absl::MakeSpan(arguments), result);
    return std::move(result);
  }
  uint64_t workspaceSize;
  auto recipe = getRecipeInfo().recipe_handle;
  CHECK_SYN1(synWorkspaceGetSize, &workspaceSize, recipe);

  CHECK_LE(workspaceSize, se::hpu::HPU_WS);

  uint64_t pWorkspace = ex->getWorkspacePtr();

  // if (workspaceSize > 0) {
  //   VLOG(1) << "allocating workspace on device. size=" << std::hex
  //           << workspaceSize;

  // se::OwningDeviceMemory workspace;
  //   TF_ASSIGN_OR_RETURN(
  //       workspace, memory_allocator->Allocate(device_ordinal,
  //       workspaceSize));
  //   pWorkspace = (uint64_t)workspace->opaque();
  // }
  se::DeviceMemoryAllocator* memory_allocator = run_options->allocator();
  std::size_t num_outputs =
      result_shape().IsTuple() ? result_shape().tuple_shapes_size() : 1;
  synLaunchTensorInfo* persistentTensorInfo =
      new synLaunchTensorInfo[arguments.size() + num_outputs];
  std::vector<se::OwningDeviceMemory> output_tensors(num_outputs);
  {
    TF_TRACE_SCOPE_ACTIVITY("HPU::Patching")
    uint32_t pt = 0;
    for (auto& argument : arguments) {
      const ShapeTree<MaybeOwningDeviceMemory>& buffers = argument.Buffers();
      argument_buffers.push_back(ShapedBuffer(buffers.shape(), device_ordinal));
      auto in_it = buffers.begin();
      auto out_it = argument_buffers.back().buffers().begin();
      for (; in_it != buffers.end();
          ++in_it, ++out_it) {  // do we execpt to see only a single buffer per
                                // argument?...
        out_it->second = in_it->second.AsDeviceMemoryBase();

        persistentTensorInfo[pt].pTensorAddress =
            (uint64_t)out_it->second.opaque();
        persistentTensorInfo[pt].tensorType = DATA_TENSOR;

        persistentTensorInfo[pt].tensorName =
            getRecipeInfo().inputs[pt].name.c_str();
        for (int i = 0; i < argument.shape().rank(); i++) {
          persistentTensorInfo[pt].tensorSize[i] = argument.shape().dimensions(i);
        }
        VLOG(1) << "buffer " << pt << " located at " << std::hex
                << (uint64_t)out_it->second.opaque() << " for buffer "
                << persistentTensorInfo[pt].tensorName;
        pt++;
      }
    }

    VLOG(2) << "now allocating buffer for result, num_outputs: " << num_outputs;

    for (int i = 0; i < num_outputs; ++i, ++pt) {
      const Shape& shape = result_shape().IsTuple()
                              ? result_shape().tuple_shapes(i)
                              : result_shape();
      TF_ASSIGN_OR_RETURN(
          output_tensors[i],
          memory_allocator->Allocate(device_ordinal, ShapeSizeBytes(shape)));
      VLOG(2) << "result tensor num is " << pt;
      persistentTensorInfo[pt].pTensorAddress =
          (uint64_t)output_tensors[i]->opaque();
      persistentTensorInfo[pt].tensorName =
          getRecipeInfo().outputs[i].name.c_str();
      persistentTensorInfo[pt].tensorType = DATA_TENSOR;
      for (int r = 0; r < shape.rank(); r++) {
        persistentTensorInfo[pt].tensorSize[r] = shape.dimensions(r);
      }
    }
  }

  VLOG(2) << "calling synlaunch with  " << num_outputs + arguments.size()
          << " tensors:";
  if (VLOG_IS_ON(2)) {
    for (int i = 0; i < num_outputs + arguments.size(); i++) {
      VLOG(2) << i << "=" << &(persistentTensorInfo[i]) << " "
              << persistentTensorInfo[i].tensorName << " at " << std::hex
              << persistentTensorInfo[i].pTensorAddress << " type "
              << persistentTensorInfo[i].tensorType;
    }
  }
  {
    // TODO: remove once synapse logger trace me implemented
    TF_TRACE_SCOPE_ACTIVITY("HPU::SynLaunch")
    CHECK_SYN1(synLaunch, s, persistentTensorInfo,
               num_outputs + arguments.size(), pWorkspace, recipe,
               SYN_FLAGS_TENSOR_NAME);
    //TODO: uncomment as part of stream<->event design
    hs->BlockUntilDone();
  }

  VLOG(2) << "after synLaunch";

  ScopedShapedBuffer buffers(result_shape(), run_options->allocator(),
                             device_ordinal);

  if (result_shape().IsTuple()) {
    for (int i = 0; i < num_outputs; ++i) {
      buffers.set_buffer(std::move(output_tensors[i]), {i});
    }
  } else {
    buffers.set_buffer(std::move(output_tensors[0]), {});
  }

  ExecutionOutput result(std::move(buffers));
  VLOG(1) << "HPU: Execute " << module().name();
  if (VLOG_IS_ON(2)) {
    for (const auto& a : argument_buffers) {
      VLOG(2) << "-- argument " << a;
    }
  }
  MarkToBeReleasedArguments(absl::MakeSpan(arguments), result);
  return std::move(result);
}
int64_t HpuExecutable::ShapeSizeBytes(const Shape& shape) {
  VLOG(2) << "ShapeSizeBytes called for shape " << shape.ToString();
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}
}  // namespace cpu
}  // namespace xla
