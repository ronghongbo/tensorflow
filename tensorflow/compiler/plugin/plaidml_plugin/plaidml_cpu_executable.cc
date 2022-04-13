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

// Note: in this file, we follow tensorflow/compiler/xla/service/cpu/cpu_executable.cc,
// and plaidml/core/core.h, instead of the Habana code, because this part is specific to
// device and compiler.

plaidml::TensorShape CreatPlaidmlTensorShape(PrimitiveType xla_type, const xla::Shape &shape) {
  plaidml::DType plaidml_type = XlaTypeToPlaidmlType(xla_type);
  std::vector<int64_t> sizes;
  for (int i = 0; i < shape.rank(); i++) {
    sizes.push_back(shape.dimensions(i));
  }
  plaidml::TensorShape plaidml_tensor_shape(plaidml_type, sizes);
  return std::move(plaidml_tensor_shape);
}

StatusOr<std::vector<plaidml::Buffer>> CreatePlaidmlInputBuffers(const std::vector<ExecutionInput> &arguments) {
  VLOG(2) << "recieved " << arguments.size() << " arguments (execution inputs)";
  std::vector<plaidml::Buffer> plaidml_input_buffers;
  plaidml_input_buffers.reserve(arguments.size());
  for (auto& argument : arguments) {
      const ShapeTree<MaybeOwningDeviceMemory>& buffers = argument.Buffers();
      xla::Shape = buffers.shape();
      plaidml::DType plaidml_type = xlaTypeToPlaidmlType(shape.element_type());
      auto in_it = buffers.begin();
      unit num_nodes = 0;
      for (; in_it != buffers.end(); ++in_it) {
        num_nodes++;
      }
      if (num_nodes != 1) {
        return InternalError("Plaidml expects 1 buffer per argument");
      }
      const Shape& shape = in_it->second
      se::DeviceMemoryBase device_memopry_base = in_it->second.AsDeviceMemoryBase();
      char* data = (char *)device_memopry_base.opaque();
      size_t size = (size_t) device_memopry_base.size();
      plaidml::TensorShape plaidml_shape= CreatPlaidmlTensorShape(xt, argumentshape);  
      plaidml::Buffer plaidml_buffer(data, size, plaidml_shape);
      plaidml_input_buffers.push_back(plaidml_buffer);
  }
  return std::move(plaidml_input_buffers);
}

std::vector<plaidml::Buffer> CreatePlaidmlOutputBuffers(ScopedShapedBuffer* result_, const xla::Shape& result_shape) {
  std::vector<plaidml::Buffer> plaidml_output_buffers;
  const ShapeTree<se::DeviceMemoryBase>& shape_tree = result_->buffers();
  xla::Shape = shape_tree.shape();
  xla::PrimitiveType xla_type = shape.element_type();
  plaidml::DType plaidml_type = xlaTypeToPlaidmlType();

  for (auto& p : shape_tree) {
    const ShapeIndex& index = p.first;
    const ShapeTree<se::DeviceMemoryBase>& sub_tree = shape_tree.SubShapeTree(index);
    const Shape& sub_tree_shape = sub_tree.shape();
    const se::DeviceMemoryBase& device_memopry_base = p.second.AsDeviceMemoryBase();
    char* data = (char *)device_memopry_base.opaque();
    size_t size = (size_t) device_memopry_base.size();
    plaidml::TensorShape plaidml_shape= CreatPlaidmlTensorShape(xla_type, sub_tree_shape);  
      plaidml::Buffer plaidml_buffer(data, size, plaidml_shape);
      plaidml_output_buffers.push_back(plaidml_buffer);
  }

  return std::move(plaidml_output_buffers);
}

PlaidmlCpuExecutable::PlaidmlCpuExecutable(std::unique_ptr<HloModule> hlo_module)
    : Executable(std::move(hlo_module), /*hlo_profile_printer_data=*/nullptr,
                 /*hlo_profile_index_map=*/nullptr) {}

StatusOr<ExecutionOutput> PlaidmlCpuExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments,
    HloExecutionProfile* hlo_execution_profile) {
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
  TF_ASSIGN_OR_RETURN(
      std::vector<MaybeOwningDeviceMemory> buffers,
      CreateBufferTable(memory_allocator, stream->parent()->device_ordinal(),
                        arguments));

  // Note: this code creates and then moves "result" into buffers.
  TF_ASSIGN_OR_RETURN(
      ExecutionOutput result,
      CreateResultShapedBuffer(run_options, absl::MakeSpan(buffers),
                               absl::MakeSpan(arguments)));

  //HloInstruction* root = hlo_module_->entry_computation()->root_instruction();
  //const Shape& root_shape = root->shape();

  // Now wrap the inputs and outputs into plaidml buffers and run the program.
  std::vector<plaidml::Buffer> plaidml_input_buffers = CreatePlaidmlInputBuffers(arguments);
  ScopedShapedBuffer* result_ = result.MutableResult();
  const xla::Shape& result_shape = this->result_shape(); 
  std::vector<plaidml::Buffer> plaidml_output_buffers = CreatePlaidmlOutputBuffers(result_, result_shape);
    
  RecipeInfo& recipe_info = this->getRecipeInfo();
  //std::unique_ptr<plaidml::compiler::Program> plaidml_program = recipe_info.plaidml_program;
  std::unique_ptr<plaidml::exec::Executable> plaidml_exe = recipe_info.plaidml_exe;

  struct AsyncRunTask {
    plaidml::exec::Executable* plaidml_exe;
    std::vector<plaidml::Buffer> plaidml_input_buffers;
    std::vector<plaidml::Buffer> plaidml_output_buffers;
    Status operator()() {
      try {
        double exec_time = plaidml_exe->run(plaidml_input_buffers, plaidml_output_buffers);
        VLOG(1) << "Execution time = " << exec_time << " ms\n";
      } catch (...) {
        return xla::Status(tensorflow::error::Code::ABORTED, "Error in executing Plaidml executable");         
      }
    }
  };
  host_stream->EnqueueTaskWithStatus(
      AsyncRunTask{plaidml_exe, plaidml_input_buffers, plaidml_output_buffers});

  // Since the result's memory is shared by the two variables: "plaidml_output_buffers" and "result",
  // the execution of the task should have changed the memory of "plaidml_output_buffers" and thus 
  // of "result". Now return the "result".
  MarkToBeReleasedArguments(absl::MakeSpan(arguments), result);
  return std::move(result);
}

int64_t PlaidmlCpuExecutable::ShapeSizeBytes(const Shape& shape) {
  VLOG(2) << "ShapeSizeBytes called for shape " << shape.ToString();
  if (shape.IsOpaque()) {
    return sizeof(void*);
  }
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}
}  // namespace cpu
}  // namespace xla
