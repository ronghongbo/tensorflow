extern several plaidml functions from core.h
Compile(mlir::ModuleOp module) {
  plaidml_init();
  plaidml_edsl_init();
  plaidml_op_init();
  plaidml_exec_init();
  plaidml_Program program(ModuleOp module);
  program.compile();
  wrap up program as PlaidmlExecutable
  return program as Executable

Run(xla::Executable program) {
  cast program into plaidml::exec::Executable
  auto exe = plaidml::exec::Executable(program);    
  std::vector<plaidml::Buffer> inputs;
  for (const plaidml::TensorShape& shape : program.inputs()) {
    inputs.emplace_back(shape);
  }
  std::vector<plaidml::Buffer> outputs;
  for (const plaidml::TensorShape& shape : program.outputs()) {
    outputs.emplace_back(shape);
  }
  exe.run(inputs, outputs);
}