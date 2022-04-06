struct RecipeTensorInfo {
+  std::string name;
+  mlir::Type md;
+};
+
+struct RecipeInfo {
+  RecipeInfo(const RecipeInfo& r) { *this = r; }
+  RecipeInfo() {}
+  unsigned graph_id;
+   recipe_handle = nullptr;
+  llvm::SmallVector<mlir::Type, 4> min_inputs;
+  llvm::SmallVector<mlir::Type, 4> max_inputs;
+  llvm::SmallVector<RecipeTensorInfo, 4> inputs;
+  llvm::SmallVector<RecipeTensorInfo, 4> outputs;
+  bool IsDynamicRecipe() { return min_inputs != max_inputs; }
+  bool IsDynamicTensor(unsigned i) {
+    assert(i < min_inputs.size() && "Index is out of range");
+    return min_inputs[i] != max_inputs[i];
+  }
+};
