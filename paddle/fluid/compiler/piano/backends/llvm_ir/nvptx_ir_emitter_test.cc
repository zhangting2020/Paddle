/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_ir_emitter.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "llvm/IR/Verifier.h"
#include "paddle/fluid/compiler/piano/backends/note_visitor_base.h"
#include "paddle/fluid/compiler/piano/note/function.h"
#include "paddle/fluid/compiler/piano/note/instruction.h"
#include "paddle/fluid/compiler/piano/note/module.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/compiler/piano/note/opcode.h"
#include "paddle/fluid/compiler/piano/shape.h"
#include "paddle/fluid/compiler/piano/symbolization/note_builder.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace piano {
namespace backends {

class BinaryOpTest {
 public:
  void SetInstructionProto(const std::vector<int64_t>& arg1_shape_vec,
                           const std::vector<int64_t>& arg2_shape_vec,
                           const std::vector<int64_t>& result_shape_vec,
                           note::ElementTypeProto type, note::OpCode op_code) {
    const Shape arg1_shape(type, arg1_shape_vec);
    const Shape arg2_shape(type, arg2_shape_vec);
    const Shape result_shape(type, result_shape_vec);
    op_code_ = op_code;

    SetProto(arg1_shape, "arg1.1", "parameter", 0, &arg1_proto_);
    SetProto(arg2_shape, "arg2.2", "parameter", 1, &arg2_proto_);
    SetProto(result_shape, note::GetOpName(op_code), note::GetOpName(op_code),
             2, &instr_proto_);
  }

  void SetProto(const Shape& shape, const std::string& name,
                const std::string& op_code, uint64_t params_index,
                note::InstructionProto* instr_proto) {
    instr_proto->set_name(name);
    instr_proto->set_opcode(op_code);
    instr_proto->set_parameter_index(params_index);
    *instr_proto->mutable_shape() = shape.ToProto();
  }

  void GenLLVMIR() {
    // build note module
    symbolization::NoteBuilder note_builder("test_note_builder");
    std::vector<symbolization::Operand> ops;
    ops.push_back(note_builder.AppendInstruction(std::move(arg1_proto_),
                                                 note::OpCode::kParameter, {}));
    ops.push_back(note_builder.AppendInstruction(std::move(arg2_proto_),
                                                 note::OpCode::kParameter, {}));
    note_builder.AppendInstruction(std::move(instr_proto_), op_code_, ops);

    auto note_proto = note_builder.Build();
    note::Module note_module(note_proto);

    auto& entry_function = note_module.entry_function();
    auto instr = entry_function.instruction(2);

    llvm::LLVMContext llvm_context;
    llvm::Module llvm_module("", llvm_context);
    KernelExecutableMap kernel_executable_map;

    NvptxIrEmitter nvptx_ir_emitter(&llvm_module, &kernel_executable_map);
    instr->Accept(&nvptx_ir_emitter);

    // Printing may be disabled with the increase of test cases.
    llvm_module.print(llvm::errs(), nullptr);

    std::string errors;
    llvm::raw_string_ostream llvm_errors(errors);
    PADDLE_ENFORCE_NE(llvm::verifyModule(llvm_module, &llvm_errors), true,
                      llvm_errors.str());
  }

 private:
  note::InstructionProto arg1_proto_;
  note::InstructionProto arg2_proto_;
  note::InstructionProto instr_proto_;
  note::OpCode op_code_;
};

TEST(NvptxIrEmitter, FP32OpTest) {
  std::vector<note::OpCode> op_codes = {note::OpCode::kAdd,
                                        note::OpCode::kMaximum};
  BinaryOpTest fp32_test;
  for (auto op_code : op_codes) {
    fp32_test.SetInstructionProto({3, 6}, {3, 6}, {3, 6},
                                  note::ElementTypeProto::F32, op_code);
    fp32_test.GenLLVMIR();
  }
}

TEST(NvptxIrEmitter, Int32OpTest) {
  std::vector<note::OpCode> op_codes = {note::OpCode::kAdd,
                                        note::OpCode::kMaximum};
  BinaryOpTest int32_test;
  for (auto op_code : op_codes) {
    int32_test.SetInstructionProto({3, 6}, {3, 6}, {3, 6},
                                   note::ElementTypeProto::S32, op_code);
    int32_test.GenLLVMIR();
  }
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
