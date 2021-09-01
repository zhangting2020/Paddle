// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/compiler/piano/backends/llvm_ir/primitive_ir_emitter.h"

namespace paddle {
namespace piano {
namespace backends {

void PrimitiveIrEmitter::VisitCast(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

void PrimitiveIrEmitter::VisitExp(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

void PrimitiveIrEmitter::VisitLog(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

void PrimitiveIrEmitter::VisitNegative(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

void PrimitiveIrEmitter::VisitNot(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

void PrimitiveIrEmitter::VisitRsqrt(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

void PrimitiveIrEmitter::VisitSqrt(const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

void PrimitiveIrEmitter::VisitAdd(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitAnd(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitCompare(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitDivide(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitMaximum(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitMinimum(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitMultiply(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitOr(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitSubtract(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

void PrimitiveIrEmitter::VisitXor(const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

llvm::Value* PrimitiveIrEmitter::Add(llvm::Value* lhs, llvm::Value* rhs,
                                     llvm::IRBuilder<>* ir_builder) {
  if (lhs->getType()->isIntegerTy()) {
    return ir_builder->CreateAdd(lhs, rhs);
  } else {
    return ir_builder->CreateFAdd(lhs, rhs);
  }
}

llvm::Value* PrimitiveIrEmitter::Multiply(llvm::Value* lhs, llvm::Value* rhs,
                                          llvm::IRBuilder<>* ir_builder) {
  if (lhs->getType()->isIntegerTy()) {
    return ir_builder->CreateMul(lhs, rhs);
  } else {
    return ir_builder->CreateFMul(lhs, rhs);
  }
}

llvm::Value* PrimitiveIrEmitter::Maximum(llvm::Value* lhs, llvm::Value* rhs,
                                         bool is_signed,
                                         llvm::IRBuilder<>* ir_builder) {
  if (lhs->getType()->isIntegerTy()) {
    llvm::CmpInst::Predicate predicate =
        is_signed ? llvm::ICmpInst::ICMP_SGE : llvm::ICmpInst::ICMP_UGE;
    return ir_builder->CreateSelect(ir_builder->CreateICmp(predicate, lhs, rhs),
                                    lhs, rhs);
  } else {
    // Implements IEEE 754-2018 maximum semantics. If one of the
    // elements being compared is a NaN, then that element is returned.
    // So we use unordered comparisons because it always return true
    // when one of the operands is NaN.
    auto cmp = ir_builder->CreateFCmpUGE(lhs, rhs);
    return ir_builder->CreateSelect(cmp, lhs, rhs);
  }
}

llvm::Value* PrimitiveIrEmitter::Load(llvm::Value* input, llvm::Value* index,
                                      llvm::IRBuilder<>* ir_builder) {
  auto val = ir_builder->CreateGEP(input, index);
  return ir_builder->CreateLoad(val);
}

llvm::Value* PrimitiveIrEmitter::Store(llvm::Value* src, llvm::Value* dst,
                                       llvm::Value* dst_index,
                                       llvm::IRBuilder<>* ir_builder) {
  auto val = ir_builder->CreateGEP(dst, dst_index);
  return ir_builder->CreateStore(src, val);
}

void PrimitiveIrEmitter::If(
    llvm::Value* cond,
    std::function<void(llvm::IRBuilder<>* ir_builder)> gen_body,
    llvm::IRBuilder<>* ir_builder) {
  auto then_block = llvm::BasicBlock::Create(*ctx_, "if_then", func_);
  auto end_block = llvm::BasicBlock::Create(*ctx_, "if_end", func_);
  ir_builder->CreateCondBr(cond, then_block, end_block);
  ir_builder->SetInsertPoint(then_block);
  gen_body(ir_builder);
  ir_builder->CreateBr(end_block);
  ir_builder->SetInsertPoint(end_block);
}

void PrimitiveIrEmitter::For(
    llvm::Value* begin, llvm::Value* end, llvm::Value* stride,
    std::function<void(llvm::IRBuilder<>* ir_builder)> gen_body,
    llvm::IRBuilder<>* ir_builder) {
  auto pre_block = ir_builder->GetInsertBlock();
  auto for_begin = llvm::BasicBlock::Create(*ctx_, "for_begin", func_);
  auto for_body = llvm::BasicBlock::Create(*ctx_, "for_body", func_);
  auto for_end = llvm::BasicBlock::Create(*ctx_, "for_end", func_);
  ir_builder->SetInsertPoint(pre_block);
  ir_builder->CreateBr(for_begin);

  // for_begin: It decides to enter for_body or
  // for_end according to the loop value.
  ir_builder->SetInsertPoint(for_begin);
  llvm::PHINode* loop_value = ir_builder->CreatePHI(begin->getType(), 2);
  loop_value->addIncoming(begin, pre_block);
  ir_builder->CreateCondBr(ir_builder->CreateICmpULT(loop_value, end), for_body,
                           for_end);
  // for_body: Implement load, computation and store.
  // Then update loop value and go to for_begin.
  ir_builder->SetInsertPoint(for_body);
  gen_body(ir_builder);
  llvm::Value* next_value = ir_builder->CreateAdd(loop_value, stride);
  loop_value->addIncoming(next_value, ir_builder->GetInsertBlock());
  ir_builder->CreateBr(for_begin);
  // for_end: Other process will be insert here.
  ir_builder->SetInsertPoint(for_end);
}

std::vector<PrimitiveIrGenerator>
PrimitiveIrEmitter::GetPrimitiveIrGenerators() {
  return primitive_ir_generators_;
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
