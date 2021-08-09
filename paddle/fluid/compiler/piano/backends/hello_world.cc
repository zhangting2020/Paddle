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

#include "paddle/fluid/compiler/piano/backends/hello_world.h"
#include <llvm/IR/Intrinsics.h>
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

const char *kTargetTriple = "nvptx64-nvidia-cuda";
const char *kDataLayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";

void testzt() {
  llvm::LLVMContext context;
  llvm::IRBuilder<> builder(context);
  std::unique_ptr<llvm::Module> m =
      std::make_unique<llvm::Module>("top", context);
  auto module = m.get();

  module->setTargetTriple(kTargetTriple);
  module->setDataLayout(kDataLayout);

  llvm::FunctionType *funcType =
      llvm::FunctionType::get(llvm::Type::getInt32Ty(context), false);
  llvm::Function *mainFunc = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, "main", module);
  llvm::BasicBlock *entry =
      llvm::BasicBlock::Create(context, "entry", mainFunc);
  builder.SetInsertPoint(entry);
  llvm::Value *helloWorld = builder.CreateGlobalStringPtr("hello world!\n");
  std::vector<llvm::Type *> putsArgs;
  putsArgs.push_back(builder.getInt8Ty()->getPointerTo());
  llvm::ArrayRef<llvm::Type *> argsRef(putsArgs);
  llvm::FunctionType *putsType =
      llvm::FunctionType::get(llvm::Type::getInt32Ty(context), argsRef, false);
  auto putsFunc = module->getOrInsertFunction("puts", putsType);
  builder.CreateCall(putsFunc, helloWorld);
  builder.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 0));

  module->print(llvm::errs(), nullptr);
}
