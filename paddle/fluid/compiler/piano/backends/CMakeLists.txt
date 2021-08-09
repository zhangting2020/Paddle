add_subdirectory(llvm_ir)
cc_library(testzt SRCS hello_world.cc)
cc_test(testztt SRCS test.cc DEPS testzt)
target_link_libraries(testztt ${LLVM_LIBS})

