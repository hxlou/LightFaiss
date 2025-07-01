## 设置shader资源文件列表（使用绝对路径确保build时正确）
#set(SHADERS
#  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul.comp
#  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul_optimized.comp
#  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul_optimized2.comp
#  ${CMAKE_CURRENT_SOURCE_DIR}/shaders/residual.spv
#)
#
#add_library(gpu-kompute STATIC
#  distance.cpp
#  readShader.cpp
#)
#
## Kompute头文件（假设KOMPUTE_PATH已在顶层CMakeLists.txt中set并add_subdirectory）
#target_include_directories(gpu-kompute
#  PRIVATE ${KOMPUTE_PATH}/src/include
#  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
#)
#
## 链接Kompute库
#target_link_libraries(gpu-kompute PUBLIC kompute::kompute)
#
## 复制shader资源到build目录，便于运行时查找
#foreach(SHADER ${SHADERS})
#  get_filename_component(SHADER_NAME ${SHADER} NAME)
#  configure_file(${SHADER} ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME} COPYONLY)
#endforeach()
#
## 找到 glslangValidator
#find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)
#
## 手动列出要编译的shader文件
#set(SHADER_SRC_FILES
#    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul.comp
#    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul_optimized.comp
#    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul_optimized2.comp
#)
#
#set(SHADER_SPV_FILES "")
#foreach(SHADER_SRC ${SHADER_SRC_FILES})
#    get_filename_component(FILE_NAME ${SHADER_SRC} NAME)
#    set(SPV_FILE "${CMAKE_CURRENT_BINARY_DIR}/shaders/${FILE_NAME}.spv")
#    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shaders")
#    add_custom_command(
#        OUTPUT ${SPV_FILE}
#        COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER_SRC} -o ${SPV_FILE}
#        DEPENDS ${SHADER_SRC}
#        COMMENT "Compiling ${FILE_NAME} to SPIR-V"
#        VERBATIM
#    )
#    list(APPEND SHADER_SPV_FILES ${SPV_FILE})
#endforeach()
#
#add_custom_target(compile_shaders ALL DEPENDS ${SHADER_SPV_FILES})