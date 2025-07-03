add_library(index STATIC
  Device.hpp
  FlatIndex.cpp
  FlatIndex.hpp
  MetricType.hpp
)

target_include_directories(index PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(index PUBLIC backend)