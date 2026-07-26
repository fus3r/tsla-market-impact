cmake_minimum_required(VERSION 3.22)

project(tsla_market_impact_cpp VERSION 0.1.0 LANGUAGES CXX)

add_library(tsla_lob INTERFACE)
target_include_directories(tsla_lob INTERFACE cpp/include)
target_compile_features(tsla_lob INTERFACE cxx_std_20)

include(CTest)
if(BUILD_TESTING)
  add_executable(tsla_lob_message_test cpp/tests/message_test.cpp)
  target_link_libraries(tsla_lob_message_test PRIVATE tsla_lob)

  if(MSVC)
    target_compile_options(tsla_lob_message_test PRIVATE /W4 /permissive-)
  else()
    target_compile_options(
      tsla_lob_message_test
      PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
  endif()

  add_test(NAME tsla_lob_message_test COMMAND tsla_lob_message_test)
endif()
