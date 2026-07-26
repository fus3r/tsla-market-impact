cmake_minimum_required(VERSION 3.22)

project(tsla_market_impact_cpp VERSION 0.1.0 LANGUAGES CXX)

add_library(tsla_lob cpp/src/csv_reader.cpp)
target_include_directories(tsla_lob PUBLIC cpp/include)
target_compile_features(tsla_lob PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(tsla_lob PRIVATE /W4 /permissive-)
else()
  target_compile_options(
    tsla_lob
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
  )
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(tsla_lob_csv_reader_test cpp/tests/csv_reader_test.cpp)
  target_link_libraries(tsla_lob_csv_reader_test PRIVATE tsla_lob)

  if(MSVC)
    target_compile_options(tsla_lob_csv_reader_test PRIVATE /W4 /permissive-)
  else()
    target_compile_options(
      tsla_lob_csv_reader_test
      PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
  endif()

  add_test(
    NAME tsla_lob_csv_reader_test
    COMMAND
      tsla_lob_csv_reader_test
      ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/fixtures
  )
endif()
