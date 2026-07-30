cmake_minimum_required(VERSION 3.22)

project(tsla_market_impact_cpp VERSION 0.1.0 LANGUAGES CXX)

add_library(
  tsla_lob
  cpp/src/analysis_policy.cpp
  cpp/src/csv_reader.cpp
  cpp/src/replay.cpp
  cpp/src/session_integrity.cpp
)
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

add_executable(lobster_replay cpp/src/main.cpp)
target_link_libraries(lobster_replay PRIVATE tsla_lob)

if(MSVC)
  target_compile_options(lobster_replay PRIVATE /W4 /permissive-)
else()
  target_compile_options(
    lobster_replay
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
      ${CMAKE_CURRENT_SOURCE_DIR}/analysis-policy.conf
  )

  add_test(
    NAME lobster_replay_cli_test
    COMMAND
      ${CMAKE_COMMAND}
      -DLOBSTER_REPLAY=$<TARGET_FILE:lobster_replay>
      -DFIXTURE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/lobster
      -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/cli-test
      -DANALYSIS_POLICY=${CMAKE_CURRENT_SOURCE_DIR}/analysis-policy.conf
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/cli_test.cmake
  )
endif()
