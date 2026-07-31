if(
  NOT DEFINED LOBSTER_REPLAY
  OR NOT DEFINED FIXTURE_DIR
  OR NOT DEFINED OUTPUT_DIR
  OR NOT DEFINED ANALYSIS_POLICY
)
  message(
    FATAL_ERROR
    "LOBSTER_REPLAY, FIXTURE_DIR, OUTPUT_DIR, and ANALYSIS_POLICY are required"
  )
endif()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
set(POLICY_FIXTURE_DIR "${OUTPUT_DIR}/policy-fixture")
file(MAKE_DIRECTORY "${POLICY_FIXTURE_DIR}")
file(READ "${ANALYSIS_POLICY}" POLICY_FIXTURE)
string(
  REGEX REPLACE
  "expected_delivered_sessions=[0-9]+"
  "expected_delivered_sessions=4"
  POLICY_FIXTURE
  "${POLICY_FIXTURE}"
)
string(
  REGEX REPLACE
  "expected_included_sessions=[0-9]+"
  "expected_included_sessions=1"
  POLICY_FIXTURE
  "${POLICY_FIXTURE}"
)
string(
  REGEX REPLACE
  "expected_development_sessions=[0-9]+"
  "expected_development_sessions=1"
  POLICY_FIXTURE
  "${POLICY_FIXTURE}"
)
string(
  REGEX REPLACE
  "expected_selection_sessions=[0-9]+"
  "expected_selection_sessions=0"
  POLICY_FIXTURE
  "${POLICY_FIXTURE}"
)
string(
  REGEX REPLACE
  "expected_test_sessions=[0-9]+"
  "expected_test_sessions=0"
  POLICY_FIXTURE
  "${POLICY_FIXTURE}"
)
set(POLICY_FIXTURE_PATH "${OUTPUT_DIR}/analysis-policy.conf")
file(WRITE "${POLICY_FIXTURE_PATH}" "${POLICY_FIXTURE}")
foreach(DATE IN ITEMS 2019-07-03 2019-01-09 2019-03-08 2019-09-18)
  file(
    COPY_FILE
    "${FIXTURE_DIR}/TSLA_2019-07-03_34200000_57600000_message_2.csv"
    "${POLICY_FIXTURE_DIR}/TSLA_${DATE}_34200000_57600000_message_2.csv"
    ONLY_IF_DIFFERENT
  )
  file(
    COPY_FILE
    "${FIXTURE_DIR}/TSLA_2019-07-03_34200000_57600000_orderbook_2.csv"
    "${POLICY_FIXTURE_DIR}/TSLA_${DATE}_34200000_57600000_orderbook_2.csv"
    ONLY_IF_DIFFERENT
  )
endforeach()

set(BENCHMARK_JSON "${OUTPUT_DIR}/benchmark.json")
set(QUEUE_BINS "${OUTPUT_DIR}/queue-bins.csv")
file(REMOVE "${BENCHMARK_JSON}" "${QUEUE_BINS}")
execute_process(
  COMMAND
    "${LOBSTER_REPLAY}"
    --raw-dir "${POLICY_FIXTURE_DIR}"
    --analysis-policy "${POLICY_FIXTURE_PATH}"
    --decode-runs 1
    --warmup-runs 1
    --replay-runs 1
    --json "${BENCHMARK_JSON}"
    --queue-bins "${QUEUE_BINS}"
    --imbalance-bins 11
    --machine "synthetic CTest fixture"
  RESULT_VARIABLE REPLAY_STATUS
  OUTPUT_VARIABLE REPLAY_STDOUT
  ERROR_VARIABLE REPLAY_STDERR
)
if(NOT REPLAY_STATUS EQUAL 0)
  message(
    FATAL_ERROR
    "lobster_replay failed with ${REPLAY_STATUS}\nstdout:\n${REPLAY_STDOUT}\nstderr:\n${REPLAY_STDERR}"
  )
endif()

file(READ "${BENCHMARK_JSON}" BENCHMARK)
string(JSON EVENTS GET "${BENCHMARK}" dataset events)
string(JSON SESSIONS GET "${BENCHMARK}" dataset included_sessions)
string(JSON DELIVERED GET "${BENCHMARK}" dataset delivered_sessions)
string(JSON EXCLUSIONS GET "${BENCHMARK}" dataset declared_source_exclusions)
string(JSON SEEDED GET "${BENCHMARK}" audit seeded_sessions)
string(JSON EXACT GET "${BENCHMARK}" audit exact_transitions)
string(JSON CENSORED GET "${BENCHMARK}" audit depth_censored_transitions)
string(JSON MISMATCHES GET "${BENCHMARK}" audit mismatches)
string(JSON UNSUPPORTED GET "${BENCHMARK}" audit unsupported)
string(JSON INVALID GET "${BENCHMARK}" audit invalid_snapshots)
string(JSON CHECKSUM GET "${BENCHMARK}" audit checksum)

if(
  NOT EVENTS EQUAL 7
  OR NOT SESSIONS EQUAL 1
  OR NOT DELIVERED EQUAL 4
  OR NOT EXCLUSIONS EQUAL 3
)
  message(
    FATAL_ERROR
    "unexpected synthetic dataset size: ${EVENTS} events, ${SESSIONS} sessions, "
    "${DELIVERED} delivered, ${EXCLUSIONS} exclusions"
  )
endif()
if(NOT SEEDED EQUAL 1 OR NOT EXACT EQUAL 4 OR NOT CENSORED EQUAL 2)
  message(
    FATAL_ERROR
    "unexpected transition counts: ${SEEDED} seeds, ${EXACT} exact, ${CENSORED} censored"
  )
endif()
if(
  NOT MISMATCHES EQUAL 0
  OR NOT UNSUPPORTED EQUAL 0
  OR NOT INVALID EQUAL 0
  OR CHECKSUM EQUAL 0
)
  message(
    FATAL_ERROR
    "unexpected audit result: ${MISMATCHES} mismatches, ${UNSUPPORTED} unsupported, "
    "${INVALID} invalid, checksum ${CHECKSUM}"
  )
endif()

file(READ "${QUEUE_BINS}" QUEUE_OUTPUT)
string(
  FIND
  "${QUEUE_OUTPUT}"
  "date,sample,spread_bucket,bin,bin_left,bin_right,bin_center,observations,up_moves,down_moves"
  HEADER_POSITION
)
string(
  REGEX
  MATCHALL
  "2019-07-03,best_quote_updates,all_spreads"
  BEST_QUOTE_ROWS
  "${QUEUE_OUTPUT}"
)
list(LENGTH BEST_QUOTE_ROWS BEST_QUOTE_ROW_COUNT)
string(
  FIND
  "${QUEUE_OUTPUT}"
  "2019-07-03,best_quote_updates,all_spreads,5,-0.09090909091,0.09090909091,-5.551115123e-17,1,0,1"
  BALANCED_DOWN_POSITION
)
string(
  FIND
  "${QUEUE_OUTPUT}"
  "2019-07-03,best_quote_updates,all_spreads,6,0.09090909091,0.2727272727,0.1818181818,2,1,1"
  MIXED_POSITION
)
string(
  FIND
  "${QUEUE_OUTPUT}"
  "2019-07-03,best_quote_updates,all_spreads,7,0.2727272727,0.4545454545,0.3636363636,2,2,0"
  UP_POSITION
)
string(
  FIND
  "${QUEUE_OUTPUT}"
  "2019-07-03,best_quote_updates,all_spreads,8,0.4545454545,0.6363636364,0.5454545455,1,1,0"
  STRONG_UP_POSITION
)
if(
  NOT HEADER_POSITION EQUAL 0
  OR NOT BEST_QUOTE_ROW_COUNT EQUAL 4
  OR BALANCED_DOWN_POSITION EQUAL -1
  OR MIXED_POSITION EQUAL -1
  OR UP_POSITION EQUAL -1
  OR STRONG_UP_POSITION EQUAL -1
)
  message(
    FATAL_ERROR
    "queue-bin output does not match the synthetic transition path"
  )
endif()
