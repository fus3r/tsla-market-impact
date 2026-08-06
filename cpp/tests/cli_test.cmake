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
set(ORDER_FLOW_BINS "${OUTPUT_DIR}/order-flow-bins.csv")
set(OFI_HORIZON_BINS "${OUTPUT_DIR}/ofi-horizon-bins.csv")
set(MARKOUT_BINS "${OUTPUT_DIR}/markout-bins.csv")
set(LANDMARK_BINS "${OUTPUT_DIR}/landmark-bins.csv")
set(ROUND_TRIP_BINS "${OUTPUT_DIR}/round-trip-bins.csv")
file(
  REMOVE
  "${BENCHMARK_JSON}"
  "${QUEUE_BINS}"
  "${ORDER_FLOW_BINS}"
  "${OFI_HORIZON_BINS}"
  "${MARKOUT_BINS}"
  "${LANDMARK_BINS}"
  "${ROUND_TRIP_BINS}"
)
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
    --order-flow-bins "${ORDER_FLOW_BINS}"
    --order-flow-grid 11
    --ofi-horizon-bins "${OFI_HORIZON_BINS}"
    --ofi-quote-windows 1,2
    --ofi-clock-windows-us 1,2
    --markout-bins "${MARKOUT_BINS}"
    --markout-latencies-us 0,1
    --landmark-bins "${LANDMARK_BINS}"
    --landmark-age-us 1
    --landmark-latencies-us 0,1
    --round-trip-bins "${ROUND_TRIP_BINS}"
    --round-trip-sizes 1,70
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
string(JSON MID_CHANGES GET "${BENCHMARK}" audit mid_price_changes)
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

file(READ "${MARKOUT_BINS}" MARKOUT_OUTPUT)
string(
  FIND
  "${MARKOUT_OUTPUT}"
  "date,sample,spread_bucket,latency_us,queue_bin,ofi_bin,queue_center,ofi_center,signals,executable,stale,up_moves,down_moves,midpoint_move_sum_bps,half_spread_sum_bps"
  MARKOUT_HEADER_POSITION
)
string(
  FIND
  "${MARKOUT_OUTPUT}"
  "2019-07-03,best_quote_updates,all_spreads,0,7,6,0.363636363636,0.181818181818,1,1,0,1,0,50,100"
  ZERO_LATENCY_MARKOUT_POSITION
)
string(
  FIND
  "${MARKOUT_OUTPUT}"
  "2019-07-03,best_quote_updates,all_spreads,1,7,6,0.363636363636,0.181818181818,1,0,1,1,0,0,0"
  STALE_MARKOUT_POSITION
)
if(
  NOT MARKOUT_HEADER_POSITION EQUAL 0
  OR ZERO_LATENCY_MARKOUT_POSITION EQUAL -1
  OR STALE_MARKOUT_POSITION EQUAL -1
)
  message(
    FATAL_ERROR
    "markout output does not match the synthetic execution path"
  )
endif()

file(READ "${LANDMARK_BINS}" LANDMARK_OUTPUT)
string(
  FIND
  "${LANDMARK_OUTPUT}"
  "date,sample,spread_bucket,landmark_age_us,latency_us,queue_bin,ofi_bin,queue_center,ofi_center,signals,executable,stale,up_moves,down_moves,midpoint_move_sum_bps,half_spread_sum_bps"
  LANDMARK_HEADER_POSITION
)
string(
  FIND
  "${LANDMARK_OUTPUT}"
  "2019-07-03,price_spell_landmarks,all_spreads,1,0,5,5,0,0,1,1,0,0,1,-49.7512437811,149.253731343"
  LANDMARK_DOWN_POSITION
)
string(
  FIND
  "${LANDMARK_OUTPUT}"
  "2019-07-03,price_spell_landmarks,all_spreads,1,0,6,5,0.181818181818,0,2,2,0,1,1,0,200"
  LANDMARK_MIXED_POSITION
)
string(
  FIND
  "${LANDMARK_OUTPUT}"
  "2019-07-03,price_spell_landmarks,all_spreads,1,0,7,5,0.363636363636,0,1,1,0,1,0,50.2512562814,150.753768844"
  LANDMARK_UP_POSITION
)
string(
  FIND
  "${LANDMARK_OUTPUT}"
  "2019-07-03,price_spell_landmarks,all_spreads,1,1,6,5,0.181818181818,0,2,0,2,1,1,0,0"
  LANDMARK_STALE_POSITION
)
if(
  NOT LANDMARK_HEADER_POSITION EQUAL 0
  OR LANDMARK_DOWN_POSITION EQUAL -1
  OR LANDMARK_MIXED_POSITION EQUAL -1
  OR LANDMARK_UP_POSITION EQUAL -1
  OR LANDMARK_STALE_POSITION EQUAL -1
)
  message(
    FATAL_ERROR
    "landmark output does not match the one-signal-per-spell path"
  )
endif()

file(READ "${ROUND_TRIP_BINS}" ROUND_TRIP_OUTPUT)
string(
  FIND
  "${ROUND_TRIP_OUTPUT}"
  "date,sample,spread_bucket,landmark_age_us,entry_latency_us,shares,queue_bin,ofi_bin,queue_center,ofi_center,signals,arrived,stale,up_moves,down_moves,long_fills,long_capacity_censored,long_midpoint_move_sum_bps,long_entry_cost_sum_bps,long_exit_cost_sum_bps,long_quoted_pnl_sum_bps,short_fills,short_capacity_censored,short_midpoint_move_sum_bps,short_entry_cost_sum_bps,short_exit_cost_sum_bps,short_quoted_pnl_sum_bps"
  ROUND_TRIP_HEADER_POSITION
)
string(
  FIND
  "${ROUND_TRIP_OUTPUT}"
  "2019-07-03,price_spell_round_trips,all_spreads,1,0,1,5,5,0,0,1,1,0,0,1,1,0,-49.7512437811,149.253731343,99.5024875622,-298.507462687,1,0,49.7512437811,149.253731343,99.5024875622,-199.004975124"
  ROUND_TRIP_ONE_SHARE_POSITION
)
string(
  FIND
  "${ROUND_TRIP_OUTPUT}"
  "2019-07-03,price_spell_round_trips,all_spreads,1,0,70,6,5,0.181818181818,0,2,2,0,1,1,0,2,0,0,0,0,1,1,-50,157.142857143,192.857142857,-400"
  ROUND_TRIP_CAPACITY_POSITION
)
string(
  FIND
  "${ROUND_TRIP_OUTPUT}"
  "2019-07-03,price_spell_round_trips,all_spreads,1,1,1,6,5,0.181818181818,0,2,0,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0"
  ROUND_TRIP_STALE_POSITION
)
if(
  NOT ROUND_TRIP_HEADER_POSITION EQUAL 0
  OR ROUND_TRIP_ONE_SHARE_POSITION EQUAL -1
  OR ROUND_TRIP_CAPACITY_POSITION EQUAL -1
  OR ROUND_TRIP_STALE_POSITION EQUAL -1
)
  message(
    FATAL_ERROR
    "round-trip output does not match the depth-constrained path"
  )
endif()

file(READ "${ORDER_FLOW_BINS}" ORDER_FLOW_OUTPUT)
string(
  FIND
  "${ORDER_FLOW_OUTPUT}"
  "date,sample,spread_bucket,queue_bin,ofi_bin,queue_center,ofi_center,observations,up_moves,down_moves"
  ORDER_FLOW_HEADER_POSITION
)
string(
  REGEX
  MATCHALL
  "2019-07-03,best_quote_updates,all_spreads"
  ORDER_FLOW_ALL_SPREAD_ROWS
  "${ORDER_FLOW_OUTPUT}"
)
list(LENGTH ORDER_FLOW_ALL_SPREAD_ROWS ORDER_FLOW_ALL_SPREAD_COUNT)
string(
  FIND
  "${ORDER_FLOW_OUTPUT}"
  "2019-07-03,best_quote_updates,all_spreads,6,5,0.1818181818,0,2,1,1"
  RESET_CELL_POSITION
)
string(
  FIND
  "${ORDER_FLOW_OUTPUT}"
  "2019-07-03,best_quote_updates,all_spreads,7,6,0.3636363636,0.1818181818,1,1,0"
  POSITIVE_OFI_POSITION
)
if(
  NOT ORDER_FLOW_HEADER_POSITION EQUAL 0
  OR NOT ORDER_FLOW_ALL_SPREAD_COUNT EQUAL 5
  OR RESET_CELL_POSITION EQUAL -1
  OR POSITIVE_OFI_POSITION EQUAL -1
)
  message(
    FATAL_ERROR
    "order-flow output does not match the synthetic reset path"
  )
endif()

file(READ "${OFI_HORIZON_BINS}" OFI_HORIZON_OUTPUT)
string(
  FIND
  "${OFI_HORIZON_OUTPUT}"
  "date,sample,spread_bucket,horizon_kind,horizon_value,queue_bin,ofi_bin,queue_center,ofi_center,observations,up_moves,down_moves"
  OFI_HORIZON_HEADER_POSITION
)
string(
  FIND
  "${OFI_HORIZON_OUTPUT}"
  "2019-07-03,best_quote_updates,all_spreads,price_spell,0,6,5,0.1818181818,0,2,1,1"
  OFI_RESET_POSITION
)
string(
  FIND
  "${OFI_HORIZON_OUTPUT}"
  "2019-07-03,best_quote_updates,all_spreads,quote_updates,1,6,4,0.1818181818,-0.1818181818,1,0,1"
  OFI_ONE_UPDATE_POSITION
)
string(
  FIND
  "${OFI_HORIZON_OUTPUT}"
  "2019-07-03,best_quote_updates,all_spreads,clock_us,2,7,3,0.3636363636,-0.3636363636,1,1,0"
  OFI_STRICT_CLOCK_POSITION
)
if(
  NOT OFI_HORIZON_HEADER_POSITION EQUAL 0
  OR OFI_RESET_POSITION EQUAL -1
  OR OFI_ONE_UPDATE_POSITION EQUAL -1
  OR OFI_STRICT_CLOCK_POSITION EQUAL -1
)
  message(
    FATAL_ERROR
    "OFI horizon output does not match the causal window path"
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
  OR NOT MID_CHANGES EQUAL 4
  OR CHECKSUM EQUAL 0
)
  message(
    FATAL_ERROR
    "unexpected audit result: ${MISMATCHES} mismatches, ${UNSUPPORTED} unsupported, "
    "${INVALID} invalid, ${MID_CHANGES} mid-price changes, checksum ${CHECKSUM}"
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
