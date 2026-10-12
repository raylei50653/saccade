# N03 pin check for the shipping entrypoint (#549 S2-2, P-N03 rev 3d): the
# binary must carry the production pin's sha256 (shipping/n03_operator_pin.json,
# compiled into saccade_shipping_n03_pin), not a fixture pin. Run as a
# POST_BUILD step on saccade_track; also usable by hand:
#
#   cmake -DBINARY=build/shipping/saccade_track -DPIN_SHA256=<sha256> \
#         -P shipping/cmake/check_n03_pin.cmake
if(NOT BINARY OR NOT PIN_SHA256)
    message(FATAL_ERROR "usage: cmake -DBINARY=<file> -DPIN_SHA256=<sha256> -P check_n03_pin.cmake")
endif()
if(NOT PIN_SHA256 MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "check_n03_pin: '${PIN_SHA256}' is not a sha256")
endif()
file(STRINGS "${BINARY}" _hits REGEX "${PIN_SHA256}" LIMIT_COUNT 1)
if(NOT _hits)
    message(FATAL_ERROR "${BINARY} does not carry the N03 pin ${PIN_SHA256} (shipping/n03_operator_pin.json)")
endif()
