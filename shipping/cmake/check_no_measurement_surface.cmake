# Measurement-surface check for the shipping entrypoint (#465 Phase C PR-C2):
# the binary must contain none of the byte strings in
# shipping/measurement_surface.json (the mutation names, the measurement
# setters, the developer options). Run as a POST_BUILD step on saccade_track,
# so a shipping build that compiles in a measurement hook or a developer
# option fails; also usable by hand:
#
#   cmake -DBINARY=build/shipping/saccade_track -DSURFACE=shipping/measurement_surface.json \
#         -P shipping/cmake/check_no_measurement_surface.cmake
#
# The installed tree is checked again by scripts/native/check_shipping_bundle.py
# static (entrypoint_no_measurement_surface).
if(NOT BINARY OR NOT SURFACE)
    message(FATAL_ERROR "usage: cmake -DBINARY=<file> -DSURFACE=<measurement_surface.json> -P check_no_measurement_surface.cmake")
endif()

file(READ "${SURFACE}" _surface)
string(JSON _n LENGTH "${_surface}" forbidden)
if(_n EQUAL 0)
    message(FATAL_ERROR "${SURFACE}: empty forbidden list")
endif()
math(EXPR _last "${_n} - 1")
set(_found "")
foreach(_i RANGE ${_last})
    string(JSON _token GET "${_surface}" forbidden ${_i})
    # Literal match: the tokens hold no regex metacharacter but '-'.
    file(STRINGS "${BINARY}" _hits REGEX "${_token}" LIMIT_COUNT 1)
    if(_hits)
        list(APPEND _found "${_token}")
    endif()
endforeach()

if(_found)
    message(FATAL_ERROR "${BINARY} contains ${_found}: the shipping entrypoint must have no measurement hook or developer option (#465 PR-C2; link the shipping runtime, not a _measurement library)")
endif()
message(STATUS "measurement surface OK: ${BINARY} (none of ${_n} strings)")
