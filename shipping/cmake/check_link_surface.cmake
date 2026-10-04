# Link-surface check for the shipping entrypoint (#465 PR-11, boundary §6
# PR-11): the binary's direct DT_NEEDED must hold no OpenCV, no libpython and
# no libtorch_python. Run as a POST_BUILD step on saccade_track, so a build
# that links any of them fails; also usable by hand:
#
#   cmake -DREADELF=readelf -DBINARY=build/shipping/saccade_track \
#         -P shipping/cmake/check_link_surface.cmake
#
# This covers the direct NEEDED entries only; the full closure, RUNPATH, SM list
# and the rest of G2 are checked on the installed tree by
# scripts/native/check_shipping_tree.py (#465 PR-12).
if(NOT BINARY OR NOT READELF)
    message(FATAL_ERROR "usage: cmake -DREADELF=<readelf> -DBINARY=<file> -P check_link_surface.cmake")
endif()

execute_process(
    COMMAND "${READELF}" -d "${BINARY}"
    OUTPUT_VARIABLE _dynamic
    ERROR_VARIABLE _dynamic_err
    RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "readelf -d ${BINARY} failed (${_rc}): ${_dynamic_err}")
endif()

string(REGEX MATCHALL "\\(NEEDED\\)[^\n]*\\[[^]\n]+\\]" _needed_lines "${_dynamic}")
if(NOT _needed_lines)
    message(FATAL_ERROR "${BINARY}: no DT_NEEDED entries found; not a dynamically linked ELF?")
endif()

set(_forbidden "")
foreach(_line IN LISTS _needed_lines)
    string(REGEX REPLACE ".*\\[([^]]+)\\]$" "\\1" _lib "${_line}")
    if(_lib MATCHES "^libopencv_" OR _lib MATCHES "^libpython" OR _lib MATCHES "^libtorch_python")
        list(APPEND _forbidden "${_lib}")
    endif()
endforeach()

if(_forbidden)
    message(FATAL_ERROR "${BINARY} links ${_forbidden}: the shipping entrypoint must not need OpenCV or Python (#465 PR-11)")
endif()
list(LENGTH _needed_lines _n)
message(STATUS "link surface OK: ${BINARY} (${_n} NEEDED, no OpenCV / libpython / libtorch_python)")
