# The runtime's N03 pin (#549 S2-2, P-N03 rev 3d): shipping/n03_operator_pin.json.
# Included by shipping/CMakeLists.txt (configure: the pin compiled into the
# loader) and cmake/install_model_root.cmake (install), so both apply one rule:
#
#   saccade_read_n03_pin(<pin file> <out sha256> <out bytes>)
#       a 64-hex sha256 and a positive byte count, or FATAL_ERROR;
#   saccade_require_attested_n03_pin(<sha256> <bytes> <attestation JSON text> <context> [<out error>])
#       the realization attestation's op_library must be that pin (the caller
#       passes the text it read, so it checks the snapshot it uses).
#       With <out error>, return the refusal text (empty on success) so the
#       install caller can clean up its temporary snapshots before refusing.
#
# Script mode (tests): cmake -DPIN=<pin file> -DATTESTATION=<attestation> -P n03_pin.cmake
function(saccade_read_n03_pin pin_file out_sha out_bytes)
    file(READ "${pin_file}" _pin)
    string(JSON _sha ERROR_VARIABLE _sha_error GET "${_pin}" sha256)
    string(JSON _bytes ERROR_VARIABLE _bytes_error GET "${_pin}" bytes)
    string(LENGTH "${_sha}" _sha_length)
    if(_sha_error OR _bytes_error OR NOT _sha MATCHES "^[0-9a-f]+$" OR NOT _sha_length EQUAL 64 OR
       NOT _bytes MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "${pin_file}: sha256 '${_sha}' / bytes '${_bytes}' is not an N03 pin")
    endif()
    set(${out_sha} "${_sha}" PARENT_SCOPE)
    set(${out_bytes} "${_bytes}" PARENT_SCOPE)
endfunction()

function(saccade_require_attested_n03_pin sha bytes attestation context)
    set(_error "")
    string(JSON _att_sha ERROR_VARIABLE _e1 GET "${attestation}" op_library sha256)
    string(JSON _att_bytes ERROR_VARIABLE _e2 GET "${attestation}" op_library bytes)
    if(_e1 OR _e2 OR NOT _att_sha STREQUAL sha OR NOT _att_bytes STREQUAL bytes)
        set(_error "${context}: the attestation's op_library (${_att_sha}, ${_att_bytes} bytes) is not the runtime's N03 pin (${sha}, ${bytes} bytes; shipping/n03_operator_pin.json)")
    endif()
    if(ARGC GREATER 4)
        set(${ARGV4} "${_error}" PARENT_SCOPE)
    elseif(_error)
        message(FATAL_ERROR "${_error}")
    endif()
endfunction()

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
    if(NOT PIN OR NOT ATTESTATION)
        message(FATAL_ERROR "usage: cmake -DPIN=<pin file> -DATTESTATION=<attestation> -P n03_pin.cmake")
    endif()
    saccade_read_n03_pin("${PIN}" _sha _bytes)
    file(READ "${ATTESTATION}" _attestation)
    saccade_require_attested_n03_pin("${_sha}" "${_bytes}" "${_attestation}" "n03 pin")
endif()
