# Install-time step of the shipping tree (#465 PR-12): the model root that
# saccade_track reads through --model-root. It holds the resolved config, the
# frozen head lineage, the realization attestation and the three files they
# bind (the TorchScript head, the backbone engine, the operator library), each
# at its repository-relative path: the lineage and the attestation name them
# that way and both are frozen, so the tree mirrors those paths instead of
# rewriting them.
#
# Text inputs are staged once, then parsed and installed from those snapshots
# (file(READ)/file(WRITE) would normalize CRLF and change their bytes).
# Every bound binary is copied to a temporary file at the destination, checked
# against its recorded sha256, then renamed into place. A source replacement
# during install cannot publish bytes that were not checked (#585).
# The operator library is the attested build (SACCADE_ATTESTED_OP_LIBRARY); a
# rebuild has another hash and is refused here, not at the first run. Its bytes
# (absolute build RUNPATH, sm_120 SASS only) are installed unchanged: it is the
# enumerated exception to the tree's $ORIGIN-only RUNPATH rule (docs §16).
# Its recorded sha256 and size must also be the runtime's N03 pin
# (P-N03 rev 3d): the pin this build compiled into the runtime
# (SACCADE_N03_BUILT_PIN_*, from configure), which must still be
# shipping/n03_operator_pin.json (a pin changed since the build needs a
# rebuild first). When the tree takes this build's saccade_track
# (SACCADE_N03_BUILT_ENTRYPOINT; empty for a pinned entrypoint, which predates
# the pin until it is re-pinned), that executable must carry the pin's digest,
# so a reconfigure without a rebuild cannot pair it with another operator.
# The install checks digests only; the ELF admission ran once when that digest
# was accepted.
foreach(_v SACCADE_REPO_ROOT SACCADE_MODEL_ROOT_DEST SACCADE_SHIPPING_CONFIG
           SACCADE_SHIPPING_LINEAGE SACCADE_SHIPPING_ATTESTATION SACCADE_ATTESTED_OP_LIBRARY
           SACCADE_N03_OPERATOR_PIN SACCADE_N03_BUILT_PIN_SHA256 SACCADE_N03_BUILT_PIN_BYTES)
    if(NOT DEFINED ${_v} OR "${${_v}}" STREQUAL "")
        message(FATAL_ERROR "install_model_root.cmake: ${_v} is not set")
    endif()
endforeach()

function(_saccade_refuse error)
    foreach(_snapshot IN ITEMS "${_config_snapshot}" "${_lineage_snapshot}" "${_attestation_snapshot}")
        if(_snapshot)
            file(REMOVE "${_snapshot}")
        endif()
    endforeach()
    message(FATAL_ERROR "${error}")
endfunction()

function(_saccade_json_get out text)
    string(JSON _value ERROR_VARIABLE _error GET "${text}" ${ARGN})
    if(_error)
        _saccade_refuse("shipping model root: invalid metadata: ${_error}")
    endif()
    set(${out} "${_value}" PARENT_SCOPE)
endfunction()

function(_saccade_stage_copy src rel out)
    if(NOT EXISTS "${src}")
        _saccade_refuse("shipping model root: ${src} does not exist")
    endif()
    get_filename_component(_dir "${SACCADE_MODEL_ROOT_DEST}/${rel}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dir}")
    string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef _nonce)
    set(_tmp "${_dir}/.saccade-install-${_nonce}")
    file(COPY_FILE "${src}" "${_tmp}" RESULT _copy_error)
    if(NOT _copy_error STREQUAL "0")
        file(REMOVE "${_tmp}")
        _saccade_refuse("shipping model root: cannot copy ${src}: ${_copy_error}")
    endif()
    set(${out} "${_tmp}" PARENT_SCOPE)
endfunction()

function(_saccade_publish_copy tmp rel)
    file(RENAME "${tmp}" "${SACCADE_MODEL_ROOT_DEST}/${rel}" RESULT _rename_error)
    if(NOT _rename_error STREQUAL "0")
        file(REMOVE "${tmp}")
        _saccade_refuse("shipping model root: cannot install ${rel}: ${_rename_error}")
    endif()
    message(STATUS "Installing: ${SACCADE_MODEL_ROOT_DEST}/${rel}")
endfunction()

function(_saccade_copy_checked src rel want)
    _saccade_stage_copy("${src}" "${rel}" _tmp)
    file(SHA256 "${_tmp}" _got)
    if(NOT _got STREQUAL want)
        file(REMOVE "${_tmp}")
        _saccade_refuse("shipping model root: copy of ${src} sha256 ${_got} != ${want} (recorded for ${rel})")
    endif()
    _saccade_publish_copy("${_tmp}" "${rel}")
endfunction()

# Check the build pin before creating snapshots: these shared helpers refuse
# with FATAL_ERROR when the pin or entrypoint is invalid.
include("${CMAKE_CURRENT_LIST_DIR}/n03_pin.cmake")
saccade_read_n03_pin("${SACCADE_N03_OPERATOR_PIN}" _pin_sha _pin_bytes)
if(NOT _pin_sha STREQUAL SACCADE_N03_BUILT_PIN_SHA256 OR NOT _pin_bytes STREQUAL SACCADE_N03_BUILT_PIN_BYTES)
    message(FATAL_ERROR "shipping model root: shipping/n03_operator_pin.json (${_pin_sha}, ${_pin_bytes} bytes) is not the pin this build compiled into the runtime (${SACCADE_N03_BUILT_PIN_SHA256}, ${SACCADE_N03_BUILT_PIN_BYTES} bytes); rebuild before installing")
endif()
if(SACCADE_N03_BUILT_ENTRYPOINT)
    file(STRINGS "${SACCADE_N03_BUILT_ENTRYPOINT}" _pin_in_entrypoint REGEX "${SACCADE_N03_BUILT_PIN_SHA256}" LIMIT_COUNT 1)
    if(NOT _pin_in_entrypoint)
        message(FATAL_ERROR "shipping model root: ${SACCADE_N03_BUILT_ENTRYPOINT} does not carry the N03 pin ${SACCADE_N03_BUILT_PIN_SHA256}; rebuild before installing")
    endif()
endif()

_saccade_stage_copy("${SACCADE_REPO_ROOT}/${SACCADE_SHIPPING_CONFIG}" "${SACCADE_SHIPPING_CONFIG}" _config_snapshot)
_saccade_stage_copy("${SACCADE_REPO_ROOT}/${SACCADE_SHIPPING_LINEAGE}" "${SACCADE_SHIPPING_LINEAGE}" _lineage_snapshot)
_saccade_stage_copy("${SACCADE_REPO_ROOT}/${SACCADE_SHIPPING_ATTESTATION}" "${SACCADE_SHIPPING_ATTESTATION}" _attestation_snapshot)
file(READ "${_lineage_snapshot}" _lineage)
file(READ "${_attestation_snapshot}" _attestation)

_saccade_json_get(_head_path "${_lineage}" torchscript path)
_saccade_json_get(_head_sha "${_lineage}" torchscript sha256)
_saccade_json_get(_engine_path "${_lineage}" companions backbone_engine path)
_saccade_json_get(_engine_sha "${_lineage}" companions backbone_engine sha256)
_saccade_json_get(_op_path "${_lineage}" op_library path)
_saccade_json_get(_att_op_path "${_attestation}" op_library path)
_saccade_json_get(_att_op_sha "${_attestation}" op_library sha256)
_saccade_json_get(_att_lineage_path "${_attestation}" frozen_lineage path)
_saccade_json_get(_att_lineage_sha "${_attestation}" frozen_lineage sha256)
if(NOT _att_op_path STREQUAL _op_path)
    _saccade_refuse("shipping model root: the attestation's op_library.path ${_att_op_path} is not the lineage's ${_op_path}")
endif()
# The attestation text read above: the snapshot whose op_library digest the
# copy below checks.
saccade_require_attested_n03_pin("${SACCADE_N03_BUILT_PIN_SHA256}" "${SACCADE_N03_BUILT_PIN_BYTES}"
    "${_attestation}" "shipping model root" _att_pin_error)
if(_att_pin_error)
    _saccade_refuse("${_att_pin_error}")
endif()
if(NOT _att_lineage_path STREQUAL SACCADE_SHIPPING_LINEAGE)
    _saccade_refuse("shipping model root: the attestation binds ${_att_lineage_path}, not ${SACCADE_SHIPPING_LINEAGE}")
endif()
file(SHA256 "${_lineage_snapshot}" _lineage_sha)
if(NOT _lineage_sha STREQUAL _att_lineage_sha)
    _saccade_refuse("shipping model root: ${SACCADE_SHIPPING_LINEAGE} snapshot sha256 ${_lineage_sha} != ${_att_lineage_sha}")
endif()

_saccade_publish_copy("${_config_snapshot}" "${SACCADE_SHIPPING_CONFIG}")
_saccade_publish_copy("${_attestation_snapshot}" "${SACCADE_SHIPPING_ATTESTATION}")
_saccade_publish_copy("${_lineage_snapshot}" "${SACCADE_SHIPPING_LINEAGE}")
_saccade_copy_checked("${SACCADE_REPO_ROOT}/${_head_path}" "${_head_path}" "${_head_sha}")
_saccade_copy_checked("${SACCADE_REPO_ROOT}/${_engine_path}" "${_engine_path}" "${_engine_sha}")
_saccade_copy_checked("${SACCADE_ATTESTED_OP_LIBRARY}" "${_op_path}" "${_att_op_sha}")
