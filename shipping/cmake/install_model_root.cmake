# Install-time step of the shipping tree (#465 PR-12): the model root that
# saccade_track reads through --model-root. It holds the resolved config, the
# frozen head lineage, the realization attestation and the three files they
# bind (the TorchScript head, the backbone engine, the operator library), each
# at its repository-relative path: the lineage and the attestation name them
# that way and both are frozen, so the tree mirrors those paths instead of
# rewriting them.
#
# Every bound file is checked against its recorded sha256 before it is copied.
# The operator library is the attested build (SACCADE_ATTESTED_OP_LIBRARY); a
# rebuild has another hash and is refused here, not at the first run. Its bytes
# (absolute build RUNPATH, sm_120 SASS only) are installed unchanged: it is the
# enumerated exception to the tree's $ORIGIN-only RUNPATH rule (docs §16).
# Its recorded sha256 and size must also be the runtime's N03 pin
# (shipping/n03_operator_pin.json, P-N03 rev 3d): the install checks digests
# only; the ELF admission ran once when that digest was accepted.
foreach(_v SACCADE_REPO_ROOT SACCADE_MODEL_ROOT_DEST SACCADE_SHIPPING_CONFIG
           SACCADE_SHIPPING_LINEAGE SACCADE_SHIPPING_ATTESTATION SACCADE_ATTESTED_OP_LIBRARY
           SACCADE_N03_OPERATOR_PIN)
    if(NOT DEFINED ${_v} OR "${${_v}}" STREQUAL "")
        message(FATAL_ERROR "install_model_root.cmake: ${_v} is not set")
    endif()
endforeach()

function(_saccade_copy_checked src rel want)
    if(NOT EXISTS "${src}")
        message(FATAL_ERROR "shipping model root: ${src} does not exist")
    endif()
    file(SHA256 "${src}" _got)
    if(NOT _got STREQUAL want)
        message(FATAL_ERROR "shipping model root: ${src} sha256 ${_got} != ${want} (recorded for ${rel})")
    endif()
    get_filename_component(_dir "${SACCADE_MODEL_ROOT_DEST}/${rel}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dir}")
    message(STATUS "Installing: ${SACCADE_MODEL_ROOT_DEST}/${rel} (sha256 ${want})")
    file(COPY_FILE "${src}" "${SACCADE_MODEL_ROOT_DEST}/${rel}" ONLY_IF_DIFFERENT)
endfunction()

function(_saccade_copy_plain rel)
    get_filename_component(_dir "${SACCADE_MODEL_ROOT_DEST}/${rel}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dir}")
    message(STATUS "Installing: ${SACCADE_MODEL_ROOT_DEST}/${rel}")
    file(COPY_FILE "${SACCADE_REPO_ROOT}/${rel}" "${SACCADE_MODEL_ROOT_DEST}/${rel}" ONLY_IF_DIFFERENT)
endfunction()

file(READ "${SACCADE_REPO_ROOT}/${SACCADE_SHIPPING_LINEAGE}" _lineage)
file(READ "${SACCADE_REPO_ROOT}/${SACCADE_SHIPPING_ATTESTATION}" _attestation)

string(JSON _head_path GET "${_lineage}" torchscript path)
string(JSON _head_sha GET "${_lineage}" torchscript sha256)
string(JSON _engine_path GET "${_lineage}" companions backbone_engine path)
string(JSON _engine_sha GET "${_lineage}" companions backbone_engine sha256)
string(JSON _op_path GET "${_lineage}" op_library path)
string(JSON _att_op_path GET "${_attestation}" op_library path)
string(JSON _att_op_sha GET "${_attestation}" op_library sha256)
string(JSON _att_lineage_path GET "${_attestation}" frozen_lineage path)
string(JSON _att_lineage_sha GET "${_attestation}" frozen_lineage sha256)
if(NOT _att_op_path STREQUAL _op_path)
    message(FATAL_ERROR "shipping model root: the attestation's op_library.path ${_att_op_path} is not the lineage's ${_op_path}")
endif()
file(READ "${SACCADE_N03_OPERATOR_PIN}" _n03_pin)
string(JSON _pin_sha GET "${_n03_pin}" sha256)
string(JSON _pin_bytes GET "${_n03_pin}" bytes)
string(JSON _att_op_bytes GET "${_attestation}" op_library bytes)
if(NOT _att_op_sha STREQUAL _pin_sha OR NOT _att_op_bytes STREQUAL _pin_bytes)
    message(FATAL_ERROR "shipping model root: the attestation's op_library (${_att_op_sha}, ${_att_op_bytes} bytes) is not the runtime's N03 pin (${_pin_sha}, ${_pin_bytes} bytes; shipping/n03_operator_pin.json)")
endif()
if(NOT _att_lineage_path STREQUAL SACCADE_SHIPPING_LINEAGE)
    message(FATAL_ERROR "shipping model root: the attestation binds ${_att_lineage_path}, not ${SACCADE_SHIPPING_LINEAGE}")
endif()

_saccade_copy_plain("${SACCADE_SHIPPING_CONFIG}")
_saccade_copy_plain("${SACCADE_SHIPPING_ATTESTATION}")
_saccade_copy_checked("${SACCADE_REPO_ROOT}/${SACCADE_SHIPPING_LINEAGE}" "${SACCADE_SHIPPING_LINEAGE}" "${_att_lineage_sha}")
_saccade_copy_checked("${SACCADE_REPO_ROOT}/${_head_path}" "${_head_path}" "${_head_sha}")
_saccade_copy_checked("${SACCADE_REPO_ROOT}/${_engine_path}" "${_engine_path}" "${_engine_sha}")
_saccade_copy_checked("${SACCADE_ATTESTED_OP_LIBRARY}" "${_op_path}" "${_att_op_sha}")
