# Install-time step of the shipping package (#465 Phase C PR-C1): the bundled
# third-party set (owner decision C-D1) and the license files.
#
# shipping/third_party_set.json pins every object by sha256: each is copied
# from its root (this venv's site-packages, or the pinned nvJPEG wheel) to
# lib/vendor/<SONAME> after its hash is checked, so any other bytes are refused
# here, not at the first run. Each wheel's license files go to
# licenses/<wheel>/ (hash-checked too), with THIRD_PARTY.md (the object ->
# license table) and Saccade's own LICENSE / NOTICE under licenses/saccade/.
# The licence texts the package supplies itself (#547: version-matched official
# terms under licenses/terms/, GNU libgomp's licence and source directions
# under licenses/libgomp/) come from shipping/licenses/, each checked against
# the sha256 shipping/license_audit.json records (supplied_texts).
# README.txt (requirements, verify / install / run, named limits; PR-C4) goes
# to the top of the tree.
foreach(_v SACCADE_REPO_ROOT SACCADE_PREFIX_DEST SACCADE_THIRD_PARTY_SET
           SACCADE_ROOT_purelib SACCADE_ROOT_nvjpeg_wheel)
    if(NOT DEFINED ${_v} OR "${${_v}}" STREQUAL "")
        message(FATAL_ERROR "install_third_party.cmake: ${_v} is not set")
    endif()
endforeach()

function(_saccade_install_checked src dst want)
    if(NOT EXISTS "${src}")
        message(FATAL_ERROR "shipping bundle: ${src} does not exist")
    endif()
    file(SHA256 "${src}" _got)
    if(NOT _got STREQUAL want)
        message(FATAL_ERROR "shipping bundle: ${src} sha256 ${_got} != ${want} (shipping/third_party_set.json or shipping/license_audit.json)")
    endif()
    get_filename_component(_dir "${dst}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dir}")
    message(STATUS "Installing: ${dst} (sha256 ${want})")
    file(COPY_FILE "${src}" "${dst}" ONLY_IF_DIFFERENT)
endfunction()

file(READ "${SACCADE_THIRD_PARTY_SET}" _set)
string(JSON _schema GET "${_set}" schema)
if(NOT _schema STREQUAL "saccade.shipping_third_party/v1")
    message(FATAL_ERROR "shipping bundle: ${SACCADE_THIRD_PARTY_SET} has schema ${_schema}")
endif()
string(JSON _n LENGTH "${_set}" entries)
math(EXPR _last "${_n} - 1")
set(_seen_wheels "")
foreach(_i RANGE ${_last})
    string(JSON _soname GET "${_set}" entries ${_i} soname)
    string(JSON _sha GET "${_set}" entries ${_i} sha256)
    string(JSON _root GET "${_set}" entries ${_i} source root)
    string(JSON _path GET "${_set}" entries ${_i} source path)
    string(JSON _wheel GET "${_set}" entries ${_i} wheel)
    if(NOT DEFINED SACCADE_ROOT_${_root})
        message(FATAL_ERROR "shipping bundle: ${_soname} names an unknown root ${_root}")
    endif()
    _saccade_install_checked("${SACCADE_ROOT_${_root}}/${_path}"
        "${SACCADE_PREFIX_DEST}/lib/vendor/${_soname}" "${_sha}")
    list(FIND _seen_wheels "${_wheel}" _k)
    if(_k EQUAL -1)
        list(APPEND _seen_wheels "${_wheel}")
        string(JSON _nl LENGTH "${_set}" entries ${_i} license_files)
        math(EXPR _lastl "${_nl} - 1")
        foreach(_j RANGE ${_lastl})
            string(JSON _lpath GET "${_set}" entries ${_i} license_files ${_j} path)
            string(JSON _lsha GET "${_set}" entries ${_i} license_files ${_j} sha256)
            get_filename_component(_lname "${_lpath}" NAME)
            _saccade_install_checked("${SACCADE_ROOT_${_root}}/${_lpath}"
                "${SACCADE_PREFIX_DEST}/licenses/${_wheel}/${_lname}" "${_lsha}")
        endforeach()
    endif()
endforeach()

file(READ "${SACCADE_REPO_ROOT}/shipping/license_audit.json" _audit)
string(JSON _schema GET "${_audit}" schema)
if(NOT _schema STREQUAL "saccade.shipping_license_audit/v2")
    message(FATAL_ERROR "shipping bundle: shipping/license_audit.json has schema ${_schema}")
endif()
string(JSON _nt LENGTH "${_audit}" supplied_texts)
math(EXPR _lastt "${_nt} - 1")
foreach(_i RANGE ${_lastt})
    string(JSON _tfile GET "${_audit}" supplied_texts ${_i} file)
    string(JSON _trepo GET "${_audit}" supplied_texts ${_i} repo_file)
    string(JSON _tsha GET "${_audit}" supplied_texts ${_i} sha256)
    if(NOT _tfile MATCHES "^licenses/(terms|libgomp)/[^/]+$")
        message(FATAL_ERROR "shipping bundle: supplied text ${_tfile} is outside licenses/terms/ and licenses/libgomp/")
    endif()
    _saccade_install_checked("${SACCADE_REPO_ROOT}/${_trepo}" "${SACCADE_PREFIX_DEST}/${_tfile}" "${_tsha}")
endforeach()

file(MAKE_DIRECTORY "${SACCADE_PREFIX_DEST}/licenses/saccade")
foreach(_f LICENSE NOTICE)
    message(STATUS "Installing: ${SACCADE_PREFIX_DEST}/licenses/saccade/${_f}")
    file(COPY_FILE "${SACCADE_REPO_ROOT}/${_f}" "${SACCADE_PREFIX_DEST}/licenses/saccade/${_f}" ONLY_IF_DIFFERENT)
endforeach()
message(STATUS "Installing: ${SACCADE_PREFIX_DEST}/licenses/THIRD_PARTY.md")
file(COPY_FILE "${SACCADE_REPO_ROOT}/shipping/THIRD_PARTY.md" "${SACCADE_PREFIX_DEST}/licenses/THIRD_PARTY.md" ONLY_IF_DIFFERENT)
message(STATUS "Installing: ${SACCADE_PREFIX_DEST}/README.txt")
file(COPY_FILE "${SACCADE_REPO_ROOT}/shipping/package/README.txt" "${SACCADE_PREFIX_DEST}/README.txt" ONLY_IF_DIFFERENT)
