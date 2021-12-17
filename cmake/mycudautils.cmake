# set cura_architecture_list parameter to the list of cuda architectures to build. ARCHITECTURE can be auto to detect
# architectures automatically or a space separated list of architecture numbers
function (select_cuda_architectures ARCHITECTURE cuda_architecture_list)
  string(TOLOWER "${ARCHITECTURE}" ARCHITECTURE_LOWER)
  if ("${ARCHITECTURE_LOWER}" STREQUAL "auto")
    include(FindCUDA)
    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS Auto)
    string(
      REGEX
      REPLACE "-gencode|arch=compute_[0-9][0-9],code=(sm|compute)_"
              ";"
              arch_list_tmp
              ${CUDA_ARCH_FLAGS})
    set(${cuda_architecture_list}
        ${arch_list_tmp}
        PARENT_SCOPE)
  else ()
    string(
      REPLACE " "
              ";"
              arch_list_tmp
              "${ARCHITECTURE}")
    set(${cuda_architecture_list}
        ${arch_list_tmp}
        PARENT_SCOPE)
  endif ()
endfunction ()