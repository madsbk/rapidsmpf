# ============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

# This function finds libcoro and sets any additional necessary environment variables.
function(find_and_configure_libcoro)
  if(TARGET libcoro)
    return()
  endif()

  rapids_cpm_find(
    libcoro 0.15.0
    GLOBAL_TARGETS libcoro
    CPM_ARGS
    GIT_REPOSITORY https://github.com/jbaldwin/libcoro
    GIT_TAG main
    GIT_SHALLOW TRUE
    OPTIONS "LIBCORO_FEATURE_NETWORKING OFF"
            "LIBCORO_EXTERNAL_DEPENDENCIES OFF"
            "LIBCORO_BUILD_EXAMPLES OFF"
            "LIBCORO_FEATURE_TLS OFF"
            "LIBCORO_BUILD_TESTS OFF"
            "BUILD_SHARED_LIBS OFF"
            "CMAKE_POSITION_INDEPENDENT_CODE ON"
  )
  if(TARGET libcoro)
    set_property(TARGET libcoro PROPERTY SYSTEM TRUE)
  endif()
endfunction()

set(_RAPIDSMPF_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})

find_and_configure_libcoro()

# libcoro overwrites BUILD_SHARED_LIBS to OFF:
# <https://github.com/jbaldwin/libcoro/blob/main/CMakeLists.txt#L81>
#
# This is a workaround that resets `BUILD_SHARED_LIBS` to ON.
if(_RAPIDSMPF_BUILD_SHARED_LIBS)
  # cmake-lint: disable=C0103
  set(BUILD_SHARED_LIBS
      ON
      CACHE INTERNAL ""
  )
endif()

# Remove old C++ flags used by libcoro, which isn't supported by TIDY.
get_target_property(flags libcoro COMPILE_OPTIONS)
list(FILTER flags EXCLUDE REGEX ".*-fconcepts.*|.*-fcoroutines.*")
set_target_properties(libcoro PROPERTIES COMPILE_OPTIONS "${flags}")
get_target_property(flags libcoro INTERFACE_COMPILE_OPTIONS)
list(FILTER flags EXCLUDE REGEX ".*-fconcepts.*|.*-fcoroutines.*")
set_target_properties(libcoro PROPERTIES INTERFACE_COMPILE_OPTIONS "${flags}")
