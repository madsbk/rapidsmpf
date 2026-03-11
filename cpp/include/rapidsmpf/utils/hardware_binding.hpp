/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <string>

namespace rapidsmpf {

/**
 * @brief Options controlling which hardware bindings are applied.
 */
struct HardwareBindingOptions {
    bool cpu{true};  ///< Apply CPU affinity (sched_setaffinity).
    bool memory{true};  ///< Apply NUMA memory binding (numa_set_membind).
    bool network{true};  ///< Set UCX_NET_DEVICES environment variable.
};

/**
 * @brief Apply hardware bindings for the GPU identified by an integer index.
 *
 * Discovers system topology and applies CPU affinity, NUMA memory binding,
 * and/or UCX network device configuration for the specified GPU, according to
 * the provided options.
 *
 * @param gpu_index Device index of the GPU (e.g., as reported by CUDA runtime).
 * @param opts Options specifying which bindings to apply.
 *
 * @throws std::logic_error if topology discovery fails or the GPU is not found.
 */
void apply_hardware_bindings(int gpu_index, HardwareBindingOptions opts = {});

/**
 * @brief Apply hardware bindings for the GPU identified by a UUID string.
 *
 * Discovers system topology and applies CPU affinity, NUMA memory binding,
 * and/or UCX network device configuration for the specified GPU, according to
 * the provided options.
 *
 * @param gpu_uuid UUID string of the GPU (e.g., "GPU-abc123...").
 * @param opts Options specifying which bindings to apply.
 *
 * @throws std::logic_error if topology discovery fails or the GPU is not found.
 */
void apply_hardware_bindings(
    std::string const& gpu_uuid, HardwareBindingOptions opts = {}
);

}  // namespace rapidsmpf
