/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include <sched.h>
#include <unistd.h>

#include <rapidsmpf/utils/hardware_binding.hpp>

#if RAPIDSMPF_HAVE_NUMA
#include <numa.h>
#endif

#include <cucascade/memory/topology_discovery.hpp>

#include <rapidsmpf/error.hpp>

namespace rapidsmpf {

namespace {

/**
 * @brief Parse a CPU list string into a cpu_set_t mask.
 *
 * Accepts strings of the form "0-31,128-159" as produced by
 * `cucascade::memory::gpu_topology_info::cpu_affinity_list`.
 *
 * @param cpulist CPU list string.
 * @param cpuset  Output cpu_set_t to populate.
 * @return true on success, false if the string is empty or malformed.
 */
bool parse_cpu_list_to_mask(std::string const& cpulist, cpu_set_t* cpuset) {
    CPU_ZERO(cpuset);
    if (cpulist.empty()) {
        return false;
    }

    std::istringstream iss(cpulist);
    std::string token;
    while (std::getline(iss, token, ',')) {
        std::size_t const dash_pos = token.find('-');
        if (dash_pos != std::string::npos) {
            try {
                int const start = std::stoi(token.substr(0, dash_pos));
                int const end = std::stoi(token.substr(dash_pos + 1));
                for (int i = start; i <= end; ++i) {
                    if (i >= 0 && i < static_cast<int>(CPU_SETSIZE)) {
                        CPU_SET(static_cast<unsigned>(i), cpuset);
                    }
                }
            } catch (...) {
                return false;
            }
        } else {
            try {
                int const core = std::stoi(token);
                if (core >= 0 && core < static_cast<int>(CPU_SETSIZE)) {
                    CPU_SET(static_cast<unsigned>(core), cpuset);
                }
            } catch (...) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Set CPU affinity for the current process.
 *
 * @param cpu_affinity_list CPU affinity list string (e.g., "0-31,128-159").
 * @return true on success, false on failure.
 */
bool set_cpu_affinity(std::string const& cpu_affinity_list) {
    if (cpu_affinity_list.empty()) {
        return false;
    }

    cpu_set_t cpuset;
    if (!parse_cpu_list_to_mask(cpu_affinity_list, &cpuset)) {
        return false;
    }

    pid_t const pid = getpid();
    return sched_setaffinity(pid, sizeof(cpu_set_t), &cpuset) == 0;
}

/**
 * @brief Set NUMA memory binding for the current process.
 *
 * @param memory_binding Vector of NUMA node IDs to bind memory to.
 * @return true on success, false on failure or if NUMA is not available.
 */
bool set_numa_memory_binding(std::vector<int> const& memory_binding) {
#if RAPIDSMPF_HAVE_NUMA
    if (memory_binding.empty()) {
        return false;
    }

    if (numa_available() == -1) {
        return false;
    }

    struct bitmask* nodemask = numa_allocate_nodemask();
    if (!nodemask) {
        return false;
    }

    numa_bitmask_clearall(nodemask);
    for (int const node : memory_binding) {
        if (node >= 0) {
            numa_bitmask_setbit(nodemask, static_cast<unsigned int>(node));
        }
    }

    numa_set_membind(nodemask);
    numa_free_nodemask(nodemask);

    return true;
#else
    std::ignore = memory_binding;
    return false;
#endif
}

/**
 * @brief Set UCX_NET_DEVICES environment variable.
 *
 * @param network_devices List of network device names.
 */
void set_ucx_net_devices(std::vector<std::string> const& network_devices) {
    if (network_devices.empty()) {
        return;
    }

    std::string ucx_net_devices;
    for (std::size_t i = 0; i < network_devices.size(); ++i) {
        if (i > 0) {
            ucx_net_devices += ",";
        }
        ucx_net_devices += network_devices[i];
    }
    ::setenv("UCX_NET_DEVICES", ucx_net_devices.c_str(), 1);
}

/**
 * @brief Apply bindings for a single GPU topology entry.
 *
 * @param gpu_info GPU topology information.
 * @param opts     Which bindings to apply.
 */
void apply_bindings_for_gpu(
    cucascade::memory::gpu_topology_info const& gpu_info,
    HardwareBindingOptions const& opts
) {
    if (opts.cpu && !gpu_info.cpu_affinity_list.empty()) {
        RAPIDSMPF_EXPECTS(
            set_cpu_affinity(gpu_info.cpu_affinity_list),
            "Failed to set CPU affinity for GPU " + std::to_string(gpu_info.id)
        );
    }

    if (opts.memory && !gpu_info.memory_binding.empty()) {
#if RAPIDSMPF_HAVE_NUMA
        RAPIDSMPF_EXPECTS(
            set_numa_memory_binding(gpu_info.memory_binding),
            "Failed to set NUMA memory binding for GPU " + std::to_string(gpu_info.id)
        );
#endif
    }

    if (opts.network && !gpu_info.network_devices.empty()) {
        set_ucx_net_devices(gpu_info.network_devices);
    }
}

/**
 * @brief Discover system topology, throwing on failure.
 */
cucascade::memory::system_topology_info discover_topology() {
    cucascade::memory::topology_discovery discovery;
    RAPIDSMPF_EXPECTS(discovery.discover(), "Failed to discover system topology");
    return discovery.get_topology();
}

}  // anonymous namespace

void apply_hardware_bindings(int gpu_index, HardwareBindingOptions opts) {
    RAPIDSMPF_EXPECTS(gpu_index >= 0, "GPU index must be non-negative");

    auto const topology = discover_topology();

    auto const it = std::ranges::find_if(
        topology.gpus, [gpu_index](cucascade::memory::gpu_topology_info const& gpu) {
            return static_cast<int>(gpu.id) == gpu_index;
        }
    );

    RAPIDSMPF_EXPECTS(
        it != topology.gpus.end(),
        "GPU with index " + std::to_string(gpu_index) + " not found in topology"
    );

    apply_bindings_for_gpu(*it, opts);
}

void apply_hardware_bindings(std::string const& gpu_uuid, HardwareBindingOptions opts) {
    RAPIDSMPF_EXPECTS(!gpu_uuid.empty(), "GPU UUID must not be empty");

    auto const topology = discover_topology();

    auto const it = std::ranges::find_if(
        topology.gpus, [&gpu_uuid](cucascade::memory::gpu_topology_info const& gpu) {
            return gpu.uuid == gpu_uuid;
        }
    );

    RAPIDSMPF_EXPECTS(
        it != topology.gpus.end(),
        "GPU with UUID '" + gpu_uuid + "' not found in topology"
    );

    apply_bindings_for_gpu(*it, opts);
}

}  // namespace rapidsmpf
