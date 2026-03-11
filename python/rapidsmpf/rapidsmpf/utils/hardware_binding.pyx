# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.string cimport string


cdef extern from "<rapidsmpf/utils/hardware_binding.hpp>" namespace "rapidsmpf" nogil:
    cdef struct cpp_HardwareBindingOptions "rapidsmpf::HardwareBindingOptions":
        bint cpu
        bint memory
        bint network

    void cpp_apply_by_index \
        "rapidsmpf::apply_hardware_bindings"(
            int gpu_index, cpp_HardwareBindingOptions opts) except +

    void cpp_apply_by_uuid \
        "rapidsmpf::apply_hardware_bindings"(
            const string& gpu_uuid, cpp_HardwareBindingOptions opts) except +


def apply_hardware_bindings(
    int gpu_index, *, bint cpu=True, bint memory=True, bint network=True
):
    """
    Apply CPU affinity, NUMA memory binding, and UCX_NET_DEVICES for a GPU.

    Discovers system topology and applies the requested hardware bindings for
    the GPU identified by its integer device index.

    Parameters
    ----------
    gpu_index
        Integer device index of the GPU (e.g., as used by the CUDA runtime or
        set via ``CUDA_VISIBLE_DEVICES``).
    cpu
        If True, apply CPU affinity (``sched_setaffinity``).
    memory
        If True, apply NUMA memory binding (``numa_set_membind``).
    network
        If True, set the ``UCX_NET_DEVICES`` environment variable.

    Raises
    ------
    RuntimeError
        If topology discovery fails or the GPU index is not found.
    """
    cdef cpp_HardwareBindingOptions opts
    opts.cpu = cpu
    opts.memory = memory
    opts.network = network
    with nogil:
        cpp_apply_by_index(gpu_index, opts)


def apply_hardware_bindings_by_uuid(
    str gpu_uuid, *, bint cpu=True, bint memory=True, bint network=True
):
    """
    Apply CPU affinity, NUMA memory binding, and UCX_NET_DEVICES for a GPU.

    Discovers system topology and applies the requested hardware bindings for
    the GPU identified by its UUID string.

    Parameters
    ----------
    gpu_uuid
        UUID string of the GPU (e.g., ``"GPU-abc123..."``), as reported by
        NVML or the ``nvidia-smi -q`` command.
    cpu
        If True, apply CPU affinity (``sched_setaffinity``).
    memory
        If True, apply NUMA memory binding (``numa_set_membind``).
    network
        If True, set the ``UCX_NET_DEVICES`` environment variable.

    Raises
    ------
    RuntimeError
        If topology discovery fails or the GPU UUID is not found.
    """
    cdef cpp_HardwareBindingOptions opts
    opts.cpu = cpu
    opts.memory = memory
    opts.network = network
    cdef string uuid = gpu_uuid.encode()
    with nogil:
        cpp_apply_by_uuid(uuid, opts)
