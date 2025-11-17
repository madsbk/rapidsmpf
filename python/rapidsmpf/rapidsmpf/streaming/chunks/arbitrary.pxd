# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr


cdef extern from "<rapidsmpf/streaming/cudf/owning_wrapper.hpp>" nogil:
    cdef cppclass cpp_OwningWrapper "rapidsmpf::streaming::OwningWrapper":
        cpp_OwningWrapper(void *, void(*)(void*)) noexcept
        void* get() noexcept
        void* release() noexcept


cdef class ArbitraryChunk:
    cdef object _obj
    cdef object _content_description
    cdef object _copy_cb
