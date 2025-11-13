# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from rapidsmpf.buffer.buffer cimport MemoryType as cpp_MemoryType
from rapidsmpf.buffer.buffer import MemoryType


@dataclass
class ContentDescription:
    content_sizes: dict[MemoryType, int]
    spillable: bool


cdef content_description_from_cpp(cpp_ContentDescription cd):
    cdef dict content_sizes = {}
    for mem_type in MemoryType:
        content_sizes[mem_type] = cd.content_size(mem_type)
    return ContentDescription(content_sizes, cd.spillable())


cdef extern from * nogil:
    """
    rapidsmpf::ContentDescription cpp_content_description_new(
        bool spillable
    ) {
        return {
            spillable?
                rapidsmpf::ContentDescription::Spillable::YES:
                rapidsmpf::ContentDescription::Spillable::NO
        };
    }

    void cpp_content_description_set_size(
        rapidsmpf::ContentDescription &cd,
        rapidsmpf::MemoryType mem_type,
        size_t size
    ) noexcept {
        cd.content_size(mem_type) = size;
    }
    """
    cpp_ContentDescription cpp_content_description_new(bool_t) except+
    cpp_ContentDescription cpp_content_description_set_size(
        cpp_ContentDescription&, cpp_MemoryType, size_t
    ) noexcept


cdef cpp_ContentDescription content_description_to_cpp(object cd):
    assert isinstance(cd, ContentDescription)
    cdef cpp_ContentDescription ret = cpp_content_description_new(cd.spillable)
    for mem_type, size in cd.content_sizes:
        cpp_content_description_set_size(ret, mem_type, size)
    return ret
