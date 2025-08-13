# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from rmm.librmm.cuda_stream_view cimport cuda_stream_view

from rapidsmpf.buffer.resource cimport BufferResource, cpp_BufferResource
from rapidsmpf.communicator.communicator cimport Communicator
from rapidsmpf.config cimport Options
from rapidsmpf.statistics cimport Statistics

from rapidsmpf.config import get_environment_variables

from libcpp.memory cimport make_unique
from rmm.pylibrmm.stream cimport Stream


cdef class Context:
    def __cinit__(
        self,
        *,
        Communicator comm,
        stream,
        BufferResource br,
        options: Options = Options(),
        Statistics statistics = None,
    ):
        if stream is None:
            raise ValueError("stream cannot be None")
        self._stream = Stream(stream)
        self._comm = comm
        self._br = br
        # Insert missing config options from environment variables.
        options.insert_if_absent(get_environment_variables())
        if statistics is None:
            statistics = Statistics(enable=False)  # Disables statistics.

        cdef cpp_BufferResource* br_ = br.ptr()
        cdef cuda_stream_view _stream = self._stream.view()
        with nogil:
            self._handle = make_unique[cpp_Context](
                options._handle,
                self._comm._handle,
                br_,
                statistics._handle,
            )
