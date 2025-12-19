# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from libcpp.memory cimport shared_ptr
from libcpp.utility cimport move

from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.core.context cimport Context, cpp_Context
from rapidsmpf.streaming.core.message cimport Message, cpp_Message
from rapidsmpf.streaming.core.utilities cimport cython_invoke_python_function

import asyncio


cdef void cpp_set_py_future(void* py_future_context, const char *error_msg) noexcept nogil:
    """py_future_obj is an asyncio.Future"""

    with gil:
        loop, future = (<object?>py_future_context)
        assert loop is future.get_loop()
        if error_msg == NULL:
            print("cpp_set_py_future() - future.set_result")
            loop.call_soon_threadsafe(future.set_result, None)
        else:
            print("cpp_set_py_future() - future.set_exception")
            loop.call_soon_threadsafe(
                future.set_exception, RuntimeError(error_msg.decode("utf-8"))
            )
