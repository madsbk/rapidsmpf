# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from rapidsmpf.owning_wrapper cimport cpp_OwningWrapper


cdef extern from * nogil:
    """
    #include <iostream>
    #include <coro/task.hpp>

    coro::task<void> cython_libcoro_task_wrapper(
        void (*cpp_set_py_future)(void*, const char *),
        rapidsmpf::OwningWrapper py_future_context,
        coro::task<void> task
    ) {
        std::cout << "cython_libcoro_task_wrapper()" << std::endl;
        try{
            co_await task;
            cpp_set_py_future(py_future_context.get(), NULL);
        } catch(std::exception const& e) {
            std::cerr << "cython_libcoro_task_wrapper() - unhandled exception: "
                      << e.what() << std::endl;
            cpp_set_py_future(py_future_context.get(), e.what());
        }
    }
    """

cdef void cpp_set_py_future(void* py_future_context, const char *error_msg) noexcept nogil
