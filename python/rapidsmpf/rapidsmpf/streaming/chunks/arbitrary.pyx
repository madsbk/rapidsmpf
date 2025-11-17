# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_DECREF, Py_INCREF
from libc.stdint cimport uint64_t
from libcpp.memory cimport make_unique, unique_ptr

from rapidsmpf._detail.exception_handling cimport (
    CppExcept, throw_py_as_cpp_exception, translate_py_to_cpp_exception)
from rapidsmpf.buffer.content_description cimport (content_description_to_cpp,
                                                   cpp_ContentDescription)
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.core.message cimport Message, cpp_Message
from rapidsmpf.streaming.core.utilities cimport cython_invoke_python_function

from rapidsmpf.buffer.content_description import ContentDescription

cdef void cython_invoke_copy_cb(
    void* py_copy_cb,
    void* py_obj
) noexcept nogil:
    cdef CppExcept err
    with gil:
        try:
            (<object?>py_copy_cb)(<object?>py_obj)
            return
        except BaseException as e:
            err = translate_py_to_cpp_exception(e)
    throw_py_as_cpp_exception(err)


cdef extern from * nogil:
    """
    namespace {
    rapidsmpf::streaming::Message cpp_to_message(
        std::uint64_t sequence_number,
        std::unique_ptr<rapidsmpf::streaming::OwningWrapper> payload,
        rapidsmpf::ContentDescription cd,
        void (*py_invoker)(void*, void*),
        void *py_copy_cb
    ) {
        using namespace rapidsmpf;
        using namespace rapidsmpf::streaming;

        Message::CopyCallback cb = [py_invoker, py_copy_cb](
            Message const& msg, MemoryReservation&
        ) {
            void const* py_obj = msg.get<rapidsmpf::streaming::OwningWrapper>().get()
            py_invoker(py_function, py_obj);
        };


        return rapidsmpf::streaming::Message{
            sequence_number,
            std::move(payload),
            cd,
        };
    }
    }
    """
    cpp_Message cpp_to_message(
        uint64_t,
        unique_ptr[cpp_OwningWrapper],
        cpp_ContentDescription,
        void (*py_invoker)(void*),
        void *py_copy_cb
    ) except +


cdef class ArbitraryChunk:
    """
    A chunk containing an arbitrary Python object as a payload.

    Parameters
    ----------
    obj
        The payload object.

    Notes
    -----
    To extract the object from the chunk, use :meth:`release`. The object
    is stored in a unique pointer with a custom deleter, so it is safe to
    drop the chunk in C++: deallocation will acquire the gil and decref the
    stored object.
    """
    def __init__(self, object obj, content_description = None, copy_cb = None):
        self._obj = obj
        self._content_description = content_description
        if self._content_description is None:
            self._content_description = ContentDescription(
                content_sizes={}, spillable=False
            )
        self._copy_cb = copy_cb

    @classmethod
    def __class_getitem__(cls, args):
        return cls

    def release(self):
        """
        Release and return the Python payload.

        Returns
        -------
        The underlying Python object.

        Raises
        ------
        ValueError
            If the chunk has already been released.

        Warnings
        --------
        The ArbitraryChunk is released and must not be used after this call.
        """
        if self._obj is None:
            raise ValueError("Chunk is uninitialized, has it already been released?")
        self._obj, ret = None, self._obj
        self._copy_cb = None
        return ret

    @property
    def content_description(self):
        """
        Return the associated content description.

        Returns
        -------
        content description of this the chunk.
        """
        return self._content_description

    @staticmethod
    def from_message(Message message not None):
        """
        Construct an ArbitraryChunk by consuming a Message.

        Parameters
        ----------
        message
            Message containing an ArbitraryChunk. The message is released
            and is empty after this call.

        Returns
        -------
        A new ArbitraryChunk extracted from the given message.
        """
        # Extract the payload from message and create a new chunk wrapping it.
        cd = message.get_content_description()
        cdef object payload = <object><PyObject *>message._handle.release[
            cpp_OwningWrapper
        ]().release()
        Py_DECREF(payload)  # Cast to object increfs, so we must decref here
        obj, copy_cb = payload
        return ArbitraryChunk(obj, cd, copy_cb)

    def into_message(self, uint64_t sequence_number, Message message not None):
        """
        Move this ArbitraryChunk into a Message.

        This method is not typically called directly. Instead, it is invoked by
        `Message.__init__()` when creating a new Message with this ArbitraryChunk
        as its payload.

        Parameters
        ----------
        sequence_number
            Ordering identifier for the message.
        message
            Message object that will take ownership of this ArbitraryChunk.

        Raises
        ------
        ValueError
            If the provided message is not empty.

        Warnings
        --------
        The ArbitraryChunk is released and must not be used after this call.
        """
        if not message.empty():
            raise ValueError("cannot move into a non-empty message")

        # We safe the object and its copy callback function in the message.
        cdef object payload = (self._obj, self._copy_cb)
        Py_INCREF(payload)
        message._handle = cpp_to_message(
            sequence_number,
            make_unique[cpp_OwningWrapper](
                <void *><PyObject *>payload, py_deleter
            ),
            content_description_to_cpp(self._content_description),
            cython_invoke_copy_cb,
            <void *>self._copy_cb
        )
        self.release()  # Clear current ArbitraryChunk instance.
