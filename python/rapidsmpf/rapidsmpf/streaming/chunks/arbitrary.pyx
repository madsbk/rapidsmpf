# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cpython.object cimport PyObject
from cpython.ref cimport Py_DECREF, Py_INCREF
from libc.stdint cimport uint64_t
from libcpp.memory cimport make_unique, unique_ptr

from rapidsmpf.buffer.content_description cimport (content_description_to_cpp,
                                                   cpp_ContentDescription)
from rapidsmpf.streaming.chunks.utils cimport py_deleter
from rapidsmpf.streaming.core.message cimport Message, cpp_Message

from rapidsmpf.buffer.content_description import ContentDescription


cdef extern from * nogil:
    """
    namespace {
    rapidsmpf::streaming::Message cpp_to_message(
        std::uint64_t sequence_number,
        std::unique_ptr<rapidsmpf::streaming::OwningWrapper> obj,
        rapidsmpf::ContentDescription cd
    ) {
        return rapidsmpf::streaming::Message{
            sequence_number,
            std::move(obj),
            cd
        };
    }
    }
    """
    cpp_Message cpp_to_message(
        uint64_t, unique_ptr[cpp_OwningWrapper], cpp_ContentDescription
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
    def __init__(self, object obj, content_description = None):
        self._obj = obj
        self._content_description = content_description
        if self._content_description is None:
            self._content_description = ContentDescription(
                content_sizes={}, spillable=False
            )

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
        cdef object obj = <object><PyObject *>message._handle.release[
            cpp_OwningWrapper
        ]().release()
        Py_DECREF(obj)  # Cast to object increfs, so we must decref here
        return ArbitraryChunk(obj, cd)

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

        obj = self.release()
        Py_INCREF(obj)
        message._handle = cpp_to_message(
            sequence_number,
            make_unique[cpp_OwningWrapper](
                <void *><PyObject *>obj, py_deleter
            ),
            content_description_to_cpp(self._content_description)
        )
