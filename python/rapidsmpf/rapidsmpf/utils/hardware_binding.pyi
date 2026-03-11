# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

def apply_hardware_bindings(
    gpu_index: int,
    *,
    cpu: bool = True,
    memory: bool = True,
    network: bool = True,
) -> None: ...
def apply_hardware_bindings_by_uuid(
    gpu_uuid: str,
    *,
    cpu: bool = True,
    memory: bool = True,
    network: bool = True,
) -> None: ...
