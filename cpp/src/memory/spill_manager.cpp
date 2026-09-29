/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/spill_manager.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {


SpillManager::SpillManager(
    BufferResource* br, std::optional<Duration> periodic_spill_check
)
    : br_{br} {
    if (periodic_spill_check.has_value()) {
        periodic_spill_thread_.emplace(
            [this]() { spill_to_make_headroom(0); }, *periodic_spill_check
        );
    }
}

SpillManager::~SpillManager() {
    if (periodic_spill_thread_.has_value()) {
        periodic_spill_thread_->stop();
    }
}

SpillManager::SpillFunctionID SpillManager::add_spill_function(
    SpillFunction spill_function, int priority
) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto const id = spill_function_id_counter_++;
    RAPIDSMPF_EXPECTS(
        spill_functions_.insert({id, std::move(spill_function)}).second,
        "corrupted id counter"
    );
    spill_function_priorities_.insert({priority, id});

    // Make sure the spill thread is running.
    if (periodic_spill_thread_.has_value()) {
        periodic_spill_thread_->resume();
    }
    return id;
}

void SpillManager::remove_spill_function(SpillFunctionID fid) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& prio = spill_function_priorities_;
    for (auto it = prio.begin(); it != prio.end(); ++it) {
        if (it->second == fid) {
            prio.erase(it);  // Erase the first occurrence
            break;  // Exit after erasing to ensure only the first one is removed
        }
    }
    spill_functions_.erase(fid);

    // Asynchronously pause the spill thread if no spill functions are left.
    if (periodic_spill_thread_.has_value() && spill_functions_.empty()) {
        periodic_spill_thread_->pause_nb();
    }
}

std::size_t SpillManager::spill_unsafe(std::size_t amount) {
    // The loop below would exit immediately anyway, and a zero-byte ask is not a spill
    // that an observer should wait for.
    if (amount == 0) {
        return 0;
    }

    // Raised for the whole call so `spilling_now()` covers the spill functions, and
    // scoped so a throwing spill function cannot leave the count raised. The generation
    // is bumped before the count drops, so an observer that sees no spill in flight also
    // sees the generation that spill produced.
    struct InFlightGuard {
        SpillManager* self;

        ~InFlightGuard() {
            self->spill_generation_.fetch_add(1, std::memory_order_release);
            self->spills_in_flight_.fetch_sub(1, std::memory_order_release);
        }
    };

    spills_in_flight_.fetch_add(1, std::memory_order_release);
    InFlightGuard const in_flight{this};

    auto const statistics = br_->statistics();
    bool const record = statistics->enabled();

    std::size_t spilled{0};
    for (auto const [priority, fid] : spill_function_priorities_) {
        if (spilled >= amount) {
            break;
        }
        auto const freed = spill_functions_.at(fid)(amount - spilled);
        spilled += freed;
        if (record) {
            // By priority, since that decides which spill function gets the chance to
            // free the memory.
            statistics->add_bytes_stat(
                "spill-freed-bytes-priority" + std::to_string(priority), freed
            );
        }
    }

    // Spilling works in whole buffers, so what it frees rarely matches what was asked
    // for, and the excess is device memory nobody requested.
    if (record) {
        statistics->add_bytes_stat("spill-freed-bytes", spilled);
        statistics->add_bytes_stat(
            "spill-excess-bytes", spilled > amount ? spilled - amount : 0
        );
    }
    return spilled;
}

bool SpillManager::spilling_now() const noexcept {
    return spills_in_flight_.load(std::memory_order_acquire) > 0;
}

std::uint64_t SpillManager::spill_generation() const noexcept {
    return spill_generation_.load(std::memory_order_acquire);
}

std::size_t SpillManager::spill(std::size_t amount) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    std::lock_guard<std::mutex> lock(mutex_);
    return spill_unsafe(amount);
}

SpillManager::HeadroomResult SpillManager::spill_to_make_headroom(std::int64_t headroom) {
    RAPIDSMPF_NVTX_FUNC_RANGE();
    std::lock_guard<std::mutex> lock(mutex_);
    // TODO: check other memory types.
    std::int64_t const available =
        br_->memory_available_for_reservation(MemoryType::DEVICE);
    if (headroom <= available) {
        return {.deficit = 0, .spilled = 0};
    }
    auto const deficit = safe_cast<std::size_t>(headroom - available);
    return {.deficit = deficit, .spilled = spill_unsafe(deficit)};
}

}  // namespace rapidsmpf
