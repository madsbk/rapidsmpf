/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <rapidsmpf/streaming/core/spillable_messages.hpp>

namespace rapidsmpf::streaming {

SpillableMessages::SpillableMessages(std::shared_ptr<Statistics> statistics)
    : statistics_{std::move(statistics)} {
    RAPIDSMPF_EXPECTS(statistics_ != nullptr, "statistics cannot be NULL");
}

std::optional<Duration> SpillableMessages::measure_residence_unsafe(
    MessageId mid, bool end_interval
) {
    auto it = spilled_at_.find(mid);
    if (it == spilled_at_.end()) {
        return std::nullopt;  // Never spilled.
    }
    auto const ret = Duration{Clock::now() - it->second};
    if (end_interval) {
        spilled_at_.erase(it);
    }
    return ret;
}

SpillableMessages::MessageId SpillableMessages::insert(Message&& message) {
    std::lock_guard<std::mutex> lock(global_mutex_);
    content_descriptions_.insert({counter_, message.content_description()});
    items_.insert({counter_, std::make_shared<Item>(std::move(message))});
    return counter_++;
}

Message SpillableMessages::extract(MessageId mid) {
    std::unique_lock global_lock(global_mutex_);
    std::shared_ptr<Item> item = extract_item(items_, mid).second;
    content_descriptions_.erase(mid);
    auto const residence = measure_residence_unsafe(mid, /* end_interval = */ true);
    global_lock.unlock();

    // Recorded outside the lock, since `Statistics` takes one of its own.
    if (residence.has_value()) {
        statistics_->add_duration_stat("message-spill-residence-time", *residence);
    }

    // If the item is being spilled, we block here until the spilling is done.
    std::unique_lock item_lock(item->mutex);
    return std::exchange(item->message, std::nullopt).value();
}

Message SpillableMessages::copy(MessageId mid, MemoryReservation& reservation) {
    // Find item, if it exist.
    std::unique_lock global_lock(global_mutex_);
    auto item_it = items_.find(mid);
    RAPIDSMPF_EXPECTS(
        item_it != items_.end(),
        "message not found " + std::to_string(mid),
        std::out_of_range
    );
    std::shared_ptr<Item> item = item_it->second;
    // Rematerialising to device means the message was needed again. It stays in the
    // container, so the entry is kept and a further copy records another sample.
    auto const residence = reservation.mem_type() == MemoryType::DEVICE
                               ? measure_residence_unsafe(mid, /* end_interval = */ false)
                               : std::nullopt;
    global_lock.unlock();

    // Acquire the item's lock and verify that it still holds a message,
    // since it may have been extracted while the global lock was released.
    std::unique_lock item_lock(item->mutex);
    RAPIDSMPF_EXPECTS(
        item->message.has_value(),
        "message not found " + std::to_string(mid),
        std::out_of_range
    );
    auto ret = item->message->copy(reservation);
    item_lock.unlock();

    // Recorded only once the copy succeeded, so a copy that lost the race against an
    // extraction adds no sample. Outside the locks, since `Statistics` takes one of
    // its own.
    if (residence.has_value()) {
        statistics_->add_duration_stat("message-spill-residence-time", *residence);
    }
    return ret;
}

std::size_t SpillableMessages::spill(MessageId mid, BufferResource* br) const {
    // Find item, if it exist.
    std::unique_lock global_lock(global_mutex_);
    auto item_it = items_.find(mid);
    if (item_it == items_.end()) {
        return 0;
    }
    std::shared_ptr<Item> item = item_it->second;
    global_lock.unlock();

    // Acquire the item's lock and verify that it still holds a message,
    // since it may have been extracted while the global lock was released.
    std::unique_lock item_lock(item->mutex, std::try_to_lock);
    if (!item_lock.owns_lock()) {
        return 0;
    }
    if (!item->message.has_value()) {
        return 0;
    }

    // Ensure the item still contains something to spill.
    auto const& msg = item->message.value();
    auto const old_cd = msg.content_description();
    if (!old_cd.spillable() || old_cd.content_size(MemoryType::DEVICE) == 0) {
        return 0;
    }

    // Spill item in-place.
    auto res = br->reserve_or_fail(msg.copy_cost(), SPILL_TARGET_MEMORY_TYPES);
    item->message = msg.copy(res);
    auto const new_cd = item->message.value().content_description();
    // Sampled while the item lock is still held, so the residence interval starts when
    // the data left device memory rather than when `global_mutex_` became available.
    auto const spilled_at = Clock::now();
    item_lock.unlock();

    // Update the content descriptions only if `mid` still exists.
    // This handles the case where it may have been extracted while the item lock
    // was released and simultaneously `extract` acquired global/item locks and
    // released the item.
    auto const spilled_bytes = old_cd.content_size(MemoryType::DEVICE);
    global_lock.lock();
    if (auto it = content_descriptions_.find(mid); it != content_descriptions_.end()) {
        it->second = new_cd;
        // Start the residence interval. Skipped when `mid` is gone, since a message
        // extracted mid-spill has no residence to measure.
        spilled_at_[mid] = spilled_at;
    }
    global_lock.unlock();

    statistics_->add_bytes_stat("message-spill-bytes", spilled_bytes);
    return spilled_bytes;
}

std::map<SpillableMessages::MessageId, ContentDescription>
SpillableMessages::get_content_descriptions() const {
    std::unique_lock global_lock(global_mutex_);
    return content_descriptions_;
}

ContentDescription rapidsmpf::streaming::SpillableMessages::get_content_description(
    MessageId mid
) const {
    std::lock_guard global_lock(global_mutex_);
    auto it = content_descriptions_.find(mid);
    RAPIDSMPF_EXPECTS(
        it != content_descriptions_.end(),
        "message not found " + std::to_string(mid),
        std::out_of_range
    );
    return it->second;
}

void SpillableMessages::clear() {
    std::lock_guard global_lock(global_mutex_);
    items_.clear();
    content_descriptions_.clear();
    spilled_at_.clear();
}

}  // namespace rapidsmpf::streaming
