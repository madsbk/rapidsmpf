/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <coro/sync_wait.hpp>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/memory_reserve_or_wait.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf::streaming {

namespace {

// EXPERIMENT, not for landing: `RAPIDSMPF_RESERVE_WAIT_FOR_SPILL=0` turns off waiting
// for an in-flight spill, which leaves the reservation path as it is on `main`, so one
// build serves both arms of an A/B.
bool wait_for_spill_enabled() {
    static bool const enabled = [] {
        char const* env = std::getenv("RAPIDSMPF_RESERVE_WAIT_FOR_SPILL");
        return env == nullptr || std::string_view{env} != "0";
    }();
    return enabled;
}

}  // namespace

MemoryReserveOrWait::MemoryReserveOrWait(
    config::Options options,
    MemoryType mem_type,
    std::shared_ptr<CoroThreadPoolExecutor> executor,
    std::shared_ptr<BufferResource> br
)
    : mem_type_{mem_type},
      executor_{std::move(executor)},
      br_{std::move(br)},
      timeout_{options.get<Duration>("memory_reserve_timeout", parse_duration)},
      spill_wait_cap_{10 * timeout_},
      stat_prefix_{"reserve-" + to_lower(to_string(mem_type)) + "-"} {
    RAPIDSMPF_EXPECTS(executor_ != nullptr, "executor cannot be NULL");
    RAPIDSMPF_EXPECTS(br_ != nullptr, "br cannot be NULL");

    // Shared with `br_` and `Context`, see `Context::statistics()`.
    statistics_ = br_->statistics();
}

void MemoryReserveOrWait::record_stat(std::string_view suffix, double value) const {
    if (!statistics_->enabled()) {
        return;
    }
    std::call_once(report_entries_once_, [this] {
        using Formatter = Statistics::Formatter;
        auto entry = [this](std::string_view name, Formatter formatter) {
            auto full = stat_prefix_ + std::string{name};
            statistics_->add_report_entry(
                full, std::vector<std::string>{full}, formatter
            );
        };
        entry("wait-avoided", Formatter::HitRate);
        entry("wait-timeout", Formatter::HitRate);
        entry("waiting-requests", Formatter::Gauge);
        entry("wait-satisfied-time", Formatter::Duration);
        entry("wait-timeout-time", Formatter::Duration);
        entry("request-bytes", Formatter::Bytes);
        entry("overbook-bytes", Formatter::Bytes);
        entry("wait-spill-extended", Formatter::HitRate);
        entry("wait-spill-rescued", Formatter::HitRate);
        entry("wait-spill-extension-time", Formatter::Duration);
        entry("wait-satisfied-peak-available-bytes", Formatter::Bytes);
        entry("wait-satisfied-request-bytes", Formatter::Bytes);
        entry("wait-timeout-peak-available-bytes", Formatter::Bytes);
        entry("wait-timeout-request-bytes", Formatter::Bytes);
        entry("wait-timeout-memory-was-available", Formatter::HitRate);
    });
    statistics_->add_stat(stat_prefix_ + std::string{suffix}, value);
}

MemoryReserveOrWait::~MemoryReserveOrWait() noexcept {
    coro::sync_wait(shutdown());
}

Actor MemoryReserveOrWait::shutdown() {
    // Move the pending requests and joinable periodic task out under the mutex,
    // then release the lock. Both the queue shutdown and the task await can block
    // or suspend, so they must not run while holding the mutex.
    std::unique_lock lock(mutex_);
    auto reservation_requests = std::move(reservation_requests_);
    auto periodic_memory_check_task =
        std::exchange(periodic_memory_check_task_, std::nullopt);
    lock.unlock();

    // Shut down all request queues so any waiters are unblocked, then wait for
    // the periodic task to exit (if one was running).
    if (!reservation_requests.empty()) {
        std::vector<Actor> actors;
        for (Request const& request : reservation_requests) {
            actors.push_back(request.queue.shutdown());
        }
        coro_results(co_await coro::when_all(std::move(actors)));
    }
    if (periodic_memory_check_task.has_value()) {
        co_await *periodic_memory_check_task;
    }
}

coro::task<MemoryReservation> MemoryReserveOrWait::reserve_or_wait(
    std::size_t size, std::int64_t net_memory_delta
) {
    record_stat("request-bytes", static_cast<double>(size));

    // First, check whether the requested memory is immediately available.
    auto [res, _] = br_->reserve(mem_type_, size, AllowOverbooking::NO);
    if (res.size() == size) {
        record_stat("wait-avoided", 1);
        co_return std::move(res);
    }
    record_stat("wait-avoided", 0);

    // Use libcoro's queue to track completion of this reservation request.
    // The queue will have at most one item: the fulfilled memory reservation.
    coro::queue<MemoryReservation> request_queue{};

    // Enqueue a reservation request under the mutex.
    std::unique_lock lock(mutex_);
    reservation_requests_.insert(
        Request{
            .size = size,
            .net_memory_delta = net_memory_delta,
            .sequence_number = sequence_counter++,
            .queue = request_queue,
            .submitted_at = Clock::now()
        }
    );
    auto const waiting_requests = reservation_requests_.size();

    // If no periodic memory check task is running, start one.
    std::optional<coro::task<void>> previous_periodic_task;
    if (!periodic_task_running_) {
        // A previous periodic task may exist but is guaranteed to be either already
        // finished or about to finish. This can happen when the last request was
        // extracted and the task is in the process of exiting.
        //
        // We take ownership of that task here and await it below before proceeding,
        // ensuring that at most one periodic task is active at any time.
        previous_periodic_task = std::move(periodic_memory_check_task_);
        periodic_memory_check_task_ = executor_->spawn_joinable(periodic_memory_check());
        // Claim the slot until the task releases it.
        periodic_task_running_ = true;
    }
    lock.unlock();

    // Recorded each time a request starts waiting, not sampled over time. The set
    // only grows at the insert above, so the maximum is exact, while the mean is the
    // queue depth seen when a request starts waiting.
    record_stat("waiting-requests", static_cast<double>(waiting_requests));

    // If a previous periodic task existed, wait for it to fully exit before
    // continuing. The await must happen without holding the mutex, otherwise the
    // periodic task could deadlock while trying to acquire the same mutex.
    if (previous_periodic_task.has_value()) {
        co_await *previous_periodic_task;
    }

    // Suspend until our request is fulfilled.
    auto request = co_await request_queue.pop();
    RAPIDSMPF_EXPECTS(
        request.has_value(), "memory reservation failed", std::runtime_error
    );
    co_return std::move(*request);
}

coro::task<std::pair<MemoryReservation, std::size_t>>
MemoryReserveOrWait::reserve_or_wait_or_overbook(
    std::size_t size, std::int64_t net_memory_delta
) {
    auto ret = co_await reserve_or_wait(size, net_memory_delta);
    if (ret.size() < size) {
        auto overbooked = br_->reserve(mem_type_, size, AllowOverbooking::YES);
        // `reserve()` returns the total overbooking after the reservation, including any
        // already outstanding, so clamp to `size` for the amount this request added.
        auto const added = std::min(size, overbooked.second);
        if (added > 0) {
            record_stat("overbook-bytes", static_cast<double>(added));
        }
        co_return overbooked;
    }
    co_return {std::move(ret), 0};
}

coro::task<MemoryReservation> MemoryReserveOrWait::reserve_or_wait_or_fail(
    std::size_t size, std::int64_t net_memory_delta
) {
    auto ret = co_await reserve_or_wait(size, net_memory_delta);
    RAPIDSMPF_EXPECTS(
        ret.size() == size,
        "cannot reserve " + std::string{to_string(mem_type_)} + " memory ("
            + format_nbytes(size) + ")",
        rapidsmpf::reservation_error
    );
    co_return ret;
}

std::size_t MemoryReserveOrWait::size() const noexcept {
    std::lock_guard lock(mutex_);
    return reservation_requests_.size();
}

std::size_t MemoryReserveOrWait::periodic_memory_check_counter() const noexcept {
    return periodic_memory_check_counter_.load(std::memory_order_acquire);
}

std::shared_ptr<CoroThreadPoolExecutor> const&
MemoryReserveOrWait::executor() const noexcept {
    return executor_;
}

std::shared_ptr<BufferResource> const& MemoryReserveOrWait::br() const noexcept {
    return br_;
}

Duration MemoryReserveOrWait::timeout() const noexcept {
    return timeout_;
}

coro::task<void> MemoryReserveOrWait::periodic_memory_check() {
    // Helper that returns the memory available for new reservations, clamped so
    // negative values become zero.
    auto memory_available = [this]() -> std::size_t {
        std::int64_t const ret = br_->memory_available_for_reservation(mem_type_);
        return safe_cast<std::size_t>(std::max(ret, std::int64_t{0}));
    };

    // Helper that returns the subrange of reservation requests with size <= max_size.
    auto eligible_requests = [this](std::size_t max_size)
        -> std::ranges::subrange<std::set<Request>::const_iterator> {
        // Since `reservation_requests_` is sorted by ascending size,
        // upper_bound finds the first element with size > max_size.
        auto last = std::ranges::upper_bound(
            reservation_requests_, max_size, std::less<>{}, &Request::size
        );
        // The range [begin, last) contains all requests with size <= max_size.
        return {reservation_requests_.begin(), last};
    };

    // Helper that pushes a memory reservation into a request's queue **without**
    // waiting on the coroutine.
    auto push_into_queue =
        [this](coro::queue<MemoryReservation>& queue, MemoryReservation res) -> void {
        auto err = executor_->spawn_detached(
            [](coro::queue<MemoryReservation>& queue, MemoryReservation res) -> Actor {
                RAPIDSMPF_EXPECTS(
                    co_await queue.push(std::move(res))
                        == coro::queue_produce_result::produced,
                    "could not push memory reservation"
                );
            }(queue, std::move(res))
        );
        RAPIDSMPF_EXPECTS(err, "cannot spawn push-into-queue task");
    };

    // RAII helper that releases `periodic_task_running_` when this task exits without
    // reaching one of the `co_return` paths below, such as on an exception. Those paths
    // release the flag under the same lock acquisition that observes the empty request
    // set, and dismiss the guard.
    struct RunningFlagGuard {
        MemoryReserveOrWait* self;

        ~RunningFlagGuard() {
            if (self != nullptr) {
                std::lock_guard lock(self->mutex_);
                self->periodic_task_running_ = false;
            }
        }

        void dismiss() noexcept {
            self = nullptr;
        }
    };

    RunningFlagGuard running_flag_guard{.self = this};

    while (true) {
        auto last_reservation_success = Clock::now();
        // When the deadline was first extended because a spill was running, so both
        // exits below can say whether the extension paid for itself. Reset with the
        // deadline, since each wait window is judged on its own.
        std::optional<Clock::time_point> extended_at;
        // Set while a spill is being waited on, so that when the spill ends the loop
        // takes one more admission pass instead of timing out on the memory it just
        // waited for.
        bool recheck_after_spill{false};
        // Set once a completed spill failed to free memory for this window. Final until
        // the next admission: spills keep coming back to back under pressure, and
        // letting the next one re-open the wait would ride the cap on every request.
        bool spill_stalled{false};
        // Spill count and available memory at the last point progress was judged, so a
        // completing spill can be asked whether it actually freed anything. Signed,
        // since while others have overbooked, a spill can make progress and still leave
        // the memory available below zero.
        std::uint64_t extend_generation{0};
        std::int64_t extend_available{0};
        while (true) {
            // Exit if no more pending requests remain.
            {
                std::unique_lock lock(mutex_);
                if (reservation_requests_.empty()) {
                    periodic_task_running_ = false;
                    running_flag_guard.dismiss();
                    co_return;
                }
            }
            periodic_memory_check_counter_.fetch_add(1, std::memory_order_acq_rel);
            co_await executor_->yield();
            if (auto const now = Clock::now(); now - last_reservation_success > timeout_)
            {
                // A spill in flight is memory about to arrive, and a spill takes longer
                // than the timeout for any partition of a useful size. Giving up now
                // overbooks memory that is already being freed. Nothing is spilled from
                // here: this only declines to give up during someone else's spill.
                auto& spill_manager = br_->spill_manager();
                if (extended_at.has_value() && !spill_stalled) {
                    if (auto const generation = spill_manager.spill_generation();
                        generation != extend_generation)
                    {
                        // A spill finished while this request waited. Carry on only
                        // while spilling still frees memory. Under pressure the spiller
                        // is busy nearly all the time, and waiting on one with nothing
                        // left to give would hold every request for the whole cap.
                        // Judging progress rather than elapsed time also keeps this
                        // correct when a spill is slow, such as one going to disk, or
                        // simply faster or slower on another machine. Judged whether or
                        // not another spill has started since, which is the usual case.
                        auto const available =
                            br_->memory_available_for_reservation(mem_type_);
                        spill_stalled = available <= extend_available;
                        extend_generation = generation;
                        extend_available = available;
                    }
                }
                // The spill manager only frees device memory. A device spill moves
                // data into host memory rather than out of it, so other memory types
                // have nothing to wait for.
                bool const keep_waiting =
                    wait_for_spill_enabled() && mem_type_ == MemoryType::DEVICE
                    && !spill_stalled
                    && now - last_reservation_success <= timeout_ + spill_wait_cap_
                    && spill_manager.spilling_now();

                if (keep_waiting) {
                    if (!extended_at.has_value()) {
                        extended_at = now;
                        extend_generation = spill_manager.spill_generation();
                        extend_available =
                            br_->memory_available_for_reservation(mem_type_);
                    }
                    // Carry on to the admission check below, since a release can make
                    // room while the spill is still running.
                    recheck_after_spill = true;
                } else if (recheck_after_spill) {
                    // The spill finished. Fall through to one admission pass, otherwise
                    // the freed memory is discarded and the request overbooks for it.
                    recheck_after_spill = false;
                } else if (extended_at.has_value()) {
                    // Extended and still timed out, so the wait was spent for nothing.
                    record_stat("wait-spill-extended", 1);
                    record_stat("wait-spill-rescued", 0);
                    record_stat(
                        "wait-spill-extension-time", Duration{now - *extended_at}.count()
                    );

                    // This is the only way out of the while-loop that doesn't shutdown
                    // the periodic memory check.
                    break;
                } else {
                    record_stat("wait-spill-extended", 0);
                    break;
                }
            }
            auto const max_size = memory_available();

            // Find the request with the smallest net_memory_delta that fits
            // into the currently available memory.
            std::unique_lock lock(mutex_);

            // Remember the most memory each still-pending request has seen available,
            // so both exits can say how close the request came on its own. Applied
            // after selection, so the pass that admits a request does not count towards
            // its own peak.
            auto note_peak = [&] {
                for (Request const& request : reservation_requests_) {
                    request.peak_available = std::max(request.peak_available, max_size);
                }
            };

            auto eligibles = eligible_requests(max_size);
            if (eligibles.empty()) {
                // Nothing currently fits. Preserve resident data while ordinary
                // admission waits for a reservation release; the timeout path
                // below remains responsible for bounded progress.
                note_peak();
                continue;  // No eligible requests.
            }

            auto it = std::ranges::min_element(
                eligibles, std::less<>{}, &Request::net_memory_delta
            );

            // Try to reserve memory for the selected request.
            auto [res, _] = br_->reserve(mem_type_, it->size, AllowOverbooking::NO);
            if (res.size() == 0) {
                note_peak();
                continue;  // Memory is no longer available.
            }

            // Read before extraction, so it covers only the passes this request
            // survived rather than the one that admitted it.
            auto const peak_before_admission = it->peak_available;

            // Extract the selected request and push the reservation into its queue.
            Request request = reservation_requests_.extract(it).value();
            note_peak();
            lock.unlock();
            last_reservation_success = Clock::now();
            spill_stalled = false;
            recheck_after_spill = false;

            if (extended_at.has_value()) {
                // Admitted during an extension, which is the case the extension exists
                // for: without it this request would have overbooked instead.
                record_stat("wait-spill-extended", 1);
                record_stat("wait-spill-rescued", 1);
                record_stat(
                    "wait-spill-extension-time",
                    Duration{last_reservation_success - *extended_at}.count()
                );
                extended_at.reset();
            }

            // Satisfied: a reservation release made room for this request, so it
            // did not reach the timeout.
            record_stat("wait-timeout", 0);
            record_stat(
                "wait-satisfied-time",
                Duration{last_reservation_success - request.submitted_at}.count()
            );
            record_stat(
                "wait-satisfied-peak-available-bytes",
                static_cast<double>(peak_before_admission)
            );
            record_stat(
                "wait-satisfied-request-bytes", static_cast<double>(request.size)
            );

            push_into_queue(request.queue, std::move(res));
        }

        // Reaching this point means we hit the timeout. Force bounded progress by
        // selecting among the smallest pending requests, preferring the one with the
        // smallest net_memory_delta. Do not spill queued data here: callers that
        // permit overbooking receive the zero-size reservation below and decide how
        // to proceed, while non-overbooking callers retain the existing failure path.
        std::unique_lock lock(mutex_);
        if (reservation_requests_.empty()) {
            periodic_task_running_ = false;
            running_flag_guard.dismiss();
            co_return;
        }

        // The set is sorted by size (ascending). First, find the smallest size.
        auto first = reservation_requests_.begin();
        auto const smallest_size = first->size;

        // Consider all requests with that size.
        auto same_size_end = std::ranges::upper_bound(
            reservation_requests_, smallest_size, std::less<>{}, &Request::size
        );

        // Among the smallest requests, pick the one with the smallest
        // net_memory_delta. If multiple requests tie, we pick the oldest one,
        // since the set is ordered by size and then sequence_number (ascending).
        auto it = std::ranges::min_element(
            std::ranges::subrange(first, same_size_end),
            std::less<>{},
            &Request::net_memory_delta
        );

        Request request = reservation_requests_.extract(it).value();
        lock.unlock();

        // Reserve memory and accept a zero-size result if it does not fit into the
        // currently available memory.
        auto [res, _] = br_->reserve(mem_type_, request.size, AllowOverbooking::NO);

        // Forced progress: this request ran out the timeout rather than being admitted
        // by a reservation release. Whether the forced attempt then found memory is
        // visible as the `overbook-bytes` count, which only moves when it did not.
        record_stat("wait-timeout", 1);
        record_stat(
            "wait-timeout-time", Duration{Clock::now() - request.submitted_at}.count()
        );
        // Whether the memory this request asked for was ever available while it
        // waited, with the two sides of that comparison over the same requests.
        record_stat(
            "wait-timeout-peak-available-bytes",
            static_cast<double>(request.peak_available)
        );
        record_stat("wait-timeout-request-bytes", static_cast<double>(request.size));
        record_stat(
            "wait-timeout-memory-was-available",
            request.peak_available >= request.size ? 1 : 0
        );

        push_into_queue(request.queue, std::move(res));
    }
}

coro::task<MemoryReservation> reserve_memory(
    std::shared_ptr<Context> ctx,
    std::size_t size,
    std::int64_t net_memory_delta,
    MemoryType mem_type,
    std::optional<AllowOverbooking> allow_overbooking
) {
    // If allow_overbooking is not specified, get it from the configuration options.
    if (!allow_overbooking.has_value()) {
        bool const allow_overbook_default =
            ctx->options().get<bool>("allow_overbooking_by_default", parse_string<bool>);
        allow_overbooking =
            allow_overbook_default ? AllowOverbooking::YES : AllowOverbooking::NO;
    }

    // Reserve memory based on the overbooking policy.
    if (allow_overbooking.value() == AllowOverbooking::YES) {
        auto [res, _] = co_await ctx->memory(mem_type)->reserve_or_wait_or_overbook(
            size, net_memory_delta
        );
        co_return std::move(res);
    } else {
        co_return co_await ctx->memory(mem_type)->reserve_or_wait_or_fail(
            size, net_memory_delta
        );
    }
}

}  // namespace rapidsmpf::streaming
