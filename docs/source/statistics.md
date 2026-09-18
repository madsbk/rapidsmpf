# Statistics

RapidsMPF can be configured to collect {term}`Statistics`, which can help you understand the performance of the system.
This table gives an overview of the different statistics collected.

| Name | Description |
| --- | --- |
| `alloc-{memtype}` | Bytes allocated via `BufferResource::allocate()`, broken down by memory type (`device`, `pinned_host`, `host`). Shows total bytes, total time, allocation throughput, and average stream delay. |
| `copy-{src}-to-{dst}` | Amount of data copied between memory types by RapidsMPF. `{src}` and `{dst}` are `device`, `pinned_host`, or `host`. Shows total bytes, total copy time, throughput, and average stream delay (time between CPU submission and GPU execution of the copy). |
| `buffer-spilled-time` | How long spilled data remains spilled until unspilled back into device memory. |
| `buffer-spilled-returned-bytes` | Spilled data that came back to device memory, which is the data `buffer-spilled-time` measures. |
| `buffer-spilled-not-returned-bytes` | Spilled data that ended its life off device, either consumed from host memory or still spilled when the run ended. A shuffle sends every outgoing chunk from host memory, so this is the normal outcome for that data rather than a sign of a good or bad spill. Together with the previous row it says what share of spilling `buffer-spilled-time` covers. |
| `spill-demand-bytes` | Device memory a reservation needed spilling to free, via `reserve_device_memory_and_spill()`. Only the part that reservation added, since the deficit the spill manager is asked for is the whole outstanding one and every reservation made while it stands asks for it again. A second way to overbook alongside `reserve-{memtype}-overbook-bytes`. |
| `spill-freed-bytes` | Device memory the spill manager actually freed. |
| `spill-candidates` | Spillable messages holding device memory that `spill_messages` had to choose between, recorded only when there was at least one. Which message is spilled can only matter in proportion to this, so it is the denominator for any claim about the selection order. |
| `spill-candidates-none` | Calls where `spill_messages` had nothing to spill, over all its calls. Whoever could spill cheaply has usually done so already, so an empty pool is common and is not the same as a small one. |
| `spill-candidate-bytes` | Device memory held by those candidates, which says whether the pool could have covered the request at all. |
| `spill-unmet-bytes` | Demanded device memory that spilling did not free. |
| `spill-excess-bytes` | Device memory freed beyond what was asked for. Spilling works in whole buffers, so a small request can free a large buffer, and the excess is memory nobody requested. |
| `event-loop-total` | Time spent in in the background `ProgressThread` event-loop. |
| `recv-into-host-memory` | Data received directly into host memory rather than device memory, due to memory pressure at receive time. |
| `reserve-{memtype}-wait-avoided` | Reservation requests that `MemoryReserveOrWait` satisfied at once, over all requests. A miss means the request had to wait. |
| `reserve-{memtype}-wait-timeout` | Waiting requests that ran out `memory_reserve_timeout`, over the requests that had to wait. A miss means memory was released in time. |
| `reserve-{memtype}-wait-satisfied-time` | Time requests spent waiting before a reservation release satisfied them. |
| `reserve-{memtype}-wait-timeout-time` | Time requests spent waiting before the progress timeout fired. |
| `reserve-{memtype}-wait-satisfied-peak-available-bytes` | The most memory seen available while a request waited, over the requests a release went on to admit, excluding the pass that admitted them. Read against `-wait-satisfied-request-bytes`: a peak close to the request size means it was nearly satisfiable on its own, so memory freed on its behalf while it waited bought little. |
| `reserve-{memtype}-wait-satisfied-request-bytes` | What those requests asked for, over the same requests as the peak above. |
| `reserve-{memtype}-wait-timeout-peak-available-bytes` | The most memory seen available while a timed-out request waited. |
| `reserve-{memtype}-wait-timeout-request-bytes` | What the timed-out requests asked for, over the same requests as the peak above. |
| `reserve-{memtype}-wait-timeout-memory-was-available` | Timed-out requests whose requested size was available at some point while they waited, over all timed-out requests. A hit means the shortage was not absolute and the request lost the memory to another request or allocation, so forcing a spill was not the only way forward. A miss means waiting longer would have achieved nothing. |
| `reserve-{memtype}-waiting-requests` | Requests waiting concurrently. Recorded each time a request starts waiting, not sampled over time, so the maximum is exact while the mean is the queue depth seen when a request starts waiting. |
| `reserve-{memtype}-request-bytes` | Bytes requested from `reserve_or_wait()`. |
| `reserve-{memtype}-overbook-bytes` | Bytes by which `reserve_or_wait_or_overbook()` exceeded the memory limit after the timeout, counting only what each request added rather than the total outstanding deficit. Only a timed-out request overbooks, so this is also the demand behind the spilling that follows: overbooking drives availability negative and the periodic spill thread frees the room. |
| `shuffle-payload-recv` | Shuffle data received by this rank, excluding self-transfers. |
| `shuffle-payload-send` | Shuffle data sent from this rank, excluding self-transfers. |
| `allgather-payload-recv` | AllGather data received by this rank, excluding self-transfers. |
| `allgather-payload-send` | AllGather data sent from this rank, excluding self-transfers. |
| `sparsealltoall-payload-recv` | SparseAlltoall data received by this rank, excluding self-transfers. |
| `sparsealltoall-payload-send` | SparseAlltoall data sent from this rank, excluding self-transfers. |
| `allreduce-payload-recv` | AllReduce data received by this rank, excluding self-transfers. |
| `allreduce-payload-send` | AllReduce data sent from this rank, excluding self-transfers. |

Statistics are available in both C++ and [Python](#api-statistics).

## Example Output

### Text (`report()`)

```
Statistics:
 - alloc-device:                         2.79 GiB | 198.84 us | 13.72 TiB/s | avg-stream-delay 26.44 ms
 - alloc-pinned_host:                    2.79 GiB | 244.62 us | 11.15 TiB/s | avg-stream-delay 21.07 ms
 - copy-device-to-pinned_host:           2.79 GiB | 467.16 ms | 5.98 GiB/s | avg-stream-delay 21.06 ms
 - copy-pinned_host-to-device:           2.79 GiB | 481.25 ms | 5.81 GiB/s | avg-stream-delay 26.44 ms
 - event-loop-total:                     49.16 ms | avg 2.76 us
 - reserve-device-overbook-bytes:        512.00 MiB | avg 42.67 MiB
 - reserve-device-request-bytes:         2.79 GiB | avg 28.61 MiB
 - reserve-device-wait-avoided:          73/100 (hits/lookups)
 - reserve-device-wait-satisfied-time:   41.20 ms | avg 2.75 ms
 - reserve-device-wait-timeout:          12/27 (hits/lookups)
 - reserve-device-wait-timeout-time:     1.20 s | avg 100.01 ms
 - reserve-device-waiting-requests:      max 4 | avg 1.8 (27 samples)
 - shuffle-payload-recv:                 2.79 GiB | avg 28.61 MiB
 - shuffle-payload-send:                 2.79 GiB | avg 28.61 MiB
```

### JSON (`write_json()`)

JSON output contains raw numeric values for all statistics. Formatters
(which produce human-readable strings such as "1.0 KiB" or "3.5 ms" in the
text report) are not applied — values remain as plain numbers to keep the
output machine-parseable. For example, a bytes statistic that reads
`"2.9957e+09"` is roughly three billion bytes; the text report would show `"2.79 GiB"`
for the same figure.

Raw units: memory sizes are in **bytes** (float), timings are in **seconds** (float).

```json
{
  "statistics": {
    "alloc-device-bytes": {"count": 100, "value": 2.9957e+09, "max": 3.0029e+07},
    "alloc-device-stream-delay": {"count": 100, "value": 2.644, "max": 2.7e-02},
    "alloc-device-time": {"count": 100, "value": 0.00019884, "max": 2.0e-06},
    "alloc-pinned_host-bytes": {"count": 100, "value": 2.9957e+09, "max": 3.0029e+07},
    "alloc-pinned_host-stream-delay": {"count": 100, "value": 2.107, "max": 2.2e-02},
    "alloc-pinned_host-time": {"count": 100, "value": 0.00024462, "max": 2.5e-06},
    "copy-device-to-pinned_host-bytes": {"count": 100, "value": 2.9957e+09, "max": 3.0029e+07},
    "copy-device-to-pinned_host-stream-delay": {"count": 100, "value": 2.106, "max": 2.2e-02},
    "copy-device-to-pinned_host-time": {"count": 100, "value": 0.46716, "max": 5.0e-03},
    "copy-pinned_host-to-device-bytes": {"count": 100, "value": 2.9957e+09, "max": 3.0029e+07},
    "copy-pinned_host-to-device-stream-delay": {"count": 100, "value": 2.644, "max": 2.7e-02},
    "copy-pinned_host-to-device-time": {"count": 100, "value": 0.48125, "max": 5.1e-03},
    "event-loop-total": {"count": 17800, "value": 0.04916, "max": 1.8e-04},
    "reserve-device-overbook-bytes": {"count": 12, "value": 5.3687e+08, "max": 6.7109e+07},
    "reserve-device-request-bytes": {"count": 100, "value": 2.9957e+09, "max": 3.0029e+07},
    "reserve-device-wait-avoided": {"count": 100, "value": 73, "max": 1},
    "reserve-device-wait-satisfied-time": {"count": 15, "value": 0.04120, "max": 8.1e-03},
    "reserve-device-wait-timeout": {"count": 27, "value": 12, "max": 1},
    "reserve-device-wait-timeout-time": {"count": 12, "value": 1.20012, "max": 1.0012e-01},
    "reserve-device-waiting-requests": {"count": 27, "value": 48.6, "max": 4},
    "shuffle-payload-recv": {"count": 100, "value": 2.9957e+09, "max": 3.0029e+07},
    "shuffle-payload-send": {"count": 100, "value": 2.9957e+09, "max": 3.0029e+07}
  },
}
```
