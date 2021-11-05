# Peer-to-peer DataFrame shuffling

This is a proposal for the high-level design of an extension built into distributed for shuffling very large DataFrames reliably and performantly. It does so by transferring data between workers out-of-band (not managed by the scheduler) using stateful worker extensions. This significantly reduces the size of the graph and eliminates the scheduler as a bottleneck, compared to the current task-based shuffle.

This work builds off the proof-of-concept in https://github.com/dask/dask/pull/8223.

## Motivation

Shuffles are an integral part of most DataFrame workloads, as part of a `merge`, `set_index`, or `groupby().apply()`. Shuffling is a poor fit for centralized graph-based scheduling, since the graph is all-to-all (O(N²) in size), yet the logic is so simple, it gets little benefit from centralized coordination, while suffering significant overhead from it. With task-based shuffles, the amount of data we can shuffle effectively (before workers run out of memory, or the scheduler crashes, or both) is severely limited. By allowing workers to autonomously exchange data with their peers, and manage disk and memory usage in a more fine-grained way, that limit becomes significantly higher.

See https://coiled.io/blog/better-shuffling-in-dask-a-proof-of-concept/ for more background.

## Goals and non-goals

End goals:
* Can reliably shuffle orders-of-magnitude larger datasets (in total size and number of partitions) than the current task-based shuffle
* Maintainable, thoroughly tested code using (or adding) public APIs
* Can shuffle larger-than-memory datasets by spilling to disk
* Constant, predictable memory footprint per worker, which scales linearly with partition size, not total number of partitions
* Just works, without users needing to tune parameters (buffer sizes, etc.)
* Graceful restarting when possible, and quick failure when not
* All state is cleaned up on success, failure, or cancellation
* Shuffle performance is IO-bound (network, disk)

Non-goals:
* Utilize new workers that enter the cluster midway though a shuffle
* Resilience via data duplication (a shuffle can continue through losing some number of workers)
* Worker loss only requires partial re-transfer of data

## Plan

The implementation will be completed in multiple stages (order TBD after #1):
1. Establish the patterns for how this out-of-band system will interact with distributed, in the simplest possible implementation with no optimizations.
1. Retries (shuffle restarts automatically if a worker leaves)
1. Improve performance with concurrency if possible
1. Spill-to-disk
1. Backpressure
1. Performance

## Design

Here we'll discuss the highest-level architectural components of the shuffling system.

_Note: unlike the POC PR, we propose keeping this code in `dask/distributed`, not `dask/dask`. The implementation will be somewhat dependent on worker internals, so internal changes in `distributed` are far more likely to break things than changes in `dask`. We'd like tests to fail (and fixes to happen) in the same repo as the changes that break things. Plus, the code is very `distributed`-specific anyway._

### Task graph

The whole point of a peer-to-peer shuffle using a separate worker extension is to not have to capture the full operation in a dask graph.
Therefore, the primary purpose of the graph is to mediate between dask-managed data and out-of-band processing:
- Hand off dask-managed data to be processed by the extension
- Bring data produced out-of-band back into dask
- Clean up when keys depending on the shuffle are cancelled

The graph also has a secondary synchronization benefit, letting us bypass some difficult distributed problems (exactly-once initialization and cleanup tasks, determining when all peer-to-peer transfers are complete) by leaning on the scheduler.

![diagram of graph](graph.png)

### `ShuffleExtension`

Problems this solves:
* Holding per-worker out-of-band state for an in-progress shuffle
* Adding new handlers in an organized way for workers to transfer shuffle data
* Doing the above cleanly with multiple concurrent shuffles (`merge`)
* Coordinating multiple concurrent shuffles which may need to share limited resources (memory, threads, etc.)
* Getting metrics back to the scheduler/dashboard, like managed memory & bytes spilled to disk (eventually)

The `ShuffleExtension` will be built into distributed and added to workers automatically (like the `PubSubWorkerExtension`). It'll add a route to the worker; something like:

```python
def shuffle_receive(comm, shuffle_id: str, data: DataFrame) -> None:
    """
    Receive an incoming shard of data from a peer worker.
    Using an unknown ``shuffle_id`` will first initialize any state needed for that new shuffle.
    """
    ...
```

The `ShuffleExtension` will hold all per-shuffle state and buffers specific to that worker. For example, things like (will vary depending on the stage of the implementation):
- a buffer of received shards
- a buffer of outgoing shards (waiting to accumulate enough to be worth sending to a worker)
- a datastore that transparently spills received shards to disk
- locks or synchronization mechanisms

The `ShuffleExtension` will also hold any global state shared by all shuffles, such as worker threads or coroutines.

Most of the implementation-specific logic will happen in the `ShuffleExtension`. As we improve performance, add concurrency, etc., this code will change the most (though the interface will likely not).

The `transfer` tasks will pass their input partitions into the `ShuffleExtension`, blocking the task until the extension is ready to receive another input. Internally, the extension will do whatever it needs to do to transfer the data, using worker comms to call the `shuffle_receive` RPC on peer workers. Simultaneously, it'll handle any incoming data from other workers.

### Retries and cancellation

Problems this solves:
* Causing all tasks in the shuffle to rerun when a worker leaves
* Cleaning up out-of-band state when a user cancels a shuffle, or it errors

Because most the tasks in the shuffle graph are impure and run for their side effects, restarting an in-progress shuffle requires rerunning _every_ task involved, even ones that appear to have successfully transitioned to `memory` and whose "results" are stored on non-yet-dead workers.

Additionally, cleanly stopping a running shuffle takes more than just releasing the shuffle tasks from memory: since there's out-of-band processing going on, the `ShuffleExtension` has to be informed in some way that it needs to stop doing whatever it's doing in the background, and clear out its buffers. Also, executing tasks may be blocking on the `ShuffleExtension` doing something; without a way to tell the extension to shut down, those tasks might block forever, deadlocking the cluster.

Therefore, we propose adding a `RerunGroup` (`ImpureGroup`? `CoExecutionGroup`? `RestartGroup`? `OutOfBandGroup`? name TBD) structure to the scheduler which intertwines the fates of all tasks within it: if any one task is to be rescheduled (due to its worker leaving), all tasks are restarted; if any one is to be prematurely released (due to cancellation), all are released.

Membership in a `RerunGroup` is implemented via task annotations, where each task gives the name of the `RerunGroup` it belongs to. A task can belong to at most one `RerunGroup`. TBD if we will enforce any structural restrictions on `RerunGroup`s to prevent odd/invalid states from emerging—we probably should, such as not allowing disjoint parts of the graph to be in the same `RerunGroup`, etc.

Additionally, the scheduler informs workers whenever a `RerunGroup` is restarted or cancelled. Workers will have a way to pass this information on to any interested out-of-band operations. This could be something like:
- Workers have a named `threading.Event` for each `RerunGroup` that any of their current tasks belong to. When the scheduler tells workers about a restart/cancellation, they `set()` the corresponding event so that some background thread can respond accordingly.
- A `register_cancellation_handler(rerun_group: str, async_callback: Callable)` method on workers that registers an async function to be run when that group is cancelled/restarted. A potential upside (and potential deadlock) is that the scheduler's `cancel_rerun_group` RPC to workers could block on this callback completing, meaning the scheduler wouldn't treat the `RerunGroup` as successfully cancelled until every callback on every worker succeeded. That could give us some nice synchronization guarantees (which we may or many not actually need) ensuring a shuffle doesn't try to restart while it's also trying to shut down.

### Peer discovery and initialization

Problems this solves:
* Workers need to all have the same list of peers participating in the shuffle (otherwise data could end up in two different places!)
* Scheduler needs to be told where to run the `unpack` tasks which bring data back in-band

We'll run a single `shuffle-setup` task before all the transfers to do some initialization.

First, it will ask the scheduler for the addresses of all workers that should participate in the shuffle (taking into account worker or resource restrictions). How this will be implemented is TBD.

Next, it will set worker restrictions on the `unpack` tasks, so each task will run on the worker that will receive that output partition. (This is computed just by binning the number of output partitions into the number of workers.) Note that we could also do this step in the barrier task; just seems nicer and potentially a tiny bit less overhead to do all scheduler comms in one place.

It'll return the list of worker addresses. This will be input to all the `transfer` tasks, which use the same binning logic to decide where to send a given row of each input partition.

Since this `shuffle-setup` task will be part of the `RerunGroup`, every time the shuffle is restarted, we'll recompute these peer addresses (accounting for any lost or gained workers) and reset any now-invalid worker restrictions on the `unpack` tasks (preventing deadlocks from waiting to schedule a task on a worker that doesn't exist).

Also possibly (TBD) `shuffle-setup` could call an RPC on all the other workers informing them that a shuffle is about to happen. This is most likely unnecessary, but in case there are some resources that need to be initialized before any data moves around, this would give us an easy way to do it.

### Spilling to disk

Problems this solves:
* Shuffling larger-than-memory datasets requires writing some of the data to disk
* Makes the worker memory requirement depend only on the size of each partition of the dataset, not the total size of the dataset
* Applies backpressure to senders when spill-to-disk can't keep up with incoming data (see backpressure section)
* Does all the above without blocking the event loop

After the initial shuffling framework is built, we'll want to support larger-than-memory datasets. To do so, workers will spill some of the data they receive from peers to disk.

For testability, we'll implement a standalone mechanism for this. Most likely, it'll be a key-value buffer that supports appends to keys, like:

```python
async? def append(**values: pd.DataFrame) -> None:
    "Append key-value pairs to the dataset"
    ...

async? def get(key: str) -> pd.DataFrame:
    "Get all data written to a key, concatenated"
    ...

async? def remove(key: str) -> None:
    "Remove all data for a key"
    ...

async? def clear() -> None:
    "Remove all data"
    ...

async? def set_available_memory(bytes: int) -> None:
    """
    Change the memory threshold for spilling to disk.

    If current memory use is greater than the new threshold, this blocks until excess memory is spilled to disk.
    """
    # NOTE: necessary for concurrent shuffles: if a shuffle is already running and another wants to start,
    # the first shuffle needs to spill excess data to disk so the two can share memory evenly.
    ...
```

Note that the interface, purpose, and functionality are essentially the same as [partd](https://github.com/dask/partd). We may just use partd here, or modify partd such that it fits our needs, rather than implementing something new.

Most likely, the writes to this buffer will be very small. If so, the system must handle small writes performantly.

If a shuffle is aborted (error, cancellation, or retry), the buffer will be told to clean up all its data (both on disk and in memory).

The buffer will have some way of applying backpressure on `append`s, so if it can't keep up with all the data it's being asked to write to disk, it can inform callers. This will be bubbled up to tell peer workers to slow down data sending (and possibly production) so this worker doesn't run out of memory.

The buffer will handle serialization, transformation, and compression of the input data. Therefore, if it ends up being an async interface and this is event-loop-blocking work, it'll have a way to offload this work to threads.

The implementation will depend a lot on our network serialization format, and our concurrency model. For example, if we were sending Arrow [`RecordBatches`](https://arrow.apache.org/docs/python/data.html#record-batches) over the network, already grouped by output partition, we could just write those serialized bytes straight to disk from an async comm handler (using [aiofiles](https://github.com/Tinche/aiofiles) or the like). On the other hand, if we're sending DataFrames over comms and deserializing them immediately on the receiving side (as in the proof-of-concept PR), then a lot more work will be involved: re-serializing them, potentially even accumulating an in-memory buffer of DataFrame shards to be re-concatenated, serialized, and compressed if the serialization scheme is inefficient for lots of small DataFrames.

For this reason, we won't make any more guesses about how this is actually going to work.

Additionally, it would be nice if the buffer could integrate with the worker's reporting of managed memory and data spilled to disk. Obviously, we'd have to create the worker and scheduler interfaces for this first, but the buffer should at least track these numbers so we can eventually report them.

### Backpressure

Problems this solves:
* If workers can't write to disk fast enough, newly-received shards could pile up in memory and possibly kill them

Backpressure will let workers that have too much data in memory tell their peers to stop sending them more data until they've caught up and spilled the excess memory to disk.

There are numerous ways to implement backpressure; I'm not sure yet how we'll do it for this system. But ultimately, it'll result in RPC calls to `shuffle_receive` informing the caller to slow down or stop in some way (by returning a value, or perhaps even just by blocking). In turn, this will cause `transfer` tasks on the callers to block when passing data into the `ShuffleExtension` until the backpressure has been resolved, preventing future `transfer` tasks from running.



#### Data submission protocol

Relevant requirements: 4., 6., 9. 10.,

The data producer will perform a groupby [M, W] where M is the number of output partitions and W is the number of participating workers

If the producer already splits by output partition we will not need to regroup on the receiver side. Concat could happen in the unpack method resolving us from any GIL/Thread problems.

```python
    # Does not need to be a *dict* but we'll need some kind of mapping, either by endoding or by data structure.
    {
        "W1": {
            # Every element is a pandas dataframe / arrow table
            "M1": ["W1-M1-N1", "W1-M1-N2", "W1-M1-N3"],
            "M2": ["W1-M2-N1", "W1-M2-N2", "W1-M2-N3"],
        },
        "W1": {
            "M1": ["W1-M1-N1", "W1-M1-N2", "W1-M1-N3"],
            "M2": ["W1-M2-N1", "W1-M2-N2", "W1-M2-N3"],
        },
        ...
    }
```

Receiving end

```python
# This is disk stuff!
class ShardCollection:

    def __init__(self):
        self._buffer = []  # or memory view, whatever
        self.thread = Thread() # likely need a thread for async

    async def append(self, shard):
        buffer.append(shard)
        if buffer.full():
            await self.dump_buffer_to_disk()

    async def put(self):
        pass

    async def get(self):
        load_data_to_buffer()
        return self._buffer

    def __del__(self):
        delete_disk()
        self._buffer = None


Worker.data[key] = ShardCollection()
```

#### Asnychronous operations

Due to 1., 2.

#### Backpressure

The prototype implementation in #XXX is not introducing data backpressure to deal with 11.

## Appendix

### Data-transfer diagrams

Some pretty pictures of the principle behind the data transfer:

![diagram of transfers](transfer.png)