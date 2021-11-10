from __future__ import annotations

import asyncio
import string
from collections import Counter
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from distributed.utils_test import gen_cluster

from ..shuffle_extension import (
    NewShuffleMetadata,
    ShuffleId,
    ShuffleMetadata,
    ShuffleWorkerExtension,
)

if TYPE_CHECKING:
    from distributed import Scheduler, Worker


@pytest.mark.parametrize("npartitions", [1, 2, 3, 5])
@pytest.mark.parametrize("n_workers", [1, 2, 3, 5])
def test_worker_for_distribution(npartitions: int, n_workers: int):
    "Test that `worker_for` distributes evenly"
    metadata = ShuffleMetadata(
        ShuffleId("foo"),
        pd.DataFrame({"A": []}),
        "A",
        npartitions,
        list(string.ascii_lowercase[:n_workers]),
    )

    assignments = [metadata.worker_for(i) for i in range(metadata.npartitions)]
    counter = Counter(assignments)
    assert len(counter) == min(npartitions, n_workers)
    # All workers should be assigned the same number of partitions, or if
    # there's an odd number, one worker will be assigned one extra partition.
    counts = set(counter.values())
    assert len(counts) <= 2
    if len(counts) == 2:
        lo, hi = sorted(counts)
        assert lo == hi - 1

    with pytest.raises(IndexError, match="does not exist"):
        metadata.worker_for(npartitions)


@gen_cluster([("", 1)])
async def test_installation(s: Scheduler, worker: Worker):
    ext = worker.extensions["shuffle"]
    assert isinstance(ext, ShuffleWorkerExtension)
    assert worker.handlers["shuffle_receive"] == ext.shuffle_receive
    assert worker.handlers["shuffle_init"] == ext.shuffle_init

    with pytest.raises(RuntimeError, match="from a different shuffle extension"):
        ShuffleWorkerExtension(worker)

    # Nothing was changed by double-installation
    assert worker.extensions["shuffle"] is ext
    assert worker.handlers["shuffle_receive"] == ext.shuffle_receive
    assert worker.handlers["shuffle_init"] == ext.shuffle_init


@gen_cluster([("", 1)])
async def test_init(s: Scheduler, worker: Worker):
    ext: ShuffleWorkerExtension = worker.extensions["shuffle"]
    assert not ext.shuffles
    metadata = ShuffleMetadata(
        ShuffleId("foo"),
        pd.DataFrame({"A": []}),
        "A",
        5,
        [worker.address],
    )

    ext.shuffle_init(None, metadata)
    assert list(ext.shuffles) == [metadata.id]

    with pytest.raises(ValueError, match="already registered"):
        ext.shuffle_init(None, metadata)

    assert list(ext.shuffles) == [metadata.id]


@gen_cluster([("", 1)] * 4)
async def test_create(s: Scheduler, *workers: Worker):
    exts: list[ShuffleWorkerExtension] = [w.extensions["shuffle"] for w in workers]

    new_metadata = NewShuffleMetadata(
        ShuffleId("foo"),
        pd.DataFrame({"A": []}),
        "A",
        5,
    )

    await exts[0]._create_shuffle(new_metadata)
    for ext in exts:
        assert len(ext.shuffles) == 1
        shuffle = ext.shuffles[new_metadata.id]
        assert shuffle.metadata.workers == [w.address for w in workers]
        # ^ TODO is this order guaranteed?

    with pytest.raises(ValueError, match="already registered"):
        await exts[0]._create_shuffle(new_metadata)

    # TODO (resilience stage) what happens if some workers already have
    # the ID registered, but others don't?


@gen_cluster([("", 1)] * 4)
async def test_add_partition(s: Scheduler, *workers: Worker):
    exts: dict[str, ShuffleWorkerExtension] = {
        w.address: w.extensions["shuffle"] for w in workers
    }

    new_metadata = NewShuffleMetadata(
        ShuffleId("foo"),
        pd.DataFrame({"A": [], "partition": []}),
        "partition",
        8,
    )

    ext = next(iter(exts.values()))
    await ext._create_shuffle(new_metadata)
    partition = pd.DataFrame(
        {
            "A": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "partition": [0, 1, 2, 3, 4, 5, 6, 7],
        }
    )
    await ext._add_partition(partition, new_metadata.id)

    with pytest.raises(ValueError, match="not registered"):
        await ext._add_partition(partition, ShuffleId("bar"))

    metadata = ext.shuffles[new_metadata.id].metadata
    for i, data in partition.groupby(new_metadata.column):
        addr = metadata.worker_for(int(i))
        ext = exts[addr]
        received = ext.shuffles[metadata.id].output_partitions[int(i)]
        assert len(received) == 1
        assert_frame_equal(data, received[0])

    # TODO (resilience stage) test failed sends


@gen_cluster([("", 1)] * 4)
async def test_get_partition(s: Scheduler, *workers: Worker):
    exts: dict[str, ShuffleWorkerExtension] = {
        w.address: w.extensions["shuffle"] for w in workers
    }

    new_metadata = NewShuffleMetadata(
        ShuffleId("foo"),
        pd.DataFrame({"A": [], "partition": []}),
        "partition",
        8,
    )

    ext = next(iter(exts.values()))
    await ext._create_shuffle(new_metadata)
    metadata = ext.shuffles[new_metadata.id].metadata
    p1 = pd.DataFrame(
        {
            "A": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "partition": [0, 1, 2, 3, 4, 5, 6, 6],
        }
    )
    p2 = pd.DataFrame(
        {
            "A": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "partition": [0, 1, 2, 3, 0, 0, 2, 3],
        }
    )
    await asyncio.gather(
        ext._add_partition(p1, new_metadata.id), ext._add_partition(p2, new_metadata.id)
    )

    full = pd.concat([p1, p2])
    expected_groups = full.groupby("partition")
    for output_i in range(metadata.npartitions):
        addr = metadata.worker_for(output_i)
        ext = exts[addr]
        result = ext.get_output_partition(metadata.id, output_i)
        try:
            expected = expected_groups.get_group(output_i)
        except KeyError:
            expected = metadata.empty
        assert_frame_equal(expected, result)

    with pytest.raises(ValueError, match="not registered"):
        ext.get_output_partition(ShuffleId("bar"), 0)

    with pytest.raises(AssertionError, match="belongs on"):
        ext.get_output_partition(metadata.id, 0)
