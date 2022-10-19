from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Collection, Iterator

if TYPE_CHECKING:
    from distributed.scheduler import TaskState

prio_getter = operator.attrgetter("priority")


def group_by_family(
    tasks: Collection[TaskState],
) -> Iterator[tuple[list[TaskState], bool]]:
    tasks = sorted(tasks, key=prio_getter)
    family_start_i = 0
    while family_start_i < len(tasks):
        ts = tasks[family_start_i]
        prev_prio = start_prio = ts.priority[-1]
        max_prio = prev_prio + 1
        proper_family: bool = False
        while ts.dependents:
            ts = min(ts.dependents, key=prio_getter)
            max_prio = ts.priority[-1]
            if max_prio == prev_prio + 1:
                # walk linear chains of consecutive priority
                # TODO if this chain is all linear, try again from the start with the next-smallest dependent
                prev_prio = max_prio
            else:
                # non-consecutive priority jump. this is our max node.
                prev_prio = max_prio
                proper_family = True
                assert max_prio > start_prio + 1, (max_prio, start_prio, ts)
                break

        # all tasks from the start to the max (inclusive) belong to the family.
        next_start_i = family_start_i + (max_prio - start_prio) + 1
        yield tasks[family_start_i:next_start_i], proper_family
        family_start_i = next_start_i
