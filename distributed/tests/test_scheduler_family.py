from __future__ import annotations

import operator

from tlz import partition, partition_all

import dask

from distributed.families import group_by_family
from distributed.scheduler import TaskState
from distributed.utils_test import async_wait_for, gen_cluster, slowidentity, slowinc

ident = dask.delayed(slowidentity, pure=True)
inc = dask.delayed(slowinc, pure=True)
add = dask.delayed(operator.add, pure=True)
dsum = dask.delayed(sum, pure=True)


async def submit_delayed(client, scheduler, x):
    "Submit a delayed object or list of them; wait until tasks are processed on scheduler"
    # dask.visualize(x, optimize_graph=True, collapse_outputs=True)
    fs = client.compute(x)
    await async_wait_for(lambda: scheduler.tasks, 5)
    try:
        key = fs.key
    except AttributeError:
        key = fs[0].key
    await async_wait_for(lambda: scheduler.tasks[key].state != "released", 5)
    return fs


_ = TaskState("_", None, "released")


def assert_family(
    family: tuple[list[TaskState], bool], pattern: list[TaskState], *, proper: bool
) -> None:
    fam, is_proper = family
    assert is_proper == proper
    assert len(fam) == len(pattern)
    for f, p in zip(fam, pattern):
        if p is not _:
            assert f is p


@gen_cluster(nthreads=[], client=True)
async def test_family_two_step_reduction(c, s):
    r"""
          z         z         z
        /  |      /  |      /  |
      x    |    x    |    x    |
     / \   |   / \   |   / \   |
    a   b  c  a   b  c  a   b  c
    """
    ax = [dask.delayed(i, name=f"a-{i}") for i in range(3)]
    bx = [dask.delayed(i, name=f"b-{i}") for i in range(3)]
    cx = [dask.delayed(i, name=f"c-{i}") for i in range(3)]

    zs = [
        add(add(a, b, dask_key_name=f"x-{i}"), c, dask_key_name=f"z-{i}")
        for i, (a, b, c) in enumerate(zip(ax, bx, cx))
    ]

    _ = await submit_delayed(c, s, zs)

    fams = list(group_by_family(s.tasks.values()))
    assert len(fams) == 6

    for i, [f0, f1] in enumerate(partition(2, fams)):
        a = s.tasks[f"a-{i}"]
        b = s.tasks[f"b-{i}"]
        x = s.tasks[f"x-{i}"]
        c = s.tasks[f"c-{i}"]
        z = s.tasks[f"z-{i}"]

        assert_family(f0, [a, b, x], proper=True)
        assert_family(f1, [c, z], proper=False)


@gen_cluster(nthreads=[], client=True)
async def test_family_two_step_reduction_linear_chains(c, s):
    r"""
                          final
                     /              \               \
                    /                \               \
            --------------
            |    s2      |            s2              s2
            | /      \   |          /     \         /     \
            |/______  \  |         |       |       |       |
     -------------  |  | |         |       |       |       |
     |     s1    |  |  | |         s1      |       s1      |
     | /   |  \  |  |  | |     /   |  \    |   /   |  \    |
     | x   |   | |  |  | |     x   |   |   |   x   |   |   |
     | |   |   | |  |  | |     |   |   |   |   |   |   |   |
     | x   |   x |  |  x |     x   |   x   x   x   |   x   x
     | |   |   | |  |  | |     |   |   |   |   |   |   |   |
     | a   b   c |  |  d |     a   b   c   d   a   b   c   d
     -------------  ------    /   /   /  /   /   /    /   /
       \   \   \   \   \  \  /   /   /  /   /   /    /   /
                            r
    """
    root = dask.delayed(0, name="root")
    ax = [
        ident(ident(inc(root, dask_key_name=f"a-{i}"))) for i in range(3)  # 2-chain(z)
    ]
    bx = [inc(root, dask_key_name=f"b-{i}") for i in range(3)]  # 0-chain
    cx = [ident(inc(root, dask_key_name=f"c-{i}")) for i in range(3)]  # 1-chain
    s1x = [
        dsum([a, b, c], dask_key_name=f"s1-{i}")
        for i, (a, b, c) in enumerate(zip(ax, bx, cx))
    ]

    dx = [ident(inc(root, dask_key_name=f"d-{i}")) for i in range(3)]  # 1-chain
    s2x = [
        add(s1, d, dask_key_name=f"s2-{i}") for i, (s1, d) in enumerate(zip(s1x, dx))
    ]

    final = dsum(s2x, dask_key_name="final")

    _ = await submit_delayed(c, s, final)

    fams = list(group_by_family(s.tasks.values()))
    assert len(fams) == 8

    st = s.tasks
    root = st["root"]
    final = st["final"]

    assert_family(fams[0], [root], proper=False)

    assert_family(
        fams[1], [st["a-0"], _, _, st["b-0"], st["c-0"], _, st["s1-0"]], proper=True
    )
    assert_family(fams[2], [st["d-0"], _, st["s2-0"]], proper=True)

    assert_family(
        fams[3], [st["a-1"], _, _, st["b-1"], st["c-1"], _, st["s1-1"]], proper=True
    )
    assert_family(fams[4], [st["d-1"], _, st["s2-1"]], proper=True)

    assert_family(
        fams[5], [st["a-2"], _, _, st["b-2"], st["c-2"], _, st["s1-2"]], proper=True
    )
    assert_family(fams[6], [st["d-2"], _, st["s2-2"]], proper=True)

    assert_family(fams[7], [final], proper=False)


@gen_cluster(nthreads=[], client=True)
async def test_family_linear_chains_plus_widely_shared(c, s):
    r"""
      s     s    s
     /|\   /|\   /\
    a a a a a a a a
    |\|\|\|\|/|/|/|
    | | | | s | | |
    r r r r r r r r
    """
    shared = dask.delayed(0, name="shared")
    roots = [dask.delayed(i, name=f"r-{i}") for i in range(8)]
    ax = [add(r, shared, dask_key_name=f"a-{i}") for i, r in enumerate(roots)]
    sx = [
        dsum(axs, dask_key_name=f"s-{i}") for i, axs in enumerate(partition_all(3, ax))
    ]

    _ = await submit_delayed(c, s, sx)

    fams = list(group_by_family(s.tasks.values()))
    assert len(fams) == 4

    st = s.tasks

    assert_family(fams[0], [st["shared"]], proper=False)
    assert_family(
        fams[1],
        [st["r-0"], st["a-0"], st["r-1"], st["a-1"], st["r-2"], st["a-2"], st["s-0"]],
        proper=True,
    )
    assert_family(
        fams[2],
        [st["r-3"], st["a-3"], st["r-4"], st["a-4"], st["r-5"], st["a-5"], st["s-1"]],
        proper=True,
    )
    assert_family(
        fams[3],
        [st["r-6"], st["a-6"], st["r-7"], st["a-7"], st["s-3"]],
        proper=True,
    )


# @gen_cluster(nthreads=[], client=True)
# async def test_family_triangle(c, s):
#     r"""
#       z
#      /|
#     y |
#     \ |
#       x
#     """
#     x = dask.delayed(0, name="x")
#     y = inc(x, dask_key_name="y")
#     z = add(x, y, dask_key_name="z")

#     _ = await submit_delayed(c, s, z)

#     x = s.tasks["x"]
#     y = s.tasks["y"]
#     z = s.tasks["z"]

#     fam = family(x, 1000, 1000)
#     assert fam
#     sibs, downstream = fam
#     assert sibs == set()
#     assert downstream == {z}  # `y` is just a linear chain, not downstream

#     fam = family(y, 1000, 1000)
#     assert fam
#     sibs, downstream = fam
#     assert sibs == {x}
#     assert downstream == {z}


# @gen_cluster(nthreads=[], client=True)
# async def test_family_wide_gather_downstream(c, s):
#     r"""
#             s
#      / / / /|\ \ \
#     i i i i i i i i
#     | | | | | | | |
#     r r r r r r r r
#     """
#     roots = [dask.delayed(i, name=f"r-{i}") for i in range(8)]
#     incs = [inc(r, dask_key_name=f"i-{i}") for i, r in enumerate(roots)]
#     sum = dsum(incs, dask_key_name="sum")

#     _ = await submit_delayed(c, s, sum)

#     rts = [s.tasks[r.key] for r in roots]
#     sts = s.tasks["sum"]

#     fam = family(rts[0], 4, 1000)
#     assert fam
#     sibs, downstream = fam
#     assert sibs == set()
#     assert downstream == set()  # `sum` not downstream because it's too large

#     fam = family(rts[0], 1000, 1000)
#     assert fam
#     sibs, downstream = fam
#     assert sibs == set(rts[1:])
#     assert downstream == {sts}


# # TODO test family commutativity. Given any node X in any graph, calculate `family(X)`.
# # For each sibling S, `family(S)` should give the same family, regardless of the
# # starting node.
# # EXECPT THIS ISN'T TRUE


# @gen_cluster(nthreads=[], client=True)
# async def test_family_non_commutative(c, s):
#     roots = [dask.delayed(i, name=f"r-{i}") for i in range(16)]
#     aggs = [dsum(rs) for rs in partition(4, roots)]
#     extra = dsum([roots[::4]], dask_key_name="extra")

#     _ = await submit_delayed(c, s, aggs + [extra])

#     rts = [s.tasks[r.key] for r in roots]
#     ats = [s.tasks[a.key] for a in aggs]
#     ets = s.tasks["extra"]

#     fam = family(rts[0], 1000, 1000)
#     assert fam
#     sibs, downstream = fam
#     assert sibs == set(rts[1:4]) | {rts[4], rts[8], rts[12]}
#     assert downstream == {ats[0], ets}

#     fam = family(rts[1], 1000, 1000)
#     assert fam
#     sibs, downstream = fam
#     assert sibs == {rts[0], rts[2], rts[3]}
#     assert downstream == {ats[0]}


# @gen_cluster(nthreads=[], client=True)
# async def test_reuse(c, s):
#     r"""
#     a + (a + 1).mean()
#     """
#     roots = [dask.delayed(i, name=f"r-{i}") for i in range(6)]
#     incs = [inc(r, name=f"i-{i}") for i, r in enumerate(roots)]
#     mean = dsum([dsum(incs[:3]), dsum(incs[3:])])
#     deltas = [add(r, mean, dask_key_name=f"d-{i}") for i, r in enumerate(roots)]

#     _ = await submit_delayed(c, s, deltas)


# @gen_cluster(nthreads=[], client=True)
# async def test_common_with_trees(c, s):
#     r"""
#      x       x        x      x
#      /|\    /|\      /|\    /|\
#     a | b  c | d    e | f  g | h
#       |      |        |      |
#        ---------- c ----------
#     """
#     pass


# @gen_cluster(nthreads=[], client=True)
# async def test_zigzag(c, s):
#     r"""
#     x  x  x  x
#     | /| /| /|
#     r  r  r  r
#     """
#     roots = [dask.delayed(i, name=f"r-{i}") for i in range(4)]
#     others = [
#         inc(roots[0]),
#         add(roots[0], roots[1]),
#         add(roots[1], roots[2]),
#         add(roots[2], roots[3]),
#     ]

#     _ = await submit_delayed(c, s, others)


# @gen_cluster(nthreads=[], client=True)
# async def test_overlap(c, s):
#     r"""
#     x  x  x  x
#     |\/|\/|\/|
#     |/\|/\|/\|
#     r  r  r  r
#     """
#     roots = [dask.delayed(i, name=f"r-{i}") for i in range(4)]
#     others = [
#         add(roots[0], roots[1]),
#         dsum(roots[0], roots[1], roots[2]),
#         dsum(roots[1], roots[2], roots[3]),
#         add(roots[2], roots[3]),
#     ]

#     _ = await submit_delayed(c, s, others)
