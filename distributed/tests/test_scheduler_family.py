from __future__ import annotations

import operator

from tlz import partition, partition_all

import dask

from distributed.scheduler import family
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


@gen_cluster(nthreads=[], client=True)
async def test_family(c, s):
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

    zs = [(a + b) + c for a, b, c in zip(ax, bx, cx)]

    _ = await submit_delayed(c, s, zs)

    a1 = s.tasks["a-1"]
    b1 = s.tasks["b-1"]
    c1 = s.tasks["c-1"]

    fam = family(a1, 1000, 1000)
    assert fam
    sibs, downstream = fam
    assert sibs == {b1}
    assert len(downstream) == 1
    add_1_1 = next(iter(downstream))

    fam = family(c1, 1000, 1000)
    assert fam
    sibs, downstream = fam
    assert sibs == {add_1_1}
    assert len(downstream) == 1  # don't know keys
    add_1_2 = next(iter(downstream))

    fam = family(add_1_1, 1000, 1000)
    assert fam
    sibs, downstream = fam
    assert sibs == {c1}
    assert downstream == {add_1_2}

    assert family(add_1_2, 1000, 1000) is None


@gen_cluster(nthreads=[], client=True)
async def test_family_linear_chains(c, s):
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

    root = s.tasks["root"]
    a1 = s.tasks["a-1"]
    b1 = s.tasks["b-1"]
    c1 = s.tasks["c-1"]
    d1 = s.tasks["d-1"]
    s1_1 = s.tasks["s1-1"]
    s2_1 = s.tasks["s2-1"]
    final = s.tasks["final"]

    await async_wait_for(lambda: final.state == "waiting", 5)

    # `a` traverses chains up and down to find `b` and `c`
    # Does *not* include `d`: `d` is not required to compute `s1`
    fam = family(a1, 1000, 4)
    assert fam
    sibs, downstream = fam
    assert sibs == {b1, c1}
    assert downstream == {s1_1}

    # `d` traverses chains up to find `s2`
    # does not traverse down past `s2`
    fam = family(d1, 1000, 4)
    assert fam
    sibs, downstream = fam
    assert sibs == {s1_1}
    assert downstream == {s2_1}

    # Don't traverse a linear chain past self
    mid_chain = next(iter(a1.dependents))
    fam = family(mid_chain, 1000, 4)
    assert fam
    sibs, downstream = fam
    assert sibs == {b1, c1}
    assert downstream == {s1_1}

    # `root` has no family with small widely-shared cutoff
    assert family(root, 1000, 4) is None

    # With large cutoff, `root` has no siblings.
    # But the `s1` and `s2` tasks are all considered downstream, if you
    # collapse the linear chains (which include `a`, `b`, `c`, `d`).

    # Note that `s1`s could be considered both siblings and downstreams
    # (siblings, since they need to be in memory along with root to compute `s2`).
    # But tasks that meet this criteria are explicitly labeled as only downstream.
    fam = family(root, 1000, 1000)
    assert fam
    sibs, downstream = fam
    assert sibs == set()
    assert {ts.key for ts in downstream} == {
        "s1-0",
        "s1-1",
        "s1-2",
        "s2-0",
        "s2-1",
        "s2-2",
    }


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

    r0 = s.tasks["r-0"]
    r1 = s.tasks["r-1"]
    r2 = s.tasks["r-2"]
    s0 = s.tasks["s-0"]

    fam = family(r0, 1000, 4)
    assert fam
    sibs, downstream = fam
    assert sibs == {r1, r2}
    assert downstream == {s0}


@gen_cluster(nthreads=[], client=True)
async def test_family_triangle(c, s):
    r"""
      z
     /|
    y |
    \ |
      x
    """
    x = dask.delayed(0, name="x")
    y = inc(x, dask_key_name="y")
    z = add(x, y, dask_key_name="z")

    _ = await submit_delayed(c, s, z)

    x = s.tasks["x"]
    y = s.tasks["y"]
    z = s.tasks["z"]

    fam = family(x, 1000, 1000)
    assert fam
    sibs, downstream = fam
    assert sibs == set()
    assert downstream == {z}  # `y` is just a linear chain, not downstream

    fam = family(y, 1000, 1000)
    assert fam
    sibs, downstream = fam
    assert sibs == {x}
    assert downstream == {z}


@gen_cluster(nthreads=[], client=True)
async def test_family_wide_gather_downstream(c, s):
    r"""
            s
     / / / /|\ \ \
    i i i i i i i i
    | | | | | | | |
    r r r r r r r r
    """
    roots = [dask.delayed(i, name=f"r-{i}") for i in range(8)]
    incs = [inc(r, dask_key_name=f"i-{i}") for i, r in enumerate(roots)]
    sum = dsum(incs, dask_key_name="sum")

    _ = await submit_delayed(c, s, sum)

    rts = [s.tasks[r.key] for r in roots]
    sts = s.tasks["sum"]

    fam = family(rts[0], 4, 1000)
    assert fam
    sibs, downstream = fam
    assert sibs == set()
    assert downstream == set()  # `sum` not downstream because it's too large

    fam = family(rts[0], 1000, 1000)
    assert fam
    sibs, downstream = fam
    assert sibs == set(rts[1:])
    assert downstream == {sts}


# TODO test family commutativity. Given any node X in any graph, calculate `family(X)`.
# For each sibling S, `family(S)` should give the same family, regardless of the
# starting node.
# EXECPT THIS ISN'T TRUE


@gen_cluster(nthreads=[], client=True)
async def test_family_non_commutative(c, s):
    roots = [dask.delayed(i, name=f"r-{i}") for i in range(16)]
    aggs = [dsum(rs) for rs in partition(4, roots)]
    extra = dsum([roots[::4]], dask_key_name="extra")

    _ = await submit_delayed(c, s, aggs + [extra])

    rts = [s.tasks[r.key] for r in roots]
    ats = [s.tasks[a.key] for a in aggs]
    ets = s.tasks["extra"]

    fam = family(rts[0], 1000, 1000)
    assert fam
    sibs, downstream = fam
    assert sibs == set(rts[1:4]) | {rts[4], rts[8], rts[12]}
    assert downstream == {ats[0], ets}

    fam = family(rts[1], 1000, 1000)
    assert fam
    sibs, downstream = fam
    assert sibs == {rts[0], rts[2], rts[3]}
    assert downstream == {ats[0]}
