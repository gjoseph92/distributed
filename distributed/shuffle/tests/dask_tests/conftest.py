import pytest

import dask


@pytest.fixture(scope="module")
def shuffle_method():
    with dask.config.set(shuffle="p2p"):
        from distributed import Client
        from distributed.utils_test import cluster

        with cluster(worker_kwargs={"nthreads": 2}) as (scheduler, _):
            with Client(scheduler["address"]):
                yield "p2p"
