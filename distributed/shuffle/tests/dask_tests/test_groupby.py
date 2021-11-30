import pytest

from dask.dataframe.tests.test_groupby import *  # noqa

pytestmark = pytest.mark.dask_shuffle
