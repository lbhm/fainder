import time
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

from fainder.execution.percentile_queries import query_hist_collection, query_index_single
from fainder.typing import Histogram
from fainder.typing import PercentileIndex as PctlIndex
from fainder.typing import PercentileQuery as PctlQuery


def run_approx(
    fainder_index: tuple[list[PctlIndex], list[NDArray[np.float64]]],
    query: PctlQuery,
    index_mode: Literal["precision", "recall"] = "recall",
    id_filter: ArrayLike | None = None,
) -> tuple[NDArray[np.uint32], float]:
    start = time.perf_counter()
    result = query_index_single(query, *fainder_index, index_mode=index_mode, id_filter=id_filter)
    end = time.perf_counter()

    return result, end - start


def run_exact(
    fainder_index: tuple[list[PctlIndex], list[NDArray[np.float64]]],
    hists: Sequence[tuple[int | np.integer[Any], Histogram]],
    query: PctlQuery,
    id_filter: ArrayLike | None = None,
) -> tuple[NDArray[np.uint32], float]:
    start = time.perf_counter()

    # Stage 1
    recall_result = query_index_single(
        query, *fainder_index, index_mode="recall", id_filter=id_filter
    )

    # Stage 2
    # NOTE: We could extend the filter with the recall result before computing the precision result
    # We need to analyze if this is faster or not
    # if id_filter is not None:
    #     id_filter = np.unique(np.concatenate([id_filter, list(recall_result)]))
    precision_result = query_index_single(
        query, *fainder_index, index_mode="precision", id_filter=id_filter
    )

    # Stage 3
    pscan_start = time.perf_counter()
    pscan_result = query_hist_collection(
        query,
        hists,
        id_filter=set(np.setdiff1d(recall_result, precision_result, assume_unique=True)),
    )
    logger.debug(f"profile-scan took {time.perf_counter() - pscan_start:.5f}s")
    result = np.union1d(pscan_result, precision_result)

    end = time.perf_counter()
    return result, end - start
