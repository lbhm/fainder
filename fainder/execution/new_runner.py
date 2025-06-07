import time
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

from fainder.execution.parallel_processing import ParallelHistogramProcessor
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


def run_exact_parallel(
    fainder_index: tuple[list[PctlIndex], list[NDArray[np.float64]]],
    query: PctlQuery,
    parallel_processor: ParallelHistogramProcessor,
    id_filter: ArrayLike | None = None,
) -> tuple[NDArray[np.uint32], float]:
    """Run an exact percentile query using parallel processing.

    This function is thread-safe. Multiple threads can safely use the same
    ParallelHistogramProcessor instance thanks to internal locking that
    protects executor access during query submission.

    Args:
        fainder_index: The index to query
        query: The percentile query to run
        parallel_processor: The ParallelHistogramProcessor instance (thread-safe)
        id_filter: Optional filter for IDs to limit the candidates

    Returns:
        A tuple of (result array, runtime in seconds)
    """
    start = time.perf_counter()

    # Stage 1: Get recall results
    recall_result = query_index_single(
        query, *fainder_index, index_mode="recall", id_filter=id_filter
    )

    # Stage 2: Get precision results
    precision_result = query_index_single(
        query, *fainder_index, index_mode="precision", id_filter=id_filter
    )

    # Stage 3: Process histograms in parallel for the candidates
    pscan_start = time.perf_counter()
    candidates = np.setdiff1d(recall_result, precision_result)

    if candidates.size <= 0:
        pscan_result = np.array([], dtype=np.uint32)
    else:
        pscan_result = parallel_processor.query(query, id_filter=candidates)

    logger.debug(f"Parallel profile-scan took {time.perf_counter() - pscan_start:.5f}s")

    result = np.union1d(pscan_result, precision_result)

    end = time.perf_counter()
    return result, end - start
