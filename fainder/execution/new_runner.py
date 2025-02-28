import time
from typing import Literal

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike, NDArray

from fainder.execution.percentile_queries import query_hist_collection, query_index_single
from fainder.typing import Histogram
from fainder.typing import PercentileIndex as PctlIndex
from fainder.typing import PercentileQuery as PctlQuery
from fainder.utils import filter_hists, filter_index


def unpack_and_filter(
    fainder_index: tuple[list[PctlIndex], list[NDArray[np.float64]]],
    id_filter: ArrayLike | None = None,
) -> tuple[list[PctlIndex], list[NDArray[np.float64]]]:
    pctl_index, cluster_bins = fainder_index
    if id_filter is not None:
        filter_start = time.perf_counter()
        pctl_index, cluster_bins = filter_index(pctl_index, cluster_bins, id_filter)
        logger.debug(f"Filtered index in {time.perf_counter() - filter_start:.5f}s")
    return pctl_index, cluster_bins


def run_approx(
    fainder_index: tuple[list[PctlIndex], list[NDArray[np.float64]]],
    query: PctlQuery,
    index_mode: Literal["precision", "recall"] = "recall",
    id_filter: ArrayLike | None = None,
) -> tuple[set[np.uint32], float]:
    start = time.perf_counter()
    pctl_index, cluster_bins = unpack_and_filter(fainder_index, id_filter)

    result = query_index_single(query, pctl_index, cluster_bins, index_mode)

    end = time.perf_counter()
    return result, end - start


def run_exact(
    fainder_index: tuple[list[PctlIndex], list[NDArray[np.float64]]],
    hists: list[tuple[np.uint32, Histogram]],
    query: PctlQuery,
    id_filter: ArrayLike | None = None,
) -> tuple[set[np.uint32], float]:
    start = time.perf_counter()
    pctl_index, cluster_bins = unpack_and_filter(fainder_index, id_filter)

    # Stage 1
    recall_result = query_index_single(query, pctl_index, cluster_bins, "recall")

    # Stage 2
    # NOTE: We could filter the index again before computing the precision result.
    # We need to analyze if this is faster.
    precision_result = query_index_single(query, pctl_index, cluster_bins, "precision")

    # Stage 3
    filtered_hists = filter_hists(hists, recall_result - precision_result)
    pscan_start = time.perf_counter()
    pscan_result = query_hist_collection(query, filtered_hists)
    logger.debug(f"profile-scan took {time.perf_counter() - pscan_start:.5f}s")
    result = pscan_result | precision_result

    end = time.perf_counter()
    return result, end - start
