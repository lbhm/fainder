import itertools
import time
from functools import partial
from multiprocessing import Pool
from typing import Any, Literal

import numpy as np
from loguru import logger
from scipy.interpolate import CubicSpline

from fainder.preprocessing.percentile_index import load_shm_index
from fainder.typing import (
    BoolArray,
    F32Array,
    F64Array,
    FArray,
    Histogram,
    PercentileIndex,
    PercentileIndexPointer,
    PercentileQuery,
    UInt32Array,
)
from fainder.utils import ROUNDING_PRECISION, unlink_pointers

pctl_index: list[PercentileIndex]
cluster_bins: list[F64Array]


def query_histogram(
    hist: Histogram,
    estimation_mode: Literal["over", "under", "continuous_value", "cubic_spline"],
    query: PercentileQuery,
    density: bool = True,
) -> bool:
    """Check if `percentile` percent of a histogram's data are `comparison` `value`.

    Args:
        hist: histogram as a values-bins tuple.
        estimation_mode: assumption for approximating intra-bin value distributions.
            "over" computes an upper bound for the fraction of the data that match the
            query and therefore yields higher recall. "under" computes the respective
            lower bound, resulting in higher precision. "continuous_value" and
            "cubic_spline" approximate the fraction of a bin using a uniformity
            assumption or spline interpolation, respectively.
        query: a percentile query (percentile, comparison, reference value).
        density: boolean flag whether the histogram contains densities or frequencies.

    Returns:
        bool: whether the histogram fulfills the query.
    """
    values, bins = hist
    percentile, comparison, reference = query
    fraction = 0.0

    assert 0 < percentile <= 1
    bin_index = np.searchsorted(bins, reference, "left") - 1  # type: ignore
    if "l" in comparison:
        if bin_index == -1:
            pass
        elif bin_index == len(bins) - 1:
            fraction = values.sum()
        else:
            fraction = values[:bin_index].sum()
            if bins[bin_index + 1] == reference:
                if "e" in comparison and estimation_mode == "over" and bin_index < len(bins) - 2:
                    # All bins except the last are left-closed, so we have to add the
                    # value of the next bin to the fraction if the reference value is
                    # equal to the upper bin edge.
                    fraction += values[bin_index] + values[bin_index + 1]
                else:
                    fraction += values[bin_index]
            if estimation_mode == "over":
                fraction += values[bin_index]
            elif estimation_mode == "under":
                pass
            elif estimation_mode == "continuous_value":
                fraction += (
                    (reference - bins[bin_index]) / (bins[bin_index + 1] - bins[bin_index])
                ) * values[bin_index]
            elif estimation_mode == "cubic_spline":
                bin_mids = np.hstack((bins[0], bins[:-1] + np.diff(bins) / 2, bins[-1]))
                bin_heights = np.hstack((0, values, 0))
                spline = CubicSpline(x=bin_mids, y=bin_heights, bc_type="clamped")
                fraction += (
                    spline.integrate(bins[bin_index], reference)
                    / spline.integrate(bins[bin_index], bins[bin_index + 1])
                ) * values[bin_index]
            else:
                raise ValueError(f"Invalid estimation mode: {estimation_mode}")
    elif "g" in comparison:
        if bin_index == -1:
            fraction = values.sum()
        elif bin_index == len(bins) - 1:
            pass
        else:
            fraction = values[bin_index + 1 :].sum()
            if bins[bin_index + 1] == reference:
                # All bins except the last are left-closed, see above.
                if "t" in comparison and estimation_mode == "under":
                    fraction -= values[bin_index + 1]
                else:
                    pass
            elif estimation_mode == "over":
                fraction += values[bin_index]
            elif estimation_mode == "under":
                pass
            elif estimation_mode == "continuous_value":
                fraction += (
                    (bins[bin_index + 1] - reference) / (bins[bin_index + 1] - bins[bin_index])
                ) * values[bin_index]
            elif estimation_mode == "cubic_spline":
                bin_mids = np.hstack((bins[0], bins[:-1] + np.diff(bins) / 2, bins[-1]))
                bin_heights = np.hstack((0, values, 0))
                spline = CubicSpline(x=bin_mids, y=bin_heights, bc_type="clamped")
                fraction += (
                    spline.integrate(reference, bins[bin_index + 1])
                    / spline.integrate(bins[bin_index], bins[bin_index + 1])
                ) * values[bin_index]
            else:
                raise ValueError(f"Invalid estimation mode: {estimation_mode}")
    else:
        raise ValueError("Invalid comparison.")

    if density:
        return bool(np.round(fraction, ROUNDING_PRECISION) >= np.float32(percentile))
    return bool(np.round(fraction / values.sum(), ROUNDING_PRECISION) >= np.float32(percentile))


def query_conversion_matrix(
    hist: Histogram,
    conversion_matrix: BoolArray,
    global_bins: F64Array,
    estimation_mode: Literal["under", "over"],
    query: PercentileQuery,
    density: bool = True,
) -> bool:
    """This method has an built-in under- or overestimation to achieve full precision or recall."""
    values, _ = hist
    percentile, comparison, reference = query
    fraction = 0.0

    assert 0 < percentile <= 1
    bin_index = np.searchsorted(global_bins, reference, "left") - 1  # type: ignore
    if "l" in comparison:
        if bin_index == -1:
            pass
        elif bin_index == len(global_bins) - 1:
            fraction = values.sum()
        else:
            if estimation_mode == "under":
                raise NotImplementedError(
                    "Underestimation with conversion matrices not implemented yet, use the index."
                )
                # mask = np.invert(np.cumsum(conversion_matrix[:, bin_index], dtype=np.bool_))
            mask = np.any(conversion_matrix[:, : bin_index + 1], axis=1)
            fraction = values[mask].sum()
    elif "g" in comparison:
        if bin_index == -1:
            fraction = values.sum()
        elif bin_index == len(global_bins) - 1:
            pass
        else:
            if estimation_mode == "under":
                raise NotImplementedError(
                    "Underestimation with conversion matrices not implemented yet, use the index."
                )
                # mask = np.invert(
                #     np.cumsum(conversion_matrix[:, bin_index][::-1], dtype=np.bool_)[::-1]
                # )
            mask = np.any(conversion_matrix[:, bin_index:], axis=1)
            fraction = values[mask].sum()
    else:
        raise ValueError("Invalid comparison.")

    if density:
        return bool(np.round(fraction, ROUNDING_PRECISION) >= np.float32(percentile))
    return bool(np.round(fraction / values.sum(), ROUNDING_PRECISION) >= np.float32(percentile))


def query_histogram_collection(
    hists: list[tuple[np.uint32, Histogram]],
    estimation_mode: Literal["over", "under", "continuous_value", "cubic_spline"],
    queries: list[PercentileQuery],
    n_workers: int | None,
    density: bool = True,
) -> list[set[np.uint32]]:
    matches: list[set[np.uint32]] = []

    if n_workers is None:
        start = time.perf_counter()
        for query in queries:
            query_start = time.perf_counter()
            query_matches: set[np.uint32] = set()
            for idx, hist in hists:
                if query_histogram(hist, estimation_mode, query, density):
                    query_matches.add(idx)
            matches.append(query_matches)
            logger.trace(f"query_time, {time.perf_counter() - query_start}")
    else:
        if n_workers <= 0:
            raise ValueError("Number of workers must greater than 0 (or None).")

        with Pool(processes=n_workers) as pool:
            start = time.perf_counter()
            for query in queries:
                query_start = time.perf_counter()
                fn = partial(
                    query_histogram, estimation_mode=estimation_mode, query=query, density=density
                )
                results = pool.map(fn, [hist for _, hist in hists])
                matches.append({idx for j, (idx, _) in enumerate(hists) if results[j]})
                logger.trace(f"query_time, {time.perf_counter() - query_start}")

    end = time.perf_counter()
    logger.debug(f"Raw naive query execution time: {end - start:.6f}s")
    logger.trace(f"query_collection_time, {end - start}")
    return matches


def query_rebinned_collection(
    hists: list[tuple[UInt32Array, F32Array]],
    cluster_bins: list[F64Array],
    estimation_mode: Literal["over", "under", "continuous_value", "cubic_spline"],
    queries: list[PercentileQuery],
    n_workers: int | None,
    density: bool = True,
) -> list[set[np.uint32]]:
    matches: list[set[np.uint32]] = []
    query_matches: set[np.uint32]

    if n_workers is None:
        start = time.perf_counter()
        for query in queries:
            query_start = time.perf_counter()
            query_matches = set()
            _, comparison, reference = query
            for i, bins in enumerate(cluster_bins):
                if bins[0] <= reference <= bins[-1]:
                    for j, hist in enumerate(hists[i][1]):
                        if query_histogram(
                            (hist, bins),
                            estimation_mode,
                            query,
                            density,
                        ):
                            query_matches.add(hists[i][0][j])
                else:
                    # Reference value not in cluster range
                    if (reference <= bins[0] and "g" in comparison) or (
                        reference >= bins[-1] and "l" in comparison
                    ):
                        query_matches.update(hists[i][0])
            matches.append(query_matches)
            logger.trace(f"query_time, {time.perf_counter() - query_start}")
    else:
        if n_workers <= 0:
            raise ValueError("Number of workers must greater than 0 (or None).")

        with Pool(processes=n_workers) as pool:
            start = time.perf_counter()
            for query in queries:
                query_start = time.perf_counter()
                query_matches = set()
                _, comparison, reference = query
                for i, bins in enumerate(cluster_bins):
                    if bins[0] <= reference <= bins[-1]:
                        # NOTE: Rebinning MP stalls with many global bins due to IPC)
                        results = pool.map(
                            partial(
                                query_histogram,
                                estimation_mode=estimation_mode,
                                query=query,
                                density=density,
                            ),
                            zip(hists[i][1], itertools.repeat(bins)),
                        )
                        query_matches.update(hists[i][0][np.array(results, dtype=np.bool_)])
                    else:
                        # Reference value not in cluster range
                        if (reference <= bins[0] and "g" in comparison) or (
                            reference >= bins[-1] and "l" in comparison
                        ):
                            query_matches.update(hists[i][0])
                matches.append(query_matches)
                logger.trace(f"query_time, {time.perf_counter() - query_start}")

    end = time.perf_counter()
    logger.debug(f"Raw rebinning query execution time: {end - start:.6f}s")
    logger.trace(f"query_collection_time, {end - start}")
    return matches


def query_conversion_collection(
    hists: list[list[tuple[np.uint32, Histogram]]],
    conversion_matrices: list[list[BoolArray]],
    cluster_bins: list[F64Array],
    estimation_mode: Literal["under", "over"],
    queries: list[PercentileQuery],
    n_workers: int | None,
    density: bool = True,
) -> list[set[np.uint32]]:
    matches: list[set[np.uint32]] = []
    query_matches: set[np.uint32]

    if n_workers is None:
        start = time.perf_counter()
        for query in queries:
            query_start = time.perf_counter()
            query_matches = set()
            _, comparison, reference = query
            for i, bins in enumerate(cluster_bins):
                if bins[0] <= reference <= bins[-1]:
                    for j, (idx, hist) in enumerate(hists[i]):
                        if query_conversion_matrix(
                            hist,
                            conversion_matrices[i][j],
                            bins,
                            estimation_mode,
                            query,
                            density,
                        ):
                            query_matches.add(idx)
                else:
                    # Reference value not in cluster range
                    if (reference <= bins[0] and "g" in comparison) or (
                        reference >= bins[-1] and "l" in comparison
                    ):
                        query_matches.update([idx for idx, _ in hists[i]])
            matches.append(query_matches)
            logger.trace(f"query_time, {time.perf_counter() - query_start}")
    else:
        if n_workers <= 0:
            raise ValueError("Number of workers must greater than 0 (or None).")

        with Pool(processes=n_workers) as pool:
            start = time.perf_counter()
            for query in queries:
                query_start = time.perf_counter()
                query_matches = set()
                _, comparison, reference = query
                for i, bins in enumerate(cluster_bins):
                    if bins[0] <= reference <= bins[-1]:
                        results = pool.starmap(
                            partial(
                                query_conversion_matrix,
                                global_bins=bins,
                                estimation_mode=estimation_mode,
                                query=query,
                                density=density,
                            ),
                            zip(
                                [hist for _, hist in hists[i]],
                                conversion_matrices[i],
                                strict=True,
                            ),
                        )
                        query_matches.update(
                            [idx for j, (idx, _) in enumerate(hists[i]) if results[j]]
                        )
                    else:
                        # Reference value not in cluster range
                        if (reference <= bins[0] and "g" in comparison) or (
                            reference >= bins[-1] and "l" in comparison
                        ):
                            query_matches.update([idx for idx, _ in hists[i]])
                matches.append(query_matches)
                logger.trace(f"query_time, {time.perf_counter() - query_start}")

    end = time.perf_counter()
    logger.debug(f"Raw conversion query execution time: {end - start:.6f}s")
    logger.trace(f"query_collection_time, {end - start}")
    return matches


def query_local_index(
    pctl_index: list[PercentileIndex],
    cluster_bins: list[F64Array],
    index_mode: Literal["precision", "recall"],
    queries: list[PercentileQuery],
    suppress_results: bool = False,
) -> list[set[np.uint32]]:
    start = time.perf_counter()
    matches: list[set[np.uint32]] = []
    method = "rebinning" if len(pctl_index[0]) == 1 else "conversion"
    dtype = pctl_index[0][0][0].dtype

    result_sum = 0
    for query in queries:
        query_start = time.perf_counter()
        query_matches: set[np.uint32] = set()
        percentile, comparison, reference = query

        assert 0 < percentile <= 1
        if "g" in comparison:
            # We can only run <(=) with our cumsum index so we have to reqrite >(=) queries
            percentile = 1.0 - percentile

        bin_mode = 0
        pctl_mode = 0
        if ("g" in comparison and index_mode == "precision") or (
            "l" in comparison and index_mode == "recall"
        ):
            if method == "rebinning":
                bin_mode = 1
            if method == "conversion":
                pctl_mode = 1

        for i, bins in enumerate(cluster_bins):
            if bins[0] <= reference <= bins[-1]:
                bin_index = (
                    np.clip(np.searchsorted(bins, reference, "left") - 1, 0, len(bins) - 1)
                    + bin_mode
                )
                if "l" in comparison:
                    hist_index = np.searchsorted(
                        pctl_index[i][pctl_mode][0][:, bin_index], dtype.type(percentile), "left"
                    )
                    query_matches.update(
                        np.zeros(1, dtype=np.uint32)
                        if suppress_results
                        else pctl_index[i][pctl_mode][1][hist_index:, bin_index]
                    )
                elif "g" in comparison:
                    hist_index = np.searchsorted(
                        pctl_index[i][pctl_mode][0][:, bin_index], dtype.type(percentile), "right"
                    )
                    query_matches.update(
                        np.zeros(1, dtype=np.uint32)
                        if suppress_results
                        else pctl_index[i][pctl_mode][1][:hist_index, bin_index]
                    )
                else:
                    raise ValueError("Invalid comparison.")
            else:
                # Reference value not in cluster range
                if (reference <= bins[0] and "g" in comparison) or (
                    reference >= bins[-1] and "l" in comparison
                ):
                    query_matches.update(
                        np.zeros(1, dtype=np.uint32)
                        if suppress_results
                        else pctl_index[i][pctl_mode][1][:, 0]
                    )
        matches.append(query_matches)
        logger.trace(f"query_time, {time.perf_counter() - query_start}")
        result_sum += len(query_matches)

    end = time.perf_counter()
    logger.debug(f"Raw index-based query execution time: {end - start:.6f}s")
    logger.trace(f"query_collection_time, {end - start}")
    logger.trace(f"avg_result_size, {result_sum / len(queries)}")
    return matches


def trace_local_index(
    pctl_index: list[PercentileIndex],
    cluster_bins: list[F64Array],
    index_mode: Literal["precision", "recall"],
    query: PercentileQuery,
) -> set[np.uint32]:
    """A reduced version of `query_local_index` to trace the execution of a single query."""
    start = time.perf_counter()
    matches: set[np.uint32] = set()
    method = "rebinning" if len(pctl_index[0]) == 1 else "conversion"
    percentile, comparison, reference = query

    assert 0 < percentile <= 1
    if "g" in comparison:
        percentile = 1.0 - percentile

    bin_mode = 0
    pctl_mode = 0
    if ("g" in comparison and index_mode == "precision") or (
        "l" in comparison and index_mode == "recall"
    ):
        if method == "rebinning":
            bin_mode = 1
        if method == "conversion":
            pctl_mode = 1

    post_bootstrap = time.perf_counter()
    logger.trace(f"query_boostrap_time, {post_bootstrap - start}")
    for i, bins in enumerate(cluster_bins):
        if bins[0] <= reference <= bins[-1]:
            pre_bin_search = time.perf_counter()
            bin_index = (
                np.clip(np.searchsorted(bins, reference, "left") - 1, 0, len(bins) - 1) + bin_mode
            )
            logger.trace(f"query_bin_search_time, {time.perf_counter() - pre_bin_search}")
            if "l" in comparison:
                pre_hist_search = time.perf_counter()
                hist_index = np.searchsorted(
                    pctl_index[i][pctl_mode][0][:, bin_index], percentile, "left"
                )
                post_hist_search = time.perf_counter()
                matches.update(pctl_index[i][pctl_mode][1][hist_index:, bin_index])
                post_result_update = time.perf_counter()

                logger.trace(f"query_hist_search_time, {post_hist_search - pre_hist_search}")
                logger.trace(f"query_result_update_time, {post_result_update - post_hist_search}")
            elif "g" in comparison:
                pre_hist_search = time.perf_counter()
                hist_index = np.searchsorted(
                    pctl_index[i][pctl_mode][0][:, bin_index], percentile, "right"
                )
                post_hist_search = time.perf_counter()
                matches.update(pctl_index[i][pctl_mode][1][:hist_index, bin_index])
                post_result_update = time.perf_counter()

                logger.trace(f"query_hist_search_time, {post_hist_search - pre_hist_search}")
                logger.trace(f"query_result_update_time, {post_result_update - post_hist_search}")
            else:
                raise ValueError("Invalid comparison.")
        else:
            pre_cluster_skip = time.perf_counter()
            if (reference <= bins[0] and "g" in comparison) or (
                reference >= bins[-1] and "l" in comparison
            ):
                matches.update(pctl_index[i][pctl_mode][1][:, 0])
            post_cluster_skip = time.perf_counter()
            logger.trace(f"query_cluster_skip_time, {post_cluster_skip - pre_cluster_skip}")

    end = time.perf_counter()
    logger.trace(f"query_time, {end - start}")
    logger.debug(f"Raw index-based query execution time: {end - start:.6f}s")
    return matches


def init_index_workers(
    shm_pointers: list[PercentileIndexPointer],
    n_hists: list[int],
    bins: list[F64Array],
    method: Literal["rebinning", "conversion"],
    index_dtype: np.dtype[Any],
) -> None:
    global pctl_index
    global cluster_bins

    assert len(shm_pointers) == len(bins)
    cluster_bins = bins
    pctl_index = []
    mode = 0 if method == "rebinning" else 1
    for i, cluster in enumerate(shm_pointers):
        n_bins = len(cluster_bins[i])
        index_part: list[tuple[FArray, UInt32Array]] = []
        for pointers in cluster:
            pctls: FArray = np.ndarray(
                (n_hists[i], n_bins - mode), dtype=index_dtype, order="F", buffer=pointers[0].buf
            )
            ids: UInt32Array = np.ndarray(
                (n_hists[i], n_bins - mode), dtype=np.uint32, order="F", buffer=pointers[1].buf
            )
            index_part.append((pctls, ids))
        pctl_index.append(tuple(index_part))  # type: ignore


def query_index_worker(
    query: PercentileQuery,
    method: Literal["rebinning", "conversion"],
    index_mode: Literal["precision", "recall"],
    suppress_results: bool = False,
) -> set[np.uint32]:
    matches: set[np.uint32] = set()
    dtype = pctl_index[0][0][0].dtype
    percentile, comparison, reference = query

    assert 0 < percentile <= 1
    if "g" in comparison:
        # We can only run <(=) with our cumsum index so we have to reqrite > queries
        percentile = 1.0 - percentile

    bin_mode = 0
    pctl_mode = 0
    if ("g" in comparison and index_mode == "precision") or (
        "l" in comparison and index_mode == "recall"
    ):
        if method == "rebinning":
            bin_mode = 1
        if method == "conversion":
            pctl_mode = 1

    for i, bins in enumerate(cluster_bins):
        if bins[0] <= reference <= bins[-1]:
            bin_index = (
                np.clip(np.searchsorted(bins, reference, "left") - 1, 0, len(bins) - 1) + bin_mode
            )
            if "l" in comparison:
                hist_index = np.searchsorted(
                    pctl_index[i][pctl_mode][0][:, bin_index], dtype.type(percentile), "left"
                )
                matches.update(pctl_index[i][pctl_mode][1][hist_index:, bin_index])
            elif "g" in comparison:
                hist_index = np.searchsorted(
                    pctl_index[i][pctl_mode][0][:, bin_index], dtype.type(percentile), "right"
                )
                matches.update(pctl_index[i][pctl_mode][1][:hist_index, bin_index])
            else:
                raise ValueError("Invalid comparison.")
        else:
            # Reference value not in cluster range
            if (reference <= bins[0] and "g" in comparison) or (
                reference >= bins[-1] and "l" in comparison
            ):
                matches.update(pctl_index[i][pctl_mode][1][:, 0])

    return set() if suppress_results else matches


def query_index(
    pctl_index: list[PercentileIndex],
    cluster_bins: list[F64Array],
    index_mode: Literal["precision", "recall"],
    queries: list[PercentileQuery],
    n_workers: int | None,
    suppress_results: bool = False,
) -> list[set[np.uint32]]:
    matches: list[set[np.uint32]]

    if n_workers is None:
        return query_local_index(pctl_index, cluster_bins, index_mode, queries, suppress_results)
    if n_workers <= 0:
        raise ValueError("Number of workers must greater than 0 (or None).")

    method: Literal["rebinning", "conversion"] = (
        "rebinning" if len(pctl_index[0]) == 1 else "conversion"
    )
    n_hists = [len(t[0][0]) for t in pctl_index]
    index_dtype = pctl_index[0][0][0].dtype
    shm_pointers = load_shm_index(pctl_index)
    try:
        logger.debug("Initalizing index worker pool")
        with Pool(
            processes=n_workers,
            initializer=init_index_workers,
            initargs=(shm_pointers, n_hists, cluster_bins, method, index_dtype),
        ) as pool:
            fn = partial(
                query_index_worker,
                method=method,
                index_mode=index_mode,
                suppress_results=suppress_results,
            )
            logger.debug("Index worker pool initialized")
            start = time.perf_counter()
            matches = pool.map(fn, queries)
            end = time.perf_counter()
            logger.debug(f"Raw index-based query execution time: {end - start:.6f}s")
            logger.trace(f"query_collection_time, {end - start}")
    finally:
        unlink_pointers(shm_pointers)

    return matches


### The following methods were added for the Fainder demo


def query_index_single(
    query: PercentileQuery,
    pctl_index: list[PercentileIndex],
    cluster_bins: list[F64Array],
    index_mode: Literal["precision", "recall"],
) -> set[np.uint32]:
    percentile, comparison, reference = query
    index_type = "rebinning" if len(pctl_index[0]) == 1 else "conversion"
    dtype = pctl_index[0][0][0].dtype

    result: set[np.uint32] = set()

    if "g" in comparison:
        # We can only run <(=) with our cumsum index so we have to reqrite >(=) queries
        percentile = 1.0 - percentile

    bin_mode = 0
    pctl_mode = 0
    if ("g" in comparison and index_mode == "precision") or (
        "l" in comparison and index_mode == "recall"
    ):
        if index_type == "rebinning":
            bin_mode = 1
        if index_type == "conversion":
            pctl_mode = 1

    for i, bins in enumerate(cluster_bins):
        if bins[0] <= reference <= bins[-1]:
            bin_index = (
                np.clip(np.searchsorted(bins, reference, "left") - 1, 0, len(bins) - 1) + bin_mode
            )
            if "l" in comparison:
                hist_index = np.searchsorted(
                    pctl_index[i][pctl_mode][0][:, bin_index], dtype.type(percentile), "left"
                )
                result.update(pctl_index[i][pctl_mode][1][hist_index:, bin_index])
            elif "g" in comparison:
                hist_index = np.searchsorted(
                    pctl_index[i][pctl_mode][0][:, bin_index], dtype.type(percentile), "right"
                )
                result.update(pctl_index[i][pctl_mode][1][:hist_index, bin_index])
            else:
                raise ValueError("Invalid comparison.")
        else:
            # Reference value not in cluster range
            if (reference <= bins[0] and "g" in comparison) or (
                reference >= bins[-1] and "l" in comparison
            ):
                result.update(pctl_index[i][pctl_mode][1][:, 0])

    return result


def query_hist_collection(
    query: PercentileQuery,
    hists: list[tuple[np.uint32, Histogram]],
    density: bool = True,
) -> set[np.uint32]:
    return {
        idx
        for idx, hist in hists
        if query_histogram(hist, estimation_mode="over", query=query, density=density)
    }
