import time
from functools import partial
from multiprocessing import Pool
from typing import Literal

import numpy as np
from loguru import logger
from scipy.stats import norm

from fainder.typing import PercentileQuery
from fainder.utils import ROUNDING_PRECISION


def query_normal_dist(dist: tuple[float, ...], query: PercentileQuery) -> bool:
    """Check if `percentile` percent of a normal distribution are `comparison` `value`.

    Args:
        dist: normal distribution as a mean-stddev tuple.
        query: a percentile query (percentile, comparison, reference value).

    Returns:
        bool: whether the distribution fulfills the query.
    """
    percentile, comparison, reference = query
    assert len(dist) == 2
    mean, stddev = dist
    if "l" in comparison:
        return bool(
            np.round(norm.cdf(reference, loc=mean, scale=stddev), ROUNDING_PRECISION)
            >= np.float32(percentile)
        )
    elif "g" in comparison:
        return bool(
            np.round(norm.sf(reference, loc=mean, scale=stddev), ROUNDING_PRECISION)
            >= np.float32(percentile)
        )
    else:
        raise ValueError("Invalid comparison.")


def query_dist_collection(
    dists: list[tuple[np.uint32, tuple[float, ...]]],
    kind: Literal["normal"],
    queries: list[PercentileQuery],
    n_workers: int | None,
) -> list[set[np.uint32]]:
    matches: list[set[np.uint32]] = []
    if kind == "normal":
        fn = query_normal_dist

    if n_workers is None:
        start = time.perf_counter()
        for query in queries:
            query_start = time.perf_counter()
            query_matches: set[np.uint32] = set()
            for idx, dist in dists:
                if fn(dist, query):
                    query_matches.add(idx)
            matches.append(query_matches)
            logger.trace(f"query_time, {time.perf_counter() - query_start}")
    else:
        if n_workers <= 0:
            raise ValueError("Number of workers must greather than 0.")

        with Pool(processes=n_workers) as pool:
            start = time.perf_counter()
            for query in queries:
                query_start = time.perf_counter()
                fn = partial(fn, query=query)
                results: list[bool] = pool.map(fn, [dist for _, dist in dists])
                matches.append({idx for i, (idx, _) in enumerate(dists) if results[i]})
                logger.trace(f"query_time, {time.perf_counter() - query_start}")

    end = time.perf_counter()
    logger.debug(f"Raw naive query execution time: {end - start:.6f}s")
    logger.trace(f"query_collection_time, {end - start}")
    return matches
