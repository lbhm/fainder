import argparse
import os
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from loguru import logger

from fainder.typing import F32Array, F64Array, Histogram, PercentileQuery
from fainder.utils import configure_run, load_input, save_output


def selectivity(
    ground_truth: list[set[np.uint32]],
    collection_size: int,
    workers: int | None = None,
) -> F32Array:
    if workers is None:
        return np.array(
            [len(result) / collection_size for result in ground_truth], dtype=np.float32
        )
    if workers < 1:
        raise ValueError("Number of workers must be greater than 0")
    with Pool(workers) as pool:
        fn = partial(_selectivity, collection_size=collection_size)
        return np.array(pool.map(fn, ground_truth), dtype=np.float32)


def _selectivity(result: set[np.uint32], collection_size: int) -> float:
    return len(result) / collection_size


def cluster_hits(
    queries: list[PercentileQuery],
    cluster_bins: list[F64Array],
    workers: int | None = None,
) -> F32Array:
    if workers is None:
        return np.array(
            [_cluster_hits(query, cluster_bins) for query in queries], dtype=np.float32
        )
    if workers < 1:
        raise ValueError("Number of workers must be greater than 0")
    with Pool(workers) as pool:
        fn = partial(_cluster_hits, cluster_bins=cluster_bins)
        return np.array(pool.map(fn, queries), dtype=np.float32)


def _cluster_hits(query: PercentileQuery, cluster_bins: list[F64Array]) -> float:
    return int(np.sum([bins[0] <= query[2] <= bins[-1] for bins in cluster_bins])) / len(
        cluster_bins
    )


def bin_edge_matches(
    queries: list[PercentileQuery],
    hists: list[tuple[np.uint32, Histogram]],
    workers: int | None = None,
) -> F32Array:
    if workers is None:
        return np.array([_bin_edge_matches(query, hists) for query in queries], dtype=np.float32)
    if workers < 1:
        raise ValueError("Number of workers must be greater than 0")
    with Pool(workers) as pool:
        fn = partial(_bin_edge_matches, hists=hists)
        return np.array(pool.map(fn, queries), dtype=np.float32)


def _bin_edge_matches(query: PercentileQuery, hists: list[tuple[np.uint32, Histogram]]) -> float:
    return int(
        np.sum(
            [np.any(np.isclose(bins, query[2], rtol=0, atol=1e-6)) for _, (_, bins) in hists],
            dtype=np.int32,
        )
    ) / len(hists)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute query metrics for a collection of queries.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["selectivity", "cluster_hits", "bin_edge_matches"],
        required=True,
        help="Metric to compute",
    )
    parser.add_argument(
        "-q",
        "--queries",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="Path to a query collection",
        metavar="SRC",
    )
    parser.add_argument(
        "-s",
        "--hists",
        type=lambda s: Path(os.path.expandvars(s)),
        help="Path to a histogram collection",
        metavar="SRC",
    )
    parser.add_argument(
        "-c",
        "--clustered-hists",
        type=lambda s: Path(os.path.expandvars(s)),
        help="Path to a collection of clustered histograms",
        metavar="SRC",
    )
    parser.add_argument(
        "-t",
        "--ground-truth",
        type=lambda s: Path(os.path.expandvars(s)),
        help="Path to the ground truth for the query collection",
        metavar="SRC",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="Path to the output file",
        metavar="DEST",
    )
    parser.add_argument(
        "-w",
        "--workers",
        default=None,
        type=int,
        help="number of workers to use (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        choices=["INFO", "DEBUG"],
        help="verbosity of STDOUT logs (default: %(default)s)",
    )

    return parser.parse_args()


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    configure_run(args.log_level)
    logger.debug(vars(args))

    n_queries = 0
    if args.metric == "selectivity":
        if not args.ground_truth or not args.hists:
            raise ValueError("Missing arguments for selectivity metric")

        ground_truth = load_input(args.ground_truth, name="ground truth")
        hists = load_input(args.hists, name="histograms")
        metrics = selectivity(ground_truth, len(hists), args.workers)
    elif args.metric == "cluster_hits":
        if not args.queries or not args.clustered_hists:
            raise ValueError("Missing arguments for cluster hits metric")

        queries = load_input(args.queries, name="queries")
        _, cluster_bins = load_input(args.clustered_hists, name="clustered histograms")
        metrics = cluster_hits(queries, cluster_bins, args.workers)
    elif args.metric == "bin_edge_matches":
        if not args.queries or not args.hists:
            raise ValueError("Missing arguments for bin edge matches metric")

        queries = load_input(args.queries, name="queries")
        hists = load_input(args.hists, name="histograms")
        metrics = bin_edge_matches(queries, hists, args.workers)
    else:
        raise ValueError(f"Unknown metric: {args.metric}")

    logger.debug("Saving output")
    save_output(args.output, metrics)

    logger.info(f"Computed metrics for {n_queries} queries in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
