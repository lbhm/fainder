import argparse
import os
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger

from fainder.execution.percentile_queries import (
    query_conversion_collection,
    query_histogram_collection,
    query_local_index,
    query_rebinned_collection,
    trace_local_index,
)
from fainder.typing import PercentileQuery
from fainder.utils import configure_run, load_input, parse_percentile_query, save_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single query over a collection of histograms.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="path to compressed input data",
        metavar="SRC",
    )
    parser.add_argument(
        "-t",
        "--input-type",
        type=str,
        choices=["histograms", "rebinned_hists", "conversion_matrices", "index", "index_trace"],
        required=True,
        help="content type of the input data",
    )
    parser.add_argument(
        "-q",
        "--query",
        nargs=3,
        type=str,
        required=True,
        help="provided as PERCENTILE COMPARISON REFERENCE",
    )
    parser.add_argument(
        "-e",
        "--estimation-mode",
        type=str,
        choices=["over", "under", "continuous_value", "cubic_spline"],
        help=(
            "intra-bin estimation approach (only over and under for conversion matrices, default:"
            " %(default)s)"
        ),
    )
    parser.add_argument(
        "-m",
        "--index-mode",
        type=str,
        choices=["precision", "recall"],
        help="whether to optimize the index for precision or recall (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=lambda s: Path(os.path.expandvars(s)),
        help="file path to store the query results (default: %(default)s)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        default=None,
        type=int,
        help="number of worker processes (default: %(default)s)",
    )
    parser.add_argument(
        "--frequency-hists",
        action="store_true",
        help="flag for frequency instead of density histograms (ignored for the index)",
    )
    parser.add_argument(
        "--suppress-results",
        action="store_true",
        help=(
            "suppress returning query results to exclude the time for serialization (only for"
            " benchmarking)"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        choices=["DEBUG", "INFO", "TRACE"],
        help="verbosity of STDOUT logs (default: %(default)s)",
    )
    parser.add_argument(
        "--log-file",
        type=lambda s: Path(os.path.expandvars(s)),
        default=None,
        help="path to log file (default: %(default)s)",
        metavar="LOG",
    )

    return parser.parse_args()


def run(
    input_data: Any,
    query: PercentileQuery,
    input_type: Literal[
        "histograms", "rebinned_hists", "conversion_matrices", "index", "index_trace"
    ],
    estimation_mode: Literal["over", "under", "continuous_value", "cubic_spline"] = "over",
    index_mode: Literal["precision", "recall"] = "recall",
    frequency_hists: bool = False,
    suppress_results: bool = False,
    workers: int | None = None,
) -> tuple[set[np.uint32], float]:
    if input_type == "index":
        if estimation_mode:
            logger.debug("Ignoring --estimation-mode as it is not relevant for the index.")
        if frequency_hists:
            raise ValueError("The index only supports density histograms.")
    else:
        if index_mode:
            logger.debug("Ignoring --index-mode as it is only relevant for non-index methods.")
        if input_type == "conversion_matrices" and estimation_mode not in ["over", "under"]:
            raise ValueError("Conversion matrices only support over and under estimation.")

    logger.debug("Starting execution")
    start = time.perf_counter()
    if input_type == "histograms":
        hists = input_data
        results = query_histogram_collection(
            hists, estimation_mode, [query], workers, not frequency_hists
        )
    elif input_type == "rebinned_hists":
        rebinned_hists, cluster_bins = input_data
        results = query_rebinned_collection(
            rebinned_hists,
            cluster_bins,
            estimation_mode,
            [query],
            workers,
            not frequency_hists,
        )
    elif input_type == "conversion_matrices":
        clustered_hists, conversion_matrices, cluster_bins = input_data
        results = query_conversion_collection(
            clustered_hists,
            conversion_matrices,
            cluster_bins,
            estimation_mode,  # type: ignore
            [query],
            workers,
            not frequency_hists,
        )
    elif "index" in input_type:
        pctl_index, cluster_bins = input_data
        if workers:
            logger.debug("Ignoring --workers as a single index query cannot be parallelized")
        if input_type == "index":
            results = query_local_index(
                pctl_index, cluster_bins, index_mode, [query], suppress_results
            )
        else:
            results = [trace_local_index(pctl_index, cluster_bins, index_mode, query)]

    else:
        raise ValueError(f"Invalid input type {input_type}.")
    end = time.perf_counter()
    logger.debug("Execution finished")

    logger.info(f"Ran query in {end - start:.4g}s")
    logger.trace(f"execution_time, {end - start}")

    return results[0], end - start


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    configure_run(args.log_level, args.log_file)
    logger.debug(vars(args))

    input_data = load_input(args.input, name="input data")
    query = parse_percentile_query(args.query)
    logger.trace(f"bootstrap_time, {time.perf_counter() - start}")
    results, _ = run(
        input_data,
        query,
        args.input_type,
        args.estimation_mode,
        args.index_mode,
        args.frequency_hists,
        args.suppress_results,
        args.workers,
    )

    if args.output:
        save_output(args.output, (query, results), name="results")

    end = time.perf_counter()
    logger.trace(f"total_time, {end - start}")


if __name__ == "main":
    main()
