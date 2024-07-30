import argparse
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from fainder.execution.baselines import query_binsort
from fainder.execution.percentile_queries import query_histogram_collection
from fainder.typing import F32Array, F64Array, Histogram, PercentileQuery, UInt32Array
from fainder.utils import configure_run, load_input, save_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the exact result of a query based on index-preprocessing.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--histogram-input",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="path to histograms",
        metavar="SRC",
    )
    parser.add_argument(
        "-p",
        "--precision-input",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="path to results of a full precision query",
        metavar="SRC",
    )
    parser.add_argument(
        "-r",
        "--recall-input",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="path to results of a full recall query",
        metavar="SRC",
    )
    parser.add_argument(
        "-e",
        "--estimation-mode",
        type=str,
        choices=["over", "under", "continuous_value", "cubic_spline"],
        required=True,
        help=(
            "intra-bin estimation approach (only over and under for conversion matrices, default:"
            " %(default)s)"
        ),
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
        "--log-level",
        default="INFO",
        type=str,
        choices=["DEBUG", "INFO"],
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


def run_pscan(
    hists: list[tuple[np.uint32, Histogram]],
    precision_results: set[np.uint32],
    recall_results: set[np.uint32],
    query: PercentileQuery,
    estimation_mode: Literal["over", "under", "continuous_value", "cubic_spline"] = "over",
    frequency_hists: bool = False,
    workers: int | None = None,
) -> tuple[set[np.uint32], float]:
    potential_results = recall_results - precision_results
    filtered_hists = [h for h in hists if h[0] in potential_results]
    logger.debug(
        f"Executing query on {len(filtered_hists)} instead of {len(hists)} histograms"
        f" ({(1 - len(filtered_hists) / len(hists)) * 100:.2f}% reduction)"
    )

    logger.debug("Starting execution")
    start = time.perf_counter()
    result_list = query_histogram_collection(
        filtered_hists, estimation_mode, [query], workers, not frequency_hists
    )
    results = result_list[0] | precision_results
    end = time.perf_counter()
    logger.debug("Execution finished")

    logger.info(f"Ran query in {end - start:.4g}s")
    logger.trace(f"execution_time, {end - start}")

    return results, end - start


def run_pscan_collection(
    hists: list[tuple[np.uint32, Histogram]],
    precision_results: list[set[np.uint32]],
    recall_results: list[set[np.uint32]],
    queries: list[PercentileQuery],
    estimation_mode: Literal["over", "under", "continuous_value", "cubic_spline"] = "over",
    frequency_hists: bool = False,
    workers: int | None = None,
) -> tuple[list[set[np.uint32]], float, float]:
    start = time.perf_counter()
    results: list[set[np.uint32]] = []
    execution_time = 0.0
    avg_reduction = 0.0

    logger.debug("Starting execution")
    for i, query in enumerate(queries):
        potential_results = recall_results[i] - precision_results[i]
        filtered_hists = [h for h in hists if h[0] in potential_results]
        avg_reduction += 1 - len(filtered_hists) / len(hists)

        execution_start = time.perf_counter()
        result_list = query_histogram_collection(
            filtered_hists, estimation_mode, [query], workers, not frequency_hists
        )
        results.append(result_list[0] | precision_results[i])
        execution_time += time.perf_counter() - execution_start
    end = time.perf_counter()
    logger.debug("Execution finished")

    avg_reduction /= len(queries)

    logger.info(
        f"Ran {len(queries)} queries in {end - start:.4g}s with {execution_time:.4g}s raw"
        f" execution time and {avg_reduction * 100:.2f}% average reduction"
    )
    logger.trace(f"execution_time, {execution_time}")
    logger.trace(f"total_time, {end - start}")

    return results, execution_time, avg_reduction


def run_binsort_collection(
    binsort: tuple[F64Array, tuple[F32Array, F32Array, F32Array], UInt32Array],
    precision_results: list[set[np.uint32]],
    recall_results: list[set[np.uint32]],
    queries: list[PercentileQuery],
    index_mode: Literal["precision", "recall"] = "recall",
    workers: int | None = None,
) -> tuple[list[set[np.uint32]], float, float]:
    start = time.perf_counter()
    results: list[set[np.uint32]] = []
    execution_time = 0.0
    avg_reduction = 0.0

    logger.debug("Starting execution")
    for i, query in enumerate(queries):
        potential_results = recall_results[i] - precision_results[i]
        mask = np.isin(binsort[2], list(potential_results))
        avg_reduction += 1 - len(potential_results) / len(np.unique(binsort[2]))

        execution_start = time.perf_counter()
        result_list = query_binsort(
            (
                binsort[0][mask],
                (binsort[1][0][mask], binsort[1][1][mask], binsort[1][0][mask]),
                binsort[2][mask],
            ),
            index_mode,
            [query],
            workers,
        )
        results.append(result_list[0] | precision_results[i])
        execution_time += time.perf_counter() - execution_start
    end = time.perf_counter()
    logger.debug("Execution finished")

    avg_reduction /= len(queries)

    logger.info(
        f"Ran {len(queries)} queries in {end - start:.4g}s with {execution_time:.4g}s raw"
        f" execution time and {avg_reduction * 100:.2f}% average reduction"
    )
    logger.trace(f"execution_time, {execution_time}")
    logger.trace(f"total_time, {end - start}")

    return results, execution_time, avg_reduction


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    configure_run(args.log_level, args.log_file)
    logger.debug(vars(args))

    p_query, precision_results = load_input(args.precision_input, name="precision results")
    r_query, recall_results = load_input(args.recall_input, name="recall results")
    assert p_query == r_query

    hists = load_input(args.histogram_input, name="histograms")
    results, _ = run_pscan(
        hists,
        precision_results,
        recall_results,
        p_query,
        args.estimation_mode,
        args.frequency_hists,
        args.workers,
    )

    if args.output:
        save_output(args.output, (p_query, results), name="results")

    end = time.perf_counter()
    logger.trace(f"total_time, {end - start}")


if __name__ == "main":
    main()
