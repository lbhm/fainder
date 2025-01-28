import argparse
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from fainder.typing import PercentileQuery
from fainder.utils import configure_run, load_input, query_accuracy_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assert the correctness of an index.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--baseline-results",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="path to query results based on intermediate result data structures",
        metavar="SRC",
    )
    parser.add_argument(
        "-i",
        "--index-results",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="path to index-based query results",
        metavar="DEST",
    )
    parser.add_argument(
        "-g",
        "--ground-truth-results",
        type=lambda s: Path(os.path.expandvars(s)),
        default=None,
        help="path to ground truth query results (default: %(default)s)",
        metavar="DEST",
    )
    parser.add_argument(
        "-a",
        "--assert-metric",
        type=str,
        choices=["precision", "recall"],
        default=None,
        help="assert that the precision or recall is always 1 (default: %(default)s)",
    )

    return parser.parse_args()


def check_correctness(
    baseline_results: tuple[list[PercentileQuery], list[set[np.uint32]]],
    index_results: tuple[list[PercentileQuery], list[set[np.uint32]]],
    ground_truth_results: tuple[list[PercentileQuery], list[set[np.uint32]]] | None = None,
    metric: Literal["precision", "recall"] | None = None,
) -> bool:
    assert baseline_results[0] == index_results[0], "Baseline and index queries not equal."

    if ground_truth_results:
        assert ground_truth_results[0] == baseline_results[0], (
            "Ground truth and baseline queries not equal."
        )

    for i in range(len(baseline_results[0])):
        assert baseline_results[1][i] == index_results[1][i], (
            f"Results of query {i} not identical."
        )
        if ground_truth_results and metric:
            precision, recall, _ = query_accuracy_metrics(
                ground_truth_results[1][i], index_results[1][i]
            )
            if metric == "precision":
                assert precision == 1, f"Precision of query {i} != 1."
            elif metric == "recall":
                assert recall == 1, f"Recall of query {i} != 1."

    return True


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    configure_run("INFO")

    baseline_results = load_input(args.baseline_results, name="baseline results")
    index_results = load_input(args.index_results, name="index results")
    if args.ground_truth_results:
        ground_truth_results = load_input(args.assert_precision, name="ground truth results")
    else:
        ground_truth_results = None

    try:
        check_correctness(
            baseline_results, index_results, ground_truth_results, args.assert_metric
        )
        end = time.perf_counter()
        logger.info(
            f"Asserted index correctness over {len(baseline_results[0])} queries in"
            f" {end - start:.2f}s."
        )
    except AssertionError as e:
        logger.warning(f"Assertion failed: {e}")


if __name__ == "__main__":
    main()
