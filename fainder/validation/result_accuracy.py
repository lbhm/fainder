import argparse
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from fainder.utils import (
    collection_accuracy_metrics,
    configure_run,
    load_input,
    query_accuracy_metrics,
    save_output,
)

if TYPE_CHECKING:
    from fainder.typing import PercentileQuery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute performance metrics for a collection of query results.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--ground-truth",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="path to execution output file with ground truth results",
        metavar="SRC",
    )
    parser.add_argument(
        "-p",
        "--prediction",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="path to execution output file with predicted results",
        metavar="DEST",
    )
    parser.add_argument(
        "--log-file",
        type=lambda s: Path(os.path.expandvars(s)),
        default=None,
        help="path to log file (default: %(default)s)",
        metavar="LOG",
    )

    return parser.parse_args()


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    configure_run("INFO")

    t_queries: list[PercentileQuery]
    p_queries: list[PercentileQuery]
    t_results: list[set[np.uint32]]
    p_results: list[set[np.uint32]]
    t_queries, t_results = load_input(args.ground_truth, name="ground truth")
    p_queries, p_results = load_input(args.prediction, name="prediction")

    assert t_queries == p_queries, "Ground truth and prediction queries are not equal."

    precision, recall, f1_score = collection_accuracy_metrics(t_results, p_results)
    logger.info(f"Precision: {np.mean(precision):.4f}")
    logger.info(f"Recall: {np.mean(recall):.4f}")
    logger.info(f"F1-score: {np.mean(f1_score):.4f}")

    if args.log_file:
        metrics: dict[str, dict[str, float]] = {}
        for i in range(len(t_queries)):
            metrics[str(t_queries)] = {}
            precision_, recall_, f1_score_ = query_accuracy_metrics(t_results[i], p_results[i])
            metrics[str(t_queries)]["precision"] = precision_
            metrics[str(t_queries)]["recall"] = recall_
            metrics[str(t_queries)]["f1_score"] = f1_score_

        save_output(args.log_file, metrics, name="metrics")

    end = time.perf_counter()
    logger.info(f"Computed metrics for {len(t_queries)} queries in {end - start:.2f}s.")


if __name__ == "__main__":
    main()
