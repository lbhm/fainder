import argparse
import os
import time
from pathlib import Path

import numpy as np
from loguru import logger

from fainder.execution import runner
from fainder.utils import (
    collection_accuracy_metrics,
    configure_run,
    get_index_size,
    load_input,
    save_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the accuracy of a Fainder index given a query set and a ground truth."
        ),
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--index",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="Path to the Fainder index file",
        metavar="SRC",
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
        "-t",
        "--ground-truth",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="Path to the ground truth for the query collection",
        metavar="SRC",
    )
    parser.add_argument(
        "-w",
        "--workers",
        default=None,
        type=int,
        help="number of worker processes (default: %(default)s)",
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
        default=None,
        type=lambda s: Path(os.path.expandvars(s)),
        help="path to log file (default: %(default)s)",
    )
    parser.add_argument(
        "--log-runtime",
        action="store_true",
        help="also log runtime of each query (default: %(default)s)",
    )

    return parser.parse_args()


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    configure_run(
        args.log_level,
        args.log_file.with_suffix(".log") if args.log_file and args.log_runtime else None,
    )

    index = load_input(args.index, name="index")
    queries = load_input(args.queries, name="queries")
    ground_truth = load_input(args.ground_truth, name="ground truth")

    precision_results, precision_time = runner.run(
        input_data=index,
        queries=queries,
        input_type="index",
        index_mode="precision",
        workers=args.workers,
    )
    recall_results, recall_time = runner.run(
        input_data=index,
        queries=queries,
        input_type="index",
        index_mode="recall",
        workers=args.workers,
    )

    n_hists = sum([i[0][0].shape[0] for i in index[0]])
    logger.debug(f"Index contains {n_hists} histograms")
    precision_metrics = (
        *collection_accuracy_metrics(ground_truth, precision_results),
        [len(result) / n_hists for result in precision_results],
    )
    recall_metrics = (
        *collection_accuracy_metrics(ground_truth, recall_results),
        [len(result) / n_hists for result in recall_results],
    )

    if args.log_file:
        save_output(
            args.log_file,
            {
                "index": args.index,
                "index_size": get_index_size(index[0]),
                "queries": args.queries,
                "ground_truth": args.ground_truth,
                "precision_mode_metrics": precision_metrics,
                "recall_mode_metrics": recall_metrics,
                "precision_mode_time": precision_time,
                "recall_mode_time": recall_time,
            },
            name="metrics",
        )

    logger.info(f"Precision mode metrics: {np.array(precision_metrics).mean(axis=1)}")
    logger.info(f"Recall mode metrics: {np.array(recall_metrics).mean(axis=1)}")
    logger.info(f"Executed experiment in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
