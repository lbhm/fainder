import argparse
import os
import time
from pathlib import Path

import numpy as np
from loguru import logger

from fainder.execution import runner
from fainder.utils import collection_accuracy_metrics, configure_run, load_input, save_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the accuracy of a binsort index given a query set and a ground truth."
        ),
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--index",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="Path to the binsort index file",
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

    return parser.parse_args()


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    configure_run(args.log_level, args.log_file.with_suffix(".log") if args.log_file else None)

    binsort = load_input(args.index, name="binsort index")
    queries = load_input(args.queries, name="queries")
    ground_truth = load_input(args.ground_truth, name="ground truth")

    binsort_results, binsort_time = runner.run(
        input_data=binsort,
        queries=queries,
        input_type="binsort",
        workers=args.workers,
    )

    binsort_metrics = (
        *collection_accuracy_metrics(ground_truth, binsort_results),
        [-1 for _ in binsort_results],  # We cannot properly compute the pruning factor for binsort
    )

    if args.log_file:
        save_output(
            args.log_file,
            {
                "binsort": args.index,
                "queries": args.queries,
                "ground_truth": args.ground_truth,
                "binsort_metrics": binsort_metrics,
                "binsort_time": binsort_time,
            },
            name="metrics",
        )

    logger.info(f"Binsort metrics: {np.array(binsort_metrics).mean(axis=1)}")
    logger.info(f"Executed experiment in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
