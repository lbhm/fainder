import argparse
import os
import time
from pathlib import Path

import numpy as np
from loguru import logger

from fainder.execution import runner
from fainder.utils import configure_run, load_input, save_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the accuracy of a normal distribution estimator given a query set and a"
            " ground truth."
        ),
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-H",
        "--histograms",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="Path to a histogram collection",
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

    hists = load_input(args.histograms, name="histograms")
    queries = load_input(args.queries, name="queries")

    ground_truth, hist_time = runner.run(
        input_data=hists,
        queries=queries,
        input_type="histograms",
        estimation_mode="over",
        workers=args.workers,
    )
    # NOTE: We do not need to store the ground truth again because we already did so

    n_hists = len(hists)
    logger.debug(f"Collection contains {n_hists} histograms")
    dummy_metric = [1 for _ in range(len(queries))]
    hist_metrics = (
        dummy_metric,
        dummy_metric,
        dummy_metric,
        [len(result) / n_hists for result in ground_truth],
    )

    if args.log_file:
        save_output(
            args.log_file,
            {
                "histograms": args.histograms,
                "queries": args.queries,
                "hist_metrics": hist_metrics,
                "hist_time": hist_time,
            },
            name="metrics",
        )

    logger.info(f"Hist metrics: {np.array(hist_metrics).mean(axis=1)}")
    logger.info(f"Executed experiment in {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
