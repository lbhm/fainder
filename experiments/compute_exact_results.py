import argparse
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from fainder.execution import runner, runner_exact
from fainder.utils import configure_run, load_input, save_output

if TYPE_CHECKING:
    import numpy as np

    from fainder.typing import F32Array, F64Array, Histogram, PercentileQuery, UInt32Array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute exact results for a set of queries based on index preprocessing.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--data",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="Path to a histogram collection or binsort index",
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
        "-i",
        "--index",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="Path to a (conversion-based) index",
        metavar="SRC",
    )
    parser.add_argument(
        "-e",
        "--exact-method",
        default="pscan",
        type=str,
        choices=["pscan", "binsort"],
        help="exact method to use (default: %(default)s)",
    )
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="do not compute ground truth results",
    )
    parser.add_argument(
        "--no-sym-difference",
        action="store_true",
        help="do not compute symmetric difference (to save memory)",
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
    configure_run(args.log_level)

    queries: list[PercentileQuery] = load_input(args.queries, name="queries")
    pctl_index, cluster_bins = load_input(args.index, name="index")

    precision_results, precision_time = runner.run(
        input_data=(pctl_index, cluster_bins),
        queries=queries,
        input_type="index",
        index_mode="precision",
        workers=args.workers,
    )
    recall_results, recall_time = runner.run(
        input_data=(pctl_index, cluster_bins),
        queries=queries,
        input_type="index",
        index_mode="recall",
        workers=args.workers,
    )

    if args.exact_method == "pscan":
        histograms: list[tuple[np.uint32, Histogram]] = load_input(args.data, name="histograms")
        exact_results, iterative_time, avg_reduction = runner_exact.run_pscan_collection(
            hists=histograms,
            precision_results=precision_results,
            recall_results=recall_results,
            queries=queries,
            estimation_mode="over",
            workers=args.workers,
        )
    elif args.exact_method == "binsort":
        binsort: tuple[F64Array, tuple[F32Array, F32Array, F32Array], UInt32Array] = load_input(
            args.data, name="binsort"
        )
        exact_results, iterative_time, avg_reduction = runner_exact.run_binsort_collection(
            binsort=binsort,
            precision_results=precision_results,
            recall_results=recall_results,
            queries=queries,
            index_mode="recall",
            workers=args.workers,
        )
    else:
        raise ValueError(f"Invalid exact method: {args.exact_method}")

    avg_sym_difference = -1.0
    if not args.no_ground_truth and args.exact_method == "pscan":
        if args.no_sym_difference:
            del precision_results
            del recall_results
            del exact_results

        ground_truth, baseline_time = runner.run(
            input_data=histograms,
            queries=queries,
            input_type="histograms",
            estimation_mode="over",
            workers=args.workers,
        )

        if not args.no_sym_difference:
            avg_sym_difference = 0.0
            for j in range(len(queries)):
                if exact_results[j] != ground_truth[j]:  # type: ignore
                    avg_sym_difference += len(exact_results[j] ^ ground_truth[j])  # type: ignore
            avg_sym_difference /= len(queries)
    else:
        baseline_time = -1.0

    if args.log_file:
        save_output(
            args.log_file,
            {
                "data": args.data,
                "precision_time": precision_time,
                "recall_time": recall_time,
                "iterative_time": iterative_time,
                "baseline_time": baseline_time,
                "avg_reduction": avg_reduction,
                "avg_sym_difference": avg_sym_difference,
            },
            name="exact results",
        )

    logger.info(f"Precision time: {precision_time:.4g}s")
    logger.info(f"Recall time: {recall_time:.4g}s")
    logger.info(f"Iterative time: {iterative_time:.4g}s")
    logger.info(f"Average reduction: {avg_reduction:.2f}")
    logger.info(f"Average symmetric difference: {avg_sym_difference:.6g}")
    logger.info(f"Executed experiment in {time.perf_counter() - start:.2f}s.")


if __name__ == "__main__":
    main()
