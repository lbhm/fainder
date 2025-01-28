import argparse
import os
import random
import subprocess
import time
from itertools import compress
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from fainder.execution import runner
from fainder.utils import configure_run, load_input, save_output
from fainder.validation import query_metrics

if TYPE_CHECKING:
    import numpy as np

    from fainder.typing import Histogram, PercentileQuery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate different query categories for the accuracy benchmark.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["sportstables", "open_data_usa", "gittables"],
        help="The dataset to use for the experiment.",
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
        "-W",
        "--workload",
        default="accuracy_benchmark",
        type=str,
        choices=["accuracy_benchmark", "llm"],
        help="type of query workload to compute results for",
    )
    parser.add_argument(
        "-c",
        "--clustering",
        default=None,
        type=lambda s: Path(os.path.expandvars(s)),
        help="Path to a clustering",
        metavar="SRC",
    )
    parser.add_argument(
        "-v",
        "--n-val-queries",
        default=100,
        type=int,
        help="number of valdiation queries per category to generate (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--n-test-queries",
        default=333,
        type=int,
        help="number of test queries per category to generate (default: %(default)s)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        default=None,
        type=int,
        help="number of worker processes (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="random seed to use (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        choices=["DEBUG", "INFO"],
        help="verbosity of STDOUT logs (default: %(default)s)",
    )

    return parser.parse_args()


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    configure_run(args.log_level)
    os.chdir(
        subprocess.run(
            "git rev-parse --show-toplevel", shell=True, check=True, stdout=subprocess.PIPE
        )
        .stdout.decode("utf-8")
        .strip()
    )
    random.seed(args.seed)

    histograms_path = Path(f"data/{args.dataset}/histograms.zst")
    result_path = Path(f"data/{args.dataset}/results/{args.workload}")
    query_path = Path(f"data/{args.dataset}/queries/{args.workload}")

    histograms: list[tuple[np.uint32, Histogram]] = load_input(histograms_path, "histograms")
    queries: list[PercentileQuery] = load_input(args.queries, "queries")

    ground_truth, _ = runner.run(
        input_data=histograms,
        queries=queries,
        input_type="histograms",
        estimation_mode="over",
        workers=args.workers,
    )

    selectivity = query_metrics.selectivity(
        ground_truth=ground_truth,
        collection_size=len(histograms),
        workers=args.workers,
    )
    logger.info("Computed selectivity metrics")

    if args.clustering:
        _, cluster_bins = load_input(args.clustering, "clustering")
        cluster_hits = query_metrics.cluster_hits(
            queries=queries,
            cluster_bins=cluster_bins,
            workers=args.workers,
        )
        logger.info("Computed cluster hits metrics")
    else:
        cluster_hits = None

    bin_edge_matches = query_metrics.bin_edge_matches(
        queries=queries,
        hists=histograms,
        workers=args.workers,
    )
    logger.info("Computed bin edge matches metrics")

    save_output(
        args.queries.with_stem(args.queries.stem + "-metrics"),
        (selectivity, cluster_hits, bin_edge_matches),
        "query metrics",
    )
    save_output(result_path / "ground_truth-all.zst", ground_truth, "ground truth")

    if args.workload == "accuracy_benchmark":
        val_queries: list[PercentileQuery] = []
        test_queries: list[PercentileQuery] = []
        test_results: list[set[np.uint32]] = []
        for category, selector in [
            ("low_selectivity", selectivity < 0.1),
            ("mid_selectivity", (selectivity > 0.1) & (selectivity < 0.9)),
            ("high_selectivity", selectivity > 0.9),
        ]:
            query_subset = list(compress(queries, selector))
            result_subset = list(compress(ground_truth, selector))
            n_queries: int = args.n_val_queries + args.n_test_queries
            if len(query_subset) > n_queries:
                ids = random.sample(range(len(query_subset)), n_queries)
                query_subset = [query for i, query in enumerate(query_subset) if i in ids]
                result_subset = [result for i, result in enumerate(result_subset) if i in ids]

                val_queries += query_subset[: args.n_val_queries]
                test_queries += query_subset[args.n_val_queries :]
                test_results += result_subset[args.n_val_queries :]

                save_output(
                    query_path / f"val-{category}.zst",
                    query_subset[: args.n_val_queries],
                    f"{category} val queries",
                )
                save_output(
                    query_path / f"test-{category}.zst",
                    query_subset[args.n_val_queries :],
                    f"{category} test queries",
                )
                save_output(
                    result_path / f"ground_truth-val-{category}.zst",
                    result_subset[: args.n_val_queries],
                    f"{category} val truth",
                )
                save_output(
                    result_path / f"ground_truth-test-{category}.zst",
                    result_subset[args.n_val_queries :],
                    f"{category} test truth",
                )
            else:
                logger.error(
                    f"Only {len(query_subset)} queries with {category} available, "
                    f"but {n_queries} requested."
                )

        save_output(query_path / "val-all.zst", val_queries, "val queries")
        save_output(query_path / "test-all.zst", test_queries, "test queries")
        save_output(result_path / "ground_truth-test-all.zst", test_results, "test results")

    logger.info(f"Executed experiment in {time.perf_counter() - start:.2f}s.")


if __name__ == "__main__":
    main()
