import argparse
import os
import time
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from fainder.typing import PercentileQuery
from fainder.utils import ROUNDING_PRECISION, configure_run, save_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a collection of queries with percentile predicates.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="output file for the generated queries",
        metavar="DEST",
    )
    parser.add_argument(
        "--n-percentiles",
        default=20,
        type=int,
        help="number of unique percentile values (default: %(default)s)",
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        default=["gt", "lt"],
        type=str,
        choices=["ge", "gt", "le", "lt"],
        help="list of operators (default: %(default)s)",
    )
    parser.add_argument(
        "--n-reference-values",
        default=20,
        type=int,
        help="number of references values to draw (default: %(default)s)",
    )
    parser.add_argument(
        "--reference-value-range",
        nargs=2,
        default=[-30, 150],
        type=int,
        help="open interval of values to draw from (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="random seed (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        choices=["INFO", "DEBUG"],
        help="verbosity of STDOUT logs (default: %(default)s)",
    )

    return parser.parse_args()


def generate_queries(
    n_percentiles: int,
    operators: list[Literal["ge", "gt", "le", "lt"]],
    n_reference_values: int,
    reference_value_range: tuple[float, float],
    seed: int,
) -> list[PercentileQuery]:
    if n_percentiles > 1000:
        logger.warning(
            f"n_percentiles too high, values with more than {ROUNDING_PRECISION} decimals will be"
            " rounded"
        )
    percentiles = np.round(np.linspace(0, 1, n_percentiles + 1)[1:], decimals=ROUNDING_PRECISION)
    references = np.round(
        np.random.default_rng(seed).uniform(*reference_value_range, size=n_reference_values),
        decimals=4,
    )
    return list(product(percentiles, operators, references))


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    configure_run("INFO")
    logger.debug(vars(args))

    queries = generate_queries(
        args.n_percentiles,
        args.operators,
        args.n_reference_values,
        args.reference_value_range,
        args.seed,
    )

    save_output(args.output, queries, name="queries")

    end = time.perf_counter()
    logger.info(f"Generated {len(queries)} queries in {end - start:.2f}s.")


if __name__ == "main":
    main()
