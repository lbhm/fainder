import argparse
import os
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from fainder.typing import Histogram
from fainder.utils import configure_run, load_input, save_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a subset of IDs from a histogram collection.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="path to compressed histogram collection",
        metavar="SRC",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=lambda s: os.path.expandvars(s),
        required=True,
        help="path to output file",
        metavar="DEST",
    )
    parser.add_argument(
        "-s",
        "--selectivity",
        type=float,
        required=True,
        help="selectivity of the subset",
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
    parser.add_argument(
        "--log-file",
        type=lambda s: Path(os.path.expandvars(s)),
        default=None,
        help="path to log file (default: %(default)s)",
        metavar="LOG",
    )
    return parser.parse_args()


def filter_histograms(
    hists: list[tuple[np.uint32, Histogram]],
    selectivity: float,
    generator: np.random.Generator,
) -> NDArray[np.uint32]:
    if not 0 < selectivity <= 1:
        raise ValueError(f"Selectivity must be in the range (0, 1], got {selectivity}")

    ids, _ = zip(*hists, strict=True)
    return np.array(
        generator.choice(ids, size=int(len(ids) * selectivity), replace=False), dtype=np.uint32
    )


def main() -> None:
    args = parse_args()
    configure_run(args.log_level, args.log_file)
    logger.debug(vars(args))

    hists = load_input(args.input, name="histograms")
    ids = filter_histograms(hists, args.selectivity, np.random.default_rng(args.seed))

    logger.info(f"Selected {len(ids)} IDs from {len(hists)} histograms.")
    if len(ids) == 0:
        logger.warning("No IDs were selected.")
    else:
        logger.debug("Saving output")
        save_output(args.output, ids, "ID subset")


if __name__ == "__main__":
    main()
