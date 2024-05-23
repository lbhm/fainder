import argparse
import os
import time
from pathlib import Path

import numpy as np
from loguru import logger

from fainder.typing import F32Array, F64Array, Histogram, UInt32Array
from fainder.utils import ROUNDING_PRECISION, configure_run, load_input, save_output


def parse_args() -> argparse.Namespace:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    parser = argparse.ArgumentParser(
        description="Compute a binsort index for a collection of histograms.",
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
        "--log-level",
        default="INFO",
        type=str,
        choices=["INFO", "DEBUG"],
        help="verbosity of STDOUT logs (default: %(default)s)",
    )
    parser.add_argument(
        "--log-file",
        type=lambda s: Path(os.path.expandvars(s)),
        default=Path(f"logs/clustering_{timestamp}.log"),
        help="path to log file (default: %(default)s)",
        metavar="LOG",
    )

    return parser.parse_args()


def compute_binsort(
    hists: list[tuple[np.uint32, Histogram]],
) -> tuple[F64Array, tuple[F32Array, F32Array, F32Array], UInt32Array]:
    """Compute the binsort representation of a histogram collection."""
    edges_list: list[F64Array] = []
    pre_pctls_list: list[F32Array] = []
    mid_pctls_list: list[F32Array] = []
    post_pctls_list: list[F32Array] = []
    ids_list: list[UInt32Array] = []

    for id_, (values, bins) in hists:
        pctls = values.cumsum(dtype=np.float32).flatten()
        pctls[:] = np.round(pctls, ROUNDING_PRECISION)
        edges_list.append(bins)
        pre_pctls_list.append(np.concatenate((np.zeros(2), pctls[:-1]), dtype=np.float32))
        mid_pctls_list.append(np.insert(pctls, 0, 0))
        post_pctls_list.append(np.append(pctls, 1))
        ids_list.append(np.full_like(bins, id_, dtype=np.uint32))

    edges = np.concatenate(edges_list, axis=None, dtype=np.float64)
    pre_pctls = np.concatenate(pre_pctls_list, axis=None, dtype=np.float32)
    mid_pctls = np.concatenate(mid_pctls_list, axis=None, dtype=np.float32)
    post_pctls = np.concatenate(post_pctls_list, axis=None, dtype=np.float32)
    ids = np.concatenate(ids_list, axis=None, dtype=np.uint32)

    sort_indices = np.argsort(edges, axis=None)
    edges = edges[sort_indices]
    pre_pctls = pre_pctls[sort_indices]
    mid_pctls = mid_pctls[sort_indices]
    post_pctls = post_pctls[sort_indices]
    ids = ids[sort_indices]

    return edges, (pre_pctls, mid_pctls, post_pctls), ids


def main() -> None:
    args = parse_args()
    configure_run(args.log_level, args.log_file)
    logger.debug(vars(args))

    binsort = compute_binsort(
        load_input(args.input, name="histograms"),
    )

    logger.debug("Saving output")
    save_output(args.output, binsort, "binsort")


if __name__ == "__main__":
    main()
