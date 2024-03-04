import argparse
import os
import time
from pathlib import Path

import numpy as np
from loguru import logger

from fainder.typing import F64Array, PercentileIndex
from fainder.utils import configure_run, load_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assert the equality of two independently constructed indices.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "index1",
        type=lambda s: Path(os.path.expandvars(s)),
        help="path to index 1",
        metavar="FILE",
    )
    parser.add_argument(
        "index2",
        type=lambda s: Path(os.path.expandvars(s)),
        help="path to index 2",
        metavar="FILE",
    )

    return parser.parse_args()


def check_equality(
    index_1: tuple[list[PercentileIndex], list[F64Array]],
    index_2: tuple[list[PercentileIndex], list[F64Array]],
) -> bool:
    assert len(index_1[0]) == len(
        index_2[0]
    ), f"Uneven index length: {len(index_1[0])} and {len(index_1[0])}."

    for i in range(len(index_1[0])):
        assert np.allclose(index_1[1][i], index_2[1][i]), "Index cluster bins are not equal."
        for j in range(len(index_1[0][i])):
            assert np.allclose(
                index_1[0][i][j][0], index_1[0][i][j][0]
            ), f"Percentiles {j} in cluster {i} not equal."

    return True


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    configure_run("INFO")

    index_1 = load_input(args.index1, name="index 1")
    index_2 = load_input(args.index2, name="index 2")
    try:
        _ = check_equality(index_1, index_2)
        end = time.perf_counter()
        logger.info(f"Asserted index equality in {end - start:.2f}s.")
    except AssertionError as e:
        logger.warning(f"Assertion failed: {e}")


if __name__ == "__main__":
    main()
