import argparse
import os
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from fainder.utils import configure_run, save_output


def parse_args() -> argparse.Namespace:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    parser = argparse.ArgumentParser(
        description="Compute distribution moments from a collection of Parquet files.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="path to Parquet dataset collection",
        metavar="SRC",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="path for the compressed distribution output",
        metavar="DEST",
    )
    parser.add_argument(
        "-k",
        "--kind",
        type=str,
        choices=["normal"],
        required=True,
        help="distribution kind",
    )
    parser.add_argument(
        "-w",
        "--workers",
        default=os.cpu_count(),
        type=int,
        help="number of worker processes (default: %(default)s)",
    )
    parser.add_argument(
        "--log-file",
        type=lambda s: Path(os.path.expandvars(s)),
        default=Path(f"logs/hist_computation_{timestamp}.log"),
        help="path to log file (default: %(default)s)",
        metavar="LOG",
    )
    return parser.parse_args()


def compute_distribution(
    input_file: Path,
    kind: Literal["normal"],
) -> list[tuple[float, ...]] | str:
    try:
        np.seterr(all="raise")
        dists: list[tuple[float, ...]] = []
        df = pd.read_parquet(input_file, engine="pyarrow").select_dtypes(include="number")
        for _, values in df.items():  # noqa: PERF102
            values.dropna(inplace=True)
            # We filter out huge values to prevent overflows in the index (and since they
            # are unrealistic for percentile queries). Since multiple large integer values are
            # represented by the same float value, we cast them before counting unique values.
            values = values[(values > -(2**53)) & (values < 2**53)].astype(dtype=np.float64)  # noqa
            if values.nunique() > 1 and values.min() != values.max():
                if kind == "normal":
                    dist = (values.mean(), values.std())
                else:
                    raise ValueError(f"Invalid distribution kind {kind}.")

                dists.append(dist)
        return dists
    except AssertionError as e:
        raise AssertionError(input_file) from e
    except Exception as e:
        return f"{input_file}: {type(e)} {e}"


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    configure_run("INFO", args.log_file)
    logger.debug(vars(args))

    n_files = len(list(args.input.iterdir()))
    with Pool(processes=args.workers) as pool:
        fn = partial(
            compute_distribution,
            kind=args.kind,
        )
        results = pool.map(fn, args.input.iterdir())

    errors: list[str] = []
    dists: list[tuple[np.uint32, tuple[float, ...]]] = []
    i = 0
    for result in results:
        if isinstance(result, str):
            errors.append(result)
        else:
            for dist in result:
                dists.append((np.uint32(i), dist))
                i += 1

    save_output(args.output, dists, name="distributions")

    end = time.perf_counter()
    logger.info(
        f"Parsed {n_files} files and generated {i} distributions in {end - start:.2f}s"
        f" with {len(errors)} errors."
    )
    for error in errors:
        logger.debug(error)


if __name__ == "__main__":
    main()
