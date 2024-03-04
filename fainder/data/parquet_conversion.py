import argparse
import os
import time
from functools import partial
from itertools import filterfalse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from loguru import logger

from fainder.utils import configure_run


def parse_args() -> argparse.Namespace:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    parser = argparse.ArgumentParser(
        description="Convert a collection of CSV files to Parquet.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="path to CSV dataset collection",
        metavar="SRC",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="path to output directory",
        metavar="DEST",
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
        default=Path(f"logs/parquet_conversion_{timestamp}.log"),
        help="path to log file (default: %(default)s)",
        metavar="LOG",
    )
    return parser.parse_args()


def to_parquet(input_file: Path, output_dir: Path) -> str | None:
    try:
        df = pd.read_csv(input_file, low_memory=False)
        df.to_parquet(output_dir / f"{input_file.stem}.pq", engine="pyarrow", index=False)
        return None
    except Exception as e:
        return f"{input_file}: {e}"


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    configure_run("INFO", args.log_file)
    logger.debug(vars(args))

    args.output.mkdir(parents=True, exist_ok=True)
    with Pool(processes=args.workers) as pool:
        fn = partial(to_parquet, output_dir=args.output)
        logs = pool.map(fn, args.input.iterdir())
    logs = list(filterfalse(lambda i: not i, logs))
    end = time.perf_counter()

    logger.info(
        f"Converted {len(list(args.input.iterdir()))} files in {end - start:.2f}s with"
        f" {len(logs)} log entries."
    )
    for log in logs:
        logger.debug(log)


if __name__ == "__main__":
    main()
