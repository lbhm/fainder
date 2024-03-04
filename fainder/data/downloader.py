import argparse
import io
import os
import time
import zipfile
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import requests
from loguru import logger

from fainder.utils import configure_run

GITTABLES_ZENODO_ID = 6517052


def download_gittables(output: Path, workers: int, zenodo_id: int = GITTABLES_ZENODO_ID) -> int:
    response = requests.get(f"https://zenodo.org/api/records/{zenodo_id}")
    if response.status_code == 200:
        file_infos: list[dict[str, Any]] = response.json()["files"]
        with Pool(processes=workers) as pool:
            fn = partial(download_zenodo_file, output=output)
            results = pool.map(fn, file_infos)

        sum = 0
        for result in results:
            if isinstance(result, str):
                logger.debug(result)
            else:
                sum += result
        return sum
    else:
        logger.error(f"Dataset request failed (response code: {response.status_code})")
        return 0


def download_zenodo_file(file_info: dict[str, Any], output: Path) -> str | int:
    response = requests.get(file_info["links"]["self"])
    if response.status_code == 200:
        if zipfile.is_zipfile(io.BytesIO(response.content)):
            with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_file:
                i = 0
                for i, info in enumerate(zip_file.infolist()):
                    info.filename = f"{Path(file_info['key']).stem}_{i}.pq"
                    zip_file.extract(info, output / "pq")
                    logger.trace(f"Extracted {info.filename} to {output / 'pq'}")
            return i + 1
        else:
            return f"File {file_info['key']} from {file_info['links']['self']} is not a zip file"
    else:
        return f"Failed to download {file_info['key']} from {file_info['links']['self']}"


def parse_args() -> argparse.Namespace:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    parser = argparse.ArgumentParser(
        description="Download and unzip a collection of parquet files from Zenodo.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--collection",
        type=str,
        choices=["gittables"],
        required=True,
        help="dataset collection to download",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="path to store the datasets",
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
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO"],
        default="INFO",
        help="logging level (default: %(default)s)",
    )
    parser.add_argument(
        "--log-file",
        type=lambda s: Path(os.path.expandvars(s)),
        default=Path(f"logs/data_download_{timestamp}.log"),
        help="path to log file (default: %(default)s)",
        metavar="LOG",
    )

    return parser.parse_args()


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    configure_run(args.log_level, args.log_file)

    (args.output / "pq").mkdir(parents=True, exist_ok=True)
    n_files = 0
    if args.collection == "gittables":
        n_files = download_gittables(args.output, args.workers)
    else:
        logger.error(f"Unknown dataset collection: {args.collection}")

    logger.info(f"Downloaded {n_files} files in {time.perf_counter() - start:.2f} seconds")


if __name__ == "__main__":
    main()
