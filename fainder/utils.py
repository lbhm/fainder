import pickle
import sys
from multiprocessing import set_start_method
from pathlib import Path
from typing import Any, Literal

import numpy as np
import zstandard as zstd
from loguru import logger

from fainder.typing import (
    ConversionIndex,
    F64Array,
    Histogram,
    PercentileIndexPointer,
    PercentileQuery,
    RebinningIndex,
)

ROUNDING_PRECISION = 4


def query_accuracy_metrics(
    truth: set[np.uint32], prediction: set[np.uint32]
) -> tuple[float, float, float]:
    """Compute precision, recall, and the F1-score for an approximate query result.

    Args:
        truth (set[np.uint32]): ground truth
        prediction (set[np.uint32]): predicted results

    Returns:
        tuple[float, float, float]: precision, recall, F1-score
    """
    if len(truth) == 0 and len(prediction) == 0:
        return 1.0, 1.0, 1.0
    elif len(truth) == 0:
        return 0.0, 1.0, 0.0
    elif len(prediction) == 0:
        return 1.0, 0.0, 0.0
    else:
        tp = len(truth & prediction)
        fp = len(prediction - truth)
        fn = len(truth - prediction)

        return tp / (fp + tp), tp / (fn + tp), 2 * tp / (2 * tp + fp + fn)


def collection_accuracy_metrics(
    truth: list[set[np.uint32]], prediction: list[set[np.uint32]]
) -> tuple[list[float], list[float], list[float]]:
    assert len(truth) == len(prediction)
    precision = []
    recall = []
    f1_score = []

    for i in range(len(truth)):
        p, r, f = query_accuracy_metrics(truth[i], prediction[i])
        precision.append(p)
        recall.append(r)
        f1_score.append(f)

    return precision, recall, f1_score


def parse_percentile_query(args: list[str]) -> PercentileQuery:
    assert len(args) == 3

    percentile = float(args[0])
    assert 0 < percentile <= 1

    reference = float(args[2])

    assert args[1] in ["ge", "gt", "le", "lt"]
    comparison: Literal["le", "lt", "ge", "gt"] = args[1]  # type: ignore

    return percentile, comparison, reference


def save_output(
    path: Path | str, data: Any, name: str | None = "output", threads: int | None = None
) -> None:
    if isinstance(path, str):
        path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.with_suffix(".zst")

    cctx = None
    if threads:
        cctx = zstd.ZstdCompressor(threads=threads)

    with zstd.open(path, "wb", cctx=cctx) as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    if name:
        logger.debug(f"Saved {name} to {path}")


def load_input(path: Path | str, name: str | None = "input") -> Any:
    if name:
        logger.debug(f"Loading {name} from {path}")
    with zstd.open(path, "rb") as file:
        return pickle.load(file)


def configure_run(
    stdout_log_level: str, log_file: str | None = None, start_method: str = "spawn"
) -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | <level>{message}</level>",
        level=stdout_log_level,
    )
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7.7} | {message}",
            level="TRACE",
            mode="w",
        )

    try:
        set_start_method(start_method)
    except RuntimeError:
        logger.debug("Start context already set, ignoring")


def unlink_pointers(shm_pointers: list[PercentileIndexPointer]) -> None:
    for cluster_pointers in shm_pointers:
        for pctl_pointer, id_pointer in cluster_pointers:
            pctl_pointer.unlink()
            id_pointer.unlink()


def get_index_size(index: list[ConversionIndex] | list[RebinningIndex]) -> int:
    size = 0
    for cluster in index:
        for pctls, ids in cluster:
            size += pctls.nbytes + ids.nbytes
    return size


def predict_index_size(
    clustered_hists: list[list[tuple[np.uint32, Histogram]]], cluster_bins: list[F64Array]
) -> int:
    size = 0
    for i in range(len(clustered_hists)):
        size += len(clustered_hists[i]) * len(cluster_bins[i]) * 6
    return size
