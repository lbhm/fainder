import pickle
import sys
from collections.abc import Container
from multiprocessing import set_start_method
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
import zstandard as zstd
from loguru import logger
from numpy.typing import ArrayLike

from fainder.typing import (
    ConversionIndex,
    F32Array,
    F64Array,
    FArray,
    Histogram,
    PercentileIndex,
    PercentileIndexPointer,
    PercentileQuery,
    RebinningIndex,
    UInt32Array,
)

ROUNDING_PRECISION = 4
T = TypeVar("T")


def filter_hists(
    hists: list[tuple[T, Histogram]],
    filter_ids: Container[T],
) -> list[tuple[T, Histogram]]:
    return [hist for hist in hists if hist[0] in filter_ids]


def filter_index(
    pctl_index: list[PercentileIndex],
    cluster_bins: list[F64Array],
    filter_ids: ArrayLike,
) -> tuple[list[PercentileIndex], list[F64Array]]:
    new_index: list[PercentileIndex] = []
    new_bins: list[F64Array] = []
    for i, cluster in enumerate(pctl_index):
        new_cluster: list[tuple[FArray, UInt32Array]] = []
        for pctls, ids in cluster:
            mask = np.isin(ids.reshape(-1, order="F"), filter_ids)
            new_pctls = np.require(
                pctls.reshape(-1, order="F")[mask].reshape((-1, pctls.shape[1]), order="F"),
                dtype=pctls.dtype,
                requirements="F",
            )
            new_ids = np.require(
                ids.reshape(-1, order="F")[mask].reshape((-1, ids.shape[1]), order="F"),
                dtype=ids.dtype,
                requirements="F",
            )
            new_cluster.append((new_pctls, new_ids))

        if mask.sum() > 0:
            new_index.append(tuple(new_cluster))  # type: ignore
            new_bins.append(cluster_bins[i])

    return new_index, new_bins


def filter_binsort(
    binsort: tuple[F64Array, tuple[F32Array, F32Array, F32Array], UInt32Array],
    filter_ids: ArrayLike,
) -> tuple[F64Array, tuple[F32Array, F32Array, F32Array], UInt32Array]:
    mask = np.isin(binsort[2], filter_ids)
    return (
        binsort[0][mask],
        (binsort[1][0][mask], binsort[1][1][mask], binsort[1][2][mask]),
        binsort[2][mask],
    )


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
    if len(truth) == 0:
        return 0.0, 1.0, 0.0
    if len(prediction) == 0:
        return 1.0, 0.0, 0.0
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
