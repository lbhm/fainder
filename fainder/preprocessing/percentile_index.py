import argparse
import os
import time
from collections.abc import Sequence
from functools import partial
from multiprocessing import Pool, Value
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger
from scipy.interpolate import CubicSpline

from fainder.typing import (
    BoolArray,
    ConversionIndex,
    F32Array,
    F64Array,
    FArray,
    Histogram,
    PercentileIndex,
    PercentileIndexPointer,
    RebinningIndex,
    UInt32Array,
)
from fainder.utils import (
    ROUNDING_PRECISION,
    configure_run,
    get_index_size,
    load_input,
    save_output,
    unlink_pointers,
)

counter: Any


def rebin_histogram(
    hist: Histogram,
    new_bins: F64Array,
    bin_approximation: Literal["continuous_value", "cubic_spline"] = "continuous_value",
) -> F32Array:
    """Re-bin a histogram based on global bins and an intra-bin value approximation."""
    new_vals = np.zeros(shape=new_bins.size - 1, dtype=np.float32)

    if bin_approximation == "cubic_spline":
        bin_mids = np.hstack((hist[1][0], hist[1][:-1] + np.diff(hist[1]) / 2, hist[1][-1]))
        bin_heights = np.hstack((0, hist[0], 0))
        spline = CubicSpline(x=bin_mids, y=bin_heights, bc_type="clamped")

    for i, old_val in enumerate(hist[0]):
        old_bin = (hist[1][i], hist[1][i + 1])
        new_bins_start = np.searchsorted(new_bins, old_bin[0], side="right") - 1
        new_bins_end = np.searchsorted(new_bins, old_bin[1], side="left")

        # Due to rounding errors, the old bin edges could lie outside of the cluster bin edges
        if new_bins_start == -1:
            if not np.isclose(new_bins[0], old_bin[0], atol=5.0e-8, rtol=1e-5):
                logger.warning(
                    f"Old bin edge {old_bin[0]} lies below the cluster bins "
                    f"(deviation: {np.abs(old_bin[0], new_bins[0]):.10g})"
                )
            new_bins_start += 1
            old_bin = (new_bins[0], hist[1][i + 1])
        if new_bins_end == len(new_bins):
            if not np.isclose(new_bins[-1], old_bin[1], atol=5.0e-8, rtol=1e-5):
                logger.warning(
                    f"Old bin edge {old_bin[1]} lies above the cluster bins "
                    f"(deviation: {np.abs(old_bin[1], new_bins[-1]):.10g})"
                )
            new_bins_end -= 1
            old_bin = (hist[1][i], new_bins[-1])

        for j in range(new_bins_start, new_bins_end):
            new_bin = (new_bins[j], new_bins[j + 1])

            overlap = (max(old_bin[0], new_bin[0]), min(old_bin[1], new_bin[1]))
            if overlap == old_bin:
                new_vals[j] += old_val
            elif overlap[0] > overlap[1]:
                raise ValueError("No overlap between old and new bin.")
            elif bin_approximation == "continuous_value":
                # Continuous-value assumption
                fraction = (overlap[1] - overlap[0]) / (old_bin[1] - old_bin[0])
                new_vals[j] += fraction * old_val
            elif bin_approximation == "cubic_spline":
                # Cubic spline interpolation between bins
                fraction = spline.integrate(overlap[0], overlap[1]) / spline.integrate(  # type: ignore
                    old_bin[0], old_bin[1]
                )
                new_vals[j] += fraction * old_val
            else:
                raise ValueError("Invalid bin_approximation method.")

    return new_vals


def rebin_collection(
    clustered_hists: list[list[tuple[np.uint32, Histogram]]],
    cluster_bins: list[F64Array],
    bin_approximation: Literal["continuous_value", "cubic_spline"],
    n_workers: int,
) -> list[tuple[UInt32Array, F32Array]]:
    rebinned_hists: list[tuple[UInt32Array, F32Array]] = []
    with Pool(processes=n_workers) as pool:
        for i in range(len(clustered_hists)):
            ids, hists = zip(*clustered_hists[i], strict=True)
            fn = partial(
                rebin_histogram, new_bins=cluster_bins[i], bin_approximation=bin_approximation
            )
            rebinned_hists.append((np.array(ids, dtype=np.uint32), np.array(pool.map(fn, hists))))

    return rebinned_hists


def convert_bins(old_bins: F64Array, new_bins: F64Array) -> BoolArray:
    conversion_matrix = np.zeros(shape=(len(old_bins) - 1, len(new_bins) - 1), dtype=np.bool_)
    for i in range(len(old_bins) - 1):
        start_bin = np.searchsorted(new_bins, old_bins[i], side="right") - 1
        end_bin = np.searchsorted(new_bins, old_bins[i + 1], side="left") - 1
        conversion_matrix[i, start_bin : end_bin + 1] = True

    return conversion_matrix


def convert_bin_collection(
    clustered_hists: list[list[tuple[np.uint32, Histogram]]],
    cluster_bins: list[F64Array],
    n_workers: int,
) -> list[list[BoolArray]]:
    conversion_matrices: list[list[BoolArray]] = []
    with Pool(processes=n_workers) as pool:
        for i in range(len(clustered_hists)):
            _, hists = zip(*clustered_hists[i], strict=True)
            fn = partial(convert_bins, new_bins=cluster_bins[i])
            conversion_matrices.append(pool.map(fn, [hist[1] for hist in hists]))

    return conversion_matrices


def create_rebinning_index(
    rebinned_hists: list[tuple[UInt32Array, F32Array]],
    cluster_bins: list[F64Array],
    dtype: np.dtype[Any],
) -> list[RebinningIndex]:
    pctl_index: list[RebinningIndex] = []
    for i in range(len(rebinned_hists)):
        n_hists = len(rebinned_hists[i][0])
        n_bins = len(cluster_bins[i])
        pctls = np.zeros((n_hists, n_bins), dtype=dtype, order="F")
        ids = np.zeros((n_hists, n_bins), dtype=np.uint32, order="F")

        pctls[:, 1:] = np.round(np.cumsum(rebinned_hists[i][1], axis=1), ROUNDING_PRECISION)
        ids[:] = rebinned_hists[i][0].reshape((-1, 1))

        sort_indices = np.argsort(pctls, axis=0, kind="stable")
        pctls[:] = np.take_along_axis(pctls, sort_indices, axis=0)
        ids[:] = np.take_along_axis(ids, sort_indices, axis=0)
        pctl_index.append(((pctls, ids),))

    return pctl_index


def create_conversion_index(
    clustered_hists: list[list[tuple[np.uint32, Histogram]]],
    cluster_bins: list[F64Array],
    conversion_matrices: list[list[BoolArray]],
    dtype: np.dtype[Any],
) -> list[ConversionIndex]:
    pctl_index: list[ConversionIndex] = []
    for i in range(len(clustered_hists)):
        n_hists = len(clustered_hists[i])
        n_bins = len(cluster_bins[i])
        lower_pctls = np.zeros((n_hists, n_bins - 1), dtype=dtype, order="F")
        lower_ids = np.zeros((n_hists, n_bins - 1), dtype=np.uint32, order="F")
        upper_pctls = np.zeros((n_hists, n_bins - 1), dtype=dtype, order="F")
        upper_ids = np.zeros((n_hists, n_bins - 1), dtype=np.uint32, order="F")

        # NOTE: We could parallelize across histograms here
        for j in range(n_hists):
            id_, hist = clustered_hists[i][j]

            mask = np.invert(
                np.cumsum(conversion_matrices[i][j][:, ::-1], axis=1, dtype=np.bool_)[:, ::-1]
            )
            lower_pctls[j, :] = np.round(
                np.where(mask, np.tile(hist[0], (mask.shape[1], 1)).T, 0).sum(axis=0),
                ROUNDING_PRECISION,
            )
            lower_ids[j, :] = id_
            mask = conversion_matrices[i][j].cumsum(axis=1, dtype=np.bool_)
            upper_pctls[j, :] = np.round(
                np.where(mask, np.tile(hist[0], (mask.shape[1], 1)).T, 0).sum(axis=0),
                ROUNDING_PRECISION,
            )
            upper_ids[j, :] = id_

        sort_indices = np.argsort(lower_pctls, axis=0, kind="stable")
        lower_pctls[:] = np.take_along_axis(lower_pctls, sort_indices, axis=0)
        lower_ids[:] = np.take_along_axis(lower_ids, sort_indices, axis=0)
        sort_indices = np.argsort(upper_pctls, axis=0, kind="stable")
        upper_pctls[:] = np.take_along_axis(upper_pctls, sort_indices, axis=0)
        upper_ids[:] = np.take_along_axis(upper_ids, sort_indices, axis=0)
        pctl_index.append(((lower_pctls, lower_ids), (upper_pctls, upper_ids)))

    return pctl_index


def init_index_workers(pool_counter: Any) -> None:
    """Initialize each worker with a global synchronized counter."""
    global counter
    counter = pool_counter


def init_rebinning_shm_index(
    clustered_hists: list[list[tuple[np.uint32, Histogram]]],
    cluster_bins: list[F64Array],
    dtype: np.dtype[Any],
) -> tuple[list[RebinningIndex], list[PercentileIndexPointer]]:
    pctl_index: list[RebinningIndex] = []
    shm_pointers: list[PercentileIndexPointer] = []
    for i in range(len(clustered_hists)):
        n_hists = len(clustered_hists[i])
        n_bins = len(cluster_bins[i])
        pctls_dummy = np.zeros((n_hists, n_bins), dtype=dtype, order="F")
        ids_dummy = np.zeros((n_hists, n_bins), dtype=np.uint32, order="F")

        pctls_shm = SharedMemory(create=True, size=pctls_dummy.nbytes)
        ids_shm = SharedMemory(create=True, size=ids_dummy.nbytes)

        pctls: FArray = np.ndarray(
            pctls_dummy.shape, dtype=pctls_dummy.dtype, order="F", buffer=pctls_shm.buf
        )
        ids: UInt32Array = np.ndarray(
            ids_dummy.shape, dtype=ids_dummy.dtype, order="F", buffer=ids_shm.buf
        )

        # Need to fill lower_pctls with zeros to account for the first row
        pctls[:] = pctls_dummy

        pctl_index.append(((pctls, ids),))
        shm_pointers.append(((pctls_shm, ids_shm),))

    return pctl_index, shm_pointers


def init_conversion_shm_index(
    clustered_hists: list[list[tuple[np.uint32, Histogram]]],
    cluster_bins: list[F64Array],
    dtype: np.dtype[Any],
) -> tuple[list[ConversionIndex], list[PercentileIndexPointer]]:
    pctl_index: list[ConversionIndex] = []
    shm_pointers: list[PercentileIndexPointer] = []
    for i in range(len(clustered_hists)):
        n_hists = len(clustered_hists[i])
        n_bins = len(cluster_bins[i])
        pctls_dummy = np.zeros((n_hists, n_bins - 1), dtype=dtype, order="F")
        ids_dummy = np.zeros((n_hists, n_bins - 1), dtype=np.uint32, order="F")

        lower_pctls_shm = SharedMemory(create=True, size=pctls_dummy.nbytes)
        lower_ids_shm = SharedMemory(create=True, size=ids_dummy.nbytes)
        upper_pctls_shm = SharedMemory(create=True, size=pctls_dummy.nbytes)
        upper_ids_shm = SharedMemory(create=True, size=ids_dummy.nbytes)

        lower_pctls: FArray = np.ndarray(
            pctls_dummy.shape, dtype=pctls_dummy.dtype, order="F", buffer=lower_pctls_shm.buf
        )
        lower_ids: UInt32Array = np.ndarray(
            ids_dummy.shape, dtype=ids_dummy.dtype, order="F", buffer=lower_ids_shm.buf
        )
        upper_pctls: FArray = np.ndarray(
            pctls_dummy.shape, dtype=pctls_dummy.dtype, order="F", buffer=upper_pctls_shm.buf
        )
        upper_ids: UInt32Array = np.ndarray(
            ids_dummy.shape, dtype=ids_dummy.dtype, order="F", buffer=upper_ids_shm.buf
        )

        # Need to fill lower_pctls with zeros to account for the first row
        lower_pctls[:] = pctls_dummy

        pctl_index.append(((lower_pctls, lower_ids), (upper_pctls, upper_ids)))
        shm_pointers.append(((lower_pctls_shm, lower_ids_shm), (upper_pctls_shm, upper_ids_shm)))

    return pctl_index, shm_pointers


def load_shm_index(pctl_index: list[PercentileIndex]) -> list[PercentileIndexPointer]:
    shm_pointers: list[PercentileIndexPointer] = []
    for cluster in pctl_index:
        pointers: list[tuple[SharedMemory, SharedMemory]] = []
        for pctls, ids in cluster:
            pctls_shm = SharedMemory(create=True, size=pctls.nbytes)
            ids_shm = SharedMemory(create=True, size=ids.nbytes)

            pctls_array: FArray = np.ndarray(
                pctls.shape, dtype=pctls.dtype, order="F", buffer=pctls_shm.buf
            )
            ids_array: UInt32Array = np.ndarray(
                ids.shape, dtype=ids.dtype, order="F", buffer=ids_shm.buf
            )

            pctls_array[:] = pctls
            ids_array[:] = ids

            pointers.append((pctls_shm, ids_shm))
        shm_pointers.append(tuple(pointers))

    return shm_pointers


def shm_rebinning_worker(
    hist: tuple[np.uint32, Histogram],
    shm_pointer: PercentileIndexPointer,
    new_bins: F64Array,
    n_hists: int,
    n_bins: int,
    bin_approximation: Literal["continuous_value", "cubic_spline"],
    dtype: np.dtype[Any],
) -> None:
    pctls: FArray = np.ndarray(
        (n_hists, n_bins), dtype=dtype, order="F", buffer=shm_pointer[0][0].buf
    )
    ids: UInt32Array = np.ndarray(
        (n_hists, n_bins), dtype=np.uint32, order="F", buffer=shm_pointer[0][1].buf
    )
    id_, hist_ = hist
    new_vals = np.zeros(shape=new_bins.size - 1, dtype=np.float32)

    if bin_approximation == "cubic_spline":
        bin_mids = np.hstack((hist_[1][0], hist_[1][:-1] + np.diff(hist_[1]) / 2, hist_[1][-1]))
        bin_heights = np.hstack((0, hist_[0], 0))
        spline = CubicSpline(x=bin_mids, y=bin_heights, bc_type="clamped")

    for i, old_val in enumerate(hist_[0]):
        old_bin = (hist_[1][i], hist_[1][i + 1])
        new_bins_start = np.searchsorted(new_bins, old_bin[0], side="right") - 1
        new_bins_end = np.searchsorted(new_bins, old_bin[1], side="left")

        for j in range(new_bins_start, new_bins_end):
            new_bin = (new_bins[j], new_bins[j + 1])

            overlap = (max(old_bin[0], new_bin[0]), min(old_bin[1], new_bin[1]))
            if overlap == old_bin:
                new_vals[j] += old_val
            elif overlap[0] > overlap[1]:
                raise ValueError("No overlap between old and new bin.")
            elif bin_approximation == "continuous_value":
                fraction = (overlap[1] - overlap[0]) / (old_bin[1] - old_bin[0])
                new_vals[j] += fraction * old_val
            elif bin_approximation == "cubic_spline":
                fraction = spline.integrate(overlap[0], overlap[1]) / spline.integrate(  # type: ignore
                    old_bin[0], old_bin[1]
                )
                new_vals[j] += fraction * old_val
            else:
                raise ValueError("Invalid bin_approximation method.")

    percentile_row = np.round(np.cumsum(new_vals), ROUNDING_PRECISION)
    with counter.get_lock():
        pctls[counter.value, 1:] = percentile_row
        ids[counter.value, :] = id_
        counter.value += 1


def create_shm_rebinning_index(
    pctl_index: list[RebinningIndex],
    shm_pointers: list[PercentileIndexPointer],
    clustered_hists: list[list[tuple[np.uint32, Histogram]]],
    cluster_bins: list[F64Array],
    bin_approximation: Literal["continuous_value", "cubic_spline"],
    n_workers: int,
    dtype: np.dtype[Any],
) -> None:
    counter = Value("i", 0)
    with Pool(processes=n_workers, initializer=init_index_workers, initargs=(counter,)) as pool:
        for i in range(len(clustered_hists)):
            counter.value = 0  # type: ignore
            n_hists = len(clustered_hists[i])
            n_bins = len(cluster_bins[i])
            fn = partial(
                shm_rebinning_worker,
                shm_pointer=shm_pointers[i],
                new_bins=cluster_bins[i],
                n_hists=n_hists,
                n_bins=n_bins,
                bin_approximation=bin_approximation,
                dtype=dtype,
            )
            pool.map(fn, clustered_hists[i])

            sort_indices = np.argsort(pctl_index[i][0][0], axis=0, kind="stable")
            pctl_index[i][0][0][:] = np.take_along_axis(pctl_index[i][0][0], sort_indices, axis=0)
            pctl_index[i][0][1][:] = np.take_along_axis(pctl_index[i][0][1], sort_indices, axis=0)


def shm_conversion_worker(
    hist: tuple[np.uint32, Histogram],
    shm_pointer: PercentileIndexPointer,
    new_bins: F64Array,
    n_hists: int,
    n_bins: int,
    dtype: np.dtype[Any],
) -> None:
    lower_pctls: FArray = np.ndarray(
        (n_hists, n_bins - 1), dtype=dtype, order="F", buffer=shm_pointer[0][0].buf
    )
    lower_ids: UInt32Array = np.ndarray(
        (n_hists, n_bins - 1), dtype=np.uint32, order="F", buffer=shm_pointer[0][1].buf
    )
    upper_pctls: FArray = np.ndarray(
        (n_hists, n_bins - 1), dtype=dtype, order="F", buffer=shm_pointer[1][0].buf
    )
    upper_ids: UInt32Array = np.ndarray(
        (n_hists, n_bins - 1), dtype=np.uint32, order="F", buffer=shm_pointer[1][1].buf
    )
    id_, (values, old_bins) = hist
    conversion_matrix = np.zeros(shape=(len(old_bins) - 1, len(new_bins) - 1), dtype=np.bool_)

    for i in range(len(old_bins) - 1):
        start_bin = np.searchsorted(new_bins, old_bins[i], side="right") - 1
        end_bin = np.searchsorted(new_bins, old_bins[i + 1], side="left") - 1
        conversion_matrix[i, start_bin : end_bin + 1] = True

    mask = np.invert(np.cumsum(conversion_matrix[:, ::-1], axis=1, dtype=np.bool_)[:, ::-1])
    lower_pctl_row = np.round(
        np.where(mask, np.tile(values, (mask.shape[1], 1)).T, 0).sum(axis=0), ROUNDING_PRECISION
    )
    mask = conversion_matrix.cumsum(axis=1, dtype=np.bool_)
    upper_pctl_row = np.round(
        np.where(mask, np.tile(values, (mask.shape[1], 1)).T, 0).sum(axis=0), ROUNDING_PRECISION
    )

    with counter.get_lock():
        lower_pctls[counter.value, :] = lower_pctl_row
        lower_ids[counter.value, :] = id_
        upper_pctls[counter.value, :] = upper_pctl_row
        upper_ids[counter.value, :] = id_
        counter.value += 1


def create_shm_conversion_index(
    pctl_index: list[ConversionIndex],
    shm_pointers: list[PercentileIndexPointer],
    clustered_hists: list[list[tuple[np.uint32, Histogram]]],
    cluster_bins: list[F64Array],
    n_workers: int,
    dtype: np.dtype[Any],
) -> None:
    counter = Value("i", 0)
    with Pool(processes=n_workers, initializer=init_index_workers, initargs=(counter,)) as pool:
        for i in range(len(clustered_hists)):
            counter.value = 0  # type: ignore
            n_hists = len(clustered_hists[i])
            n_bins = len(cluster_bins[i])
            fn = partial(
                shm_conversion_worker,
                shm_pointer=shm_pointers[i],
                new_bins=cluster_bins[i],
                n_hists=n_hists,
                n_bins=n_bins,
                dtype=dtype,
            )
            pool.map(fn, clustered_hists[i])

            sort_indices = np.argsort(pctl_index[i][0][0], axis=0, kind="stable")
            pctl_index[i][0][0][:] = np.take_along_axis(pctl_index[i][0][0], sort_indices, axis=0)
            pctl_index[i][0][1][:] = np.take_along_axis(pctl_index[i][0][1], sort_indices, axis=0)

            sort_indices = np.argsort(pctl_index[i][1][0], axis=0, kind="stable")
            pctl_index[i][1][0][:] = np.take_along_axis(pctl_index[i][1][0], sort_indices, axis=0)
            pctl_index[i][1][1][:] = np.take_along_axis(pctl_index[i][1][1], sort_indices, axis=0)


def parse_args() -> argparse.Namespace:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    parser = argparse.ArgumentParser(
        description="Construct an index over heterogeneous histograms.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="path to clustered histogram collection",
        metavar="SRC",
    )

    # Index parameters
    parser.add_argument(
        "-m",
        "--index-method",
        default="rebinning",
        type=str,
        choices=["rebinning", "rebinning-shm", "conversion", "conversion-shm"],
        help="histogram alignment technique (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--bin-estimation",
        default="continuous_value",
        type=str,
        choices=["continuous_value", "cubic_spline"],
        help="intra-bin estimation approach, only applies to rebinning (default: %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--index-precision",
        default="float16",
        type=str,
        choices=["float16", "float32", "float64", "float128"],
        help="precision of the index (default: %(default)s)",
    )

    # Output options
    parser.add_argument(
        "-o",
        "--output-dir",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="output directory path",
        metavar="DEST",
    )
    parser.add_argument(
        "--index-file",
        default=f"index_{timestamp}.zst",
        type=str,
        help="file name of the generated index (default: %(default)s)",
    )
    parser.add_argument(
        "--intermediate-output",
        default=None,
        type=str,
        help="file to store intermediate results (default: %(default)s)",
    )

    # Misc
    parser.add_argument(
        "-w",
        "--workers",
        default=os.cpu_count(),
        type=int,
        help="number of worker processes (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        choices=["DEBUG", "INFO"],
        help="verbosity of STDOUT logs (default: %(default)s)",
    )
    parser.add_argument(
        "--log-file",
        type=lambda s: Path(os.path.expandvars(s)),
        default=Path(f"logs/index_creation_{timestamp}.log"),
        help="path to log file (default: %(default)s)",
        metavar="LOG",
    )

    return parser.parse_args()


def create_index(
    clustered_hists: list[list[tuple[np.uint32, Histogram]]],
    cluster_bins: list[F64Array],
    index_method: Literal["rebinning", "rebinning-shm", "conversion", "conversion-shm"],
    index_precision: Literal["float16", "float32", "float64", "float128"] = "float32",
    bin_estimation: Literal["continuous_value", "cubic_spline"] = "continuous_value",
    workers: int | None = 1,
) -> tuple[Sequence[PercentileIndex], Any, list[PercentileIndexPointer] | None]:
    start = time.perf_counter()

    if "conversion" in index_method:
        logger.debug("Selected conversion method, ignoring bin estimation choice")
    if "shm" in index_method:
        logger.debug("The SHM indexing methods do not support storing intermediate results")
    if workers is None or workers < 1:
        raise ValueError("Workers must be a positive integer.")

    index_dtype = np.dtype(index_precision)
    shm_pointers: list[PercentileIndexPointer] | None = []
    pctl_index: Sequence[PercentileIndex] = []
    intermediates: Any
    if "rebinning" in index_method:
        if index_method == "rebinning":
            logger.debug("Starting rebinning")
            rebinned_hists = rebin_collection(
                clustered_hists, cluster_bins, bin_estimation, workers
            )
            logger.debug("Starting index creation")
            pctl_index = create_rebinning_index(rebinned_hists, cluster_bins, index_dtype)
            shm_pointers = None
            intermediates = (rebinned_hists, cluster_bins)

        elif index_method == "rebinning-shm":
            logger.debug("Initializing SHM index")
            pctl_index, shm_pointers = init_rebinning_shm_index(
                clustered_hists, cluster_bins, index_dtype
            )
            logger.debug("Starting index creation")
            create_shm_rebinning_index(
                pctl_index,
                shm_pointers,
                clustered_hists,
                cluster_bins,
                bin_estimation,
                workers,
                index_dtype,
            )
            intermediates = None
        else:
            raise ValueError("Invalid index method.")

        logger.debug("Veryfying index")
        # Verify that the cumsum for each histogram starts at 0 and adds up to 1
        # We use a higher rounding tolerance due to floating point precision issues
        for cluster in pctl_index:
            if not np.allclose(cluster[0][0][:, 0], 0, rtol=0):
                deviations = np.isclose(cluster[0][0][:, 0], 0, rtol=0)
                logger.warning(
                    f"{np.sum(deviations)} percentiles do not start at 0 "
                    f"(max deviation: {np.max(deviations):.10g})"
                )
            # NOTE: Rebinning can cause larger rounding errors so the check is more lenient
            if not np.allclose(cluster[0][0][:, -1], 1, atol=10**-ROUNDING_PRECISION, rtol=0):
                deviations = np.isclose(
                    cluster[0][0][:, -1], 1, atol=10**-ROUNDING_PRECISION, rtol=0
                )
                logger.warning(
                    f"{np.sum(deviations)} percentiles do not add up to 1 "
                    f"(max deviation: {np.max(deviations):.10g})"
                )

        index_size = get_index_size(pctl_index)
        logger.debug(f"Index size: {index_size / 1000**2:.2f} MB")
        logger.trace(f"index_size, {index_size / 1000**2}")
    elif "conversion" in index_method:
        if index_method == "conversion":
            logger.debug("Starting conversion")
            conversion_matrices = convert_bin_collection(clustered_hists, cluster_bins, workers)
            logger.debug("Starting index creation")
            pctl_index = create_conversion_index(
                clustered_hists, cluster_bins, conversion_matrices, index_dtype
            )
            shm_pointers = None
            intermediates = (clustered_hists, conversion_matrices, cluster_bins)
        elif index_method == "conversion-shm":
            logger.debug("Initializing SHM index")
            pctl_index, shm_pointers = init_conversion_shm_index(
                clustered_hists, cluster_bins, index_dtype
            )
            logger.debug("Starting index creation")
            create_shm_conversion_index(
                pctl_index, shm_pointers, clustered_hists, cluster_bins, workers, index_dtype
            )
            intermediates = None
        else:
            raise ValueError("Invalid index method.")

        logger.debug("Veryfying index")
        for (lower_pctls, _), (upper_pctls, _) in pctl_index:
            if not np.allclose(lower_pctls[:, 0], 0, rtol=0):
                deviations = np.isclose(lower_pctls[:, 0], 0, rtol=0)
                logger.warning(
                    f"{np.sum(deviations)} percentiles do not start at 0 "
                    f"(max deviation: {np.max(deviations):.10g})"
                )
            if not np.allclose(upper_pctls[:, -1], 1, rtol=0):
                deviations = np.isclose(upper_pctls[:, -1], 1, rtol=0)
                logger.warning(
                    f"{np.sum(deviations)} percentiles do not end at 1 "
                    f"(max deviation: {np.max(deviations):.10g})"
                )

        index_size = get_index_size(pctl_index)
        logger.debug(f"Index size: {index_size / 1000**2:.2f} MB")
        logger.trace(f"index_size, {index_size / 1000**2}")
    else:
        raise ValueError("Invalid index method.")

    end = time.perf_counter()
    logger.info(f"Created {index_method}-based index in {end - start:.2f}s")
    logger.trace(f"total_time, {end - start}")

    return pctl_index, intermediates, shm_pointers


def main() -> None:
    args = parse_args()
    configure_run(args.log_level, args.log_file)
    logger.debug(vars(args))

    clustered_hists, cluster_bins = load_input(args.input, name="histogram clusters")
    shm_pointers = None
    try:
        pctl_index, intermediates, shm_pointers = create_index(
            clustered_hists,
            cluster_bins,
            args.index_method,
            args.index_precision,
            args.bin_estimation,
            args.workers,
        )
    finally:
        if shm_pointers is not None:
            unlink_pointers(shm_pointers)

    save_output(args.output_dir / args.index_file, (pctl_index, cluster_bins), name="index")
    if args.intermediate_output and intermediates is not None:
        save_output(
            args.output_dir / args.intermediate_output, intermediates, name="intermediate results"
        )


if __name__ == "__main__":
    main()
