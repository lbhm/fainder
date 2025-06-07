import atexit
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import StrEnum, auto
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from fainder.execution.percentile_queries import query_histogram
from fainder.typing import Histogram, PercentileQuery
from fainder.utils import load_input


class FainderChunkLayout(StrEnum):
    CONTIGUOUS = auto()
    ROUND_ROBIN = auto()


class WorkerState:
    def __init__(self) -> None:
        self.hists: dict[int | np.integer[Any], Histogram] = {}
        self.worker_id: int | None = None


_worker_state: WorkerState = WorkerState()


def init_worker(worker_id: int, histogram_paths: list[Path]) -> tuple[int, NDArray[np.uint32]]:
    global _worker_state
    _worker_state = WorkerState()
    _worker_state.worker_id = worker_id
    hist_ids: list[int] = []
    for histogram_path in histogram_paths:
        hists = load_input(histogram_path, "histograms")
        if hists is None:
            logger.error(f"Worker {worker_id} failed to load histograms from {histogram_path}")
            continue
        for id_, hist in hists:
            _worker_state.hists[id_] = hist
            hist_ids.append(id_)

    logger.info(f"Worker {worker_id} initialized with {len(_worker_state.hists)} histograms")

    return worker_id, np.array(hist_ids, dtype=np.uint32)


def process_hist_chunk(
    query: PercentileQuery, id_filter: NDArray[np.uint32]
) -> NDArray[np.uint32]:
    global _worker_state

    if _worker_state.hists is None:
        logger.error("Worker called without being initialized!")
        return np.array([], dtype=np.uint32)

    return np.fromiter(
        (
            np.uint32(id_)
            for id_ in id_filter
            if query_histogram(
                _worker_state.hists[id_], estimation_mode="over", query=query, density=True
            )
        ),
        dtype=np.uint32,
    )


def partition_histogram_ids(
    hist_ids: list[int],
    num_partitions: int,
    chunk_layout: FainderChunkLayout = FainderChunkLayout.CONTIGUOUS,
) -> dict[int, set[int]]:
    """Partition histogram IDs into roughly equal chunks.

    Args:
        hist_ids: List of histogram IDs to partition
        num_partitions: Number of partitions to create
        chunk_layout: Layout strategy for partitioning

    Returns:
        Dictionary mapping partition ID to list of histogram IDs
    """
    chunks: dict[int, set[int]] = {i: set() for i in range(num_partitions)}

    if chunk_layout == FainderChunkLayout.CONTIGUOUS:
        # Original contiguous chunking strategy
        chunk_size = len(hist_ids) // num_partitions
        remainder = len(hist_ids) % num_partitions

        start_idx = 0
        for i in range(num_partitions):
            end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
            chunks[i] = set(hist_ids[start_idx:end_idx])
            start_idx = end_idx
    elif chunk_layout == FainderChunkLayout.ROUND_ROBIN:
        # Round-robin distribution for more balanced workload
        current_partition = 0
        for hist_id in hist_ids:
            chunks[current_partition].add(hist_id)
            current_partition = (current_partition + 1) % num_partitions
    else:
        raise ValueError(f"Unsupported chunk layout: {chunk_layout}")

    return chunks


class ParallelHistogramProcessor:
    def __init__(
        self,
        histogram_path: str | Path,
        num_workers: int = (os.cpu_count() or 2) - 1,
        num_chunks: int | None = None,
        chunk_layout: FainderChunkLayout = FainderChunkLayout.CONTIGUOUS,
    ) -> None:
        self.num_workers = num_workers
        self.histogram_path = histogram_path
        self.num_chunks = num_chunks or self.num_workers

        parent_path = (
            Path(histogram_path).parent
            if isinstance(histogram_path, str)
            else histogram_path.parent
        )

        # Distribute chunks among workers
        chunks_per_worker = self.num_chunks // self.num_workers
        remaining_chunks = self.num_chunks % self.num_workers

        # Store initialization parameters with multiple chunks per worker
        self._init_params = []
        chunk_idx = 0

        for worker_id in range(self.num_workers):
            # Calculate how many chunks this worker gets
            worker_chunk_count = chunks_per_worker + (1 if worker_id < remaining_chunks else 0)

            # Collect histogram paths for this worker
            worker_hist_paths = []
            for _ in range(worker_chunk_count):
                if chunk_layout == FainderChunkLayout.CONTIGUOUS:
                    hist_path = (
                        parent_path
                        / f"histograms_split_contiguous_{self.num_chunks}"
                        / f"histograms_{chunk_idx}.zst"
                    )
                elif chunk_layout == FainderChunkLayout.ROUND_ROBIN:
                    hist_path = (
                        parent_path
                        / f"histograms_split_round_robin_{self.num_chunks}"
                        / f"histograms_{chunk_idx}.zst"
                    )
                else:
                    raise ValueError(f"Unsupported chunk layout: {chunk_layout}")
                worker_hist_paths.append(hist_path)
                chunk_idx += 1

            self._init_params.append((worker_id, worker_hist_paths))

        mp_context = mp.get_context("forkserver")
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers, mp_context=mp_context)
        atexit.register(self.shutdown)

        logger.info(f"Initializing {self.num_workers} workers")
        futures = [
            self.executor.submit(init_worker, worker_id, hist_paths)
            for worker_id, hist_paths in self._init_params
        ]

        self.worker_partitions: dict[int, NDArray[np.uint32]] = {}
        # Collect results from worker initialization
        for future in as_completed(futures):
            worker_id, hist_ids = future.result()
            self.worker_partitions[worker_id] = hist_ids

        logger.info("ParallelHistogramProcessor initialized")

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)
        logger.debug("ParallelHistogramProcessor shutdown complete")

    def query(self, query: PercentileQuery, id_filter: NDArray[np.uint32]) -> NDArray[np.uint32]:
        # Partition using the worker partitions
        partition_filter = [
            np.intersect1d(id_filter, worker_ids) for worker_ids in self.worker_partitions.values()
        ]
        futures = [
            self.executor.submit(process_hist_chunk, query, part) for part in partition_filter
        ]

        combined_results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                combined_results.append(result)
            except Exception as e:
                logger.error(f"Worker failed with exception: {e}")

        return (
            np.concatenate(combined_results) if combined_results else np.array([], dtype=np.uint32)
        )
