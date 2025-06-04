"""
Module for parallel processing of histogram queries.
"""

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

from fainder.typing import Histogram, PercentileQuery
from fainder.utils import load_input


class FainderChunkLayout(StrEnum):
    CONTIGUOUS = auto()
    ROUND_ROBIN = auto()


class WorkerState:
    """Encapsulates the state for a worker process."""

    def __init__(self) -> None:
        self.hists: dict[int | np.integer[Any], Histogram] = {}  # Loaded histograms
        self.worker_id: int | None = None


# Process-local worker state
_worker_state: WorkerState = WorkerState()


def init_worker(worker_id: int, histogram_paths: list[Path]) -> None:
    """Initialize the worker process with its chunk of histograms.

    Args:
        worker_id: The ID of this worker process
        histogram_paths: List of paths to histogram files to load
    """
    global _worker_state
    _worker_state = WorkerState()  # Reset worker state for this process
    _worker_state.worker_id = worker_id

    # Load histograms from all assigned chunks
    for histogram_path in histogram_paths:
        hists: list[tuple[int | np.integer[Any], Histogram]] = load_input(
            histogram_path, "histograms"
        )
        if hists is None:
            logger.error(f"Worker {worker_id} failed to load histograms from {histogram_path}")
            continue

        # Merge into worker's histogram dictionary
        for id_, hist in hists:
            _worker_state.hists[id_] = hist

    logger.info(
        f"Worker {worker_id} initialized with {len(_worker_state.hists)}"
        f" histograms from {len(histogram_paths)} chunks"
    )


def process_hist_chunk(
    query: PercentileQuery, id_filter: NDArray[np.uint32]
) -> NDArray[np.uint32]:
    """Process the chunk of histograms assigned to this worker."""
    global _worker_state  # noqa: PLW0602
    from fainder.execution.percentile_queries import query_histogram

    if _worker_state.hists is None:
        logger.error("Worker called without being initialized!")
        return np.array([], dtype=np.uint32)

    # Filter histograms by ID
    filtered_hists: list[tuple[int | np.integer[Any], Histogram]] = [
        (id_f, _worker_state.hists[id_f]) for id_f in id_filter if id_f in _worker_state.hists
    ]

    # Process the histograms
    return np.fromiter(
        (
            np.uint32(id_)
            for id_, hist in filtered_hists
            if query_histogram(hist, estimation_mode="over", query=query, density=True)
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
        contiguous: If True, use contiguous chunks; if False, distribute in round-robin fashion

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
            if current_partition == num_partitions - 1:
                current_partition = 0
            else:
                current_partition += 1
    else:
        raise ValueError(f"Unsupported chunk layout: {chunk_layout}")

    return chunks


class ParallelHistogramProcessor:
    """Class for parallel processing of histogram queries."""

    def __init__(
        self,
        histogram_path: str | Path,
        num_workers: int = (os.cpu_count() or 2) - 1,
        num_chunks: int | None = None,
        chunk_layout: FainderChunkLayout = FainderChunkLayout.CONTIGUOUS,
    ) -> None:
        """Initialize the parallel processor with histograms.

        Args:
            histogram_path: Path to the histogram file or base file path for split files
            num_workers: Number of worker processes to use. Defaults to number of CPU cores - 1.
            num_chunks: Number of chunks to split the histograms into. If None, uses num_workers.
            contiguous: If True, use contiguous chunks of histograms;
                        if False, distribute in round-robin fashion
        """
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

        # Initialize executor with fork server method to avoid copying objects in memory
        mp_context = mp.get_context("forkserver")
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=mp_context,
        )

        # Register shutdown handler
        atexit.register(self.shutdown)

        logger.info(f"Initializing {self.num_workers} workers for parallel histogram processing")

        # Initialize workers immediately
        futures = []
        for worker_id, hist_paths in self._init_params:
            future = self.executor.submit(init_worker, worker_id, hist_paths)
            futures.append(future)

        # Wait for all workers to initialize
        for future in as_completed(futures):
            future.result()

        logger.info(f"Parallel histogram processor initialized with {self.num_workers} workers")

    def shutdown(self) -> None:
        """Shutdown the executor."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)
            logger.debug("ParallelHistogramProcessor shutdown complete")

    def query(self, query: PercentileQuery, id_filter: NDArray[np.uint32]) -> NDArray[np.uint32]:
        """Query histograms in parallel."""
        futures = [
            self.executor.submit(process_hist_chunk, query, id_filter)
            for _ in range(self.num_workers)
        ]

        combined_result = np.array([], dtype=np.uint32)
        for future in as_completed(futures):
            try:
                result = future.result()
                combined_result = np.concatenate((combined_result, result))
            except Exception as e:
                logger.error(f"Worker failed with exception: {e}")
                continue

        logger.info(f"Combined result size: {combined_result.size} from {len(futures)} workers")
        return combined_result
