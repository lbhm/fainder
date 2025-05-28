"""
Module for parallel processing of histogram queries.
"""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from fainder.typing import Histogram, PercentileQuery
from fainder.utils import load_input


class WorkerState:
    """Encapsulates the state for a worker process."""

    def __init__(self) -> None:
        self.hists: list[tuple[int | np.integer[Any], Histogram]] | None = None
        self.worker_id: int | None = None
        self.contiguous: bool = False  # Whether to use contiguous chunks
        self.id_map: dict[int | np.integer[Any], int] = {}  # for round-robin distribution
        self.start_idx: int | np.integer[Any] = 0  # Start index for contiguous chunks
        self.end_idx: int | np.integer[Any] = 0  # End index for contiguous chunks


# Process-local worker state
_worker_state: WorkerState = WorkerState()


def init_worker(worker_id: int, histogram_path: str | Path, contiguous: bool) -> None:
    """Initialize the worker process with its chunk of histograms.

    Args:
        worker_id: The ID of this worker process
        histogram_path: Path to the histogram file or base directory for split files
    """
    global _worker_state
    _worker_state = WorkerState()  # Reset worker state for this process
    _worker_state.worker_id = worker_id
    _worker_state.contiguous = contiguous

    _worker_state.hists = load_input(histogram_path, "histograms")
    if _worker_state.hists is None:
        logger.error(f"Worker {worker_id} failed to load histograms from {histogram_path}")
        return

    if contiguous:
        _worker_state.start_idx = _worker_state.hists[0][
            0
        ]  # Use the first histogram ID as start index
        _worker_state.end_idx = _worker_state.hists[-1][
            0
        ]  # Use the last histogram ID as end index
    else:
        # Build ID mapping
        _worker_state.id_map = {}
        for i, (hist_id, _) in enumerate(_worker_state.hists):
            _worker_state.id_map[hist_id] = i

        logger.debug(f"Worker {worker_id} initialized with {len(_worker_state.hists)} histograms")


def process_hist_chunk(
    query: PercentileQuery, id_filter: NDArray[np.uint32] | None = None
) -> NDArray[np.uint32]:
    """Process the chunk of histograms assigned to this worker."""
    global _worker_state
    from fainder.execution.percentile_queries import query_histogram

    if _worker_state.hists is None:
        logger.error("Worker called without being initialized!")
        return np.array([], dtype=np.uint32)

    # Filter histograms by ID if needed
    filtered_hists = _worker_state.hists
    if id_filter is not None:
        filtered_hists = []
        for id_f in id_filter:
            if _worker_state.contiguous:
                if _worker_state.start_idx <= id_f <= _worker_state.end_idx:
                    filtered_hists.append(_worker_state.hists[id_f - _worker_state.start_idx])
            else:
                if id_f in _worker_state.id_map:
                    filtered_hists.append(_worker_state.hists[_worker_state.id_map[id_f]])
        if not filtered_hists:
            logger.debug(
                f"Worker {_worker_state.worker_id} found "
                f"no histograms matching the filter {id_filter}"
            )
            return np.array([], dtype=np.uint32)

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
    hist_ids: list[int], num_partitions: int, contiguous: bool = False
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

    if contiguous:
        # Original contiguous chunking strategy
        chunk_size = len(hist_ids) // num_partitions
        remainder = len(hist_ids) % num_partitions

        start_idx = 0
        for i in range(num_partitions):
            end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
            chunks[i] = set(hist_ids[start_idx:end_idx])
            start_idx = end_idx
    else:
        # Round-robin distribution for more balanced workload
        current_partition = 0
        for hist_id in hist_ids:
            chunks[current_partition].add(hist_id)
            if current_partition == num_partitions - 1:
                current_partition = 0
            else:
                current_partition += 1

    return chunks


class ParallelHistogramProcessor:
    """Class for parallel processing of histogram queries."""

    def __init__(
        self, histogram_path: str | Path, num_workers: int | None = None, contiguous: bool = False
    ) -> None:
        """Initialize the parallel processor with histograms.

        Args:
            histogram_path: Path to the histogram file or base file path for split files
            num_workers: Number of worker processes to use. If None, uses CPU count - 1.
            contiguous: If True, use contiguous chunks of histograms;
                        if False, distribute in round-robin fashion
        """
        self.num_workers = (num_workers or os.cpu_count() or 2) - 1
        self.histogram_path = histogram_path
        self.contiguous = contiguous

        parent_path = (
            Path(histogram_path).parent
            if isinstance(histogram_path, str)
            else histogram_path.parent
        )

        # Store initialization parameters
        self._init_params = []
        for i in range(self.num_workers):
            if contiguous:
                hist_path = (
                    parent_path
                    / f"histograms_split_contiguous_{self.num_workers + 1}"
                    / f"histograms_{i}.zst"
                )
            else:
                hist_path = (
                    parent_path
                    / f"histograms_split_round_robin_{self.num_workers + 1}"
                    / f"histograms_{i}.zst"
                )
            self._init_params.append((i, hist_path))

        # Initialize executor with fork start method to avoid copying objects in memory
        mp_context = mp.get_context("fork")
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=self._init_worker_wrapper,
            initargs=(),
            mp_context=mp_context,
        )

        logger.info(f"Initializing {self.num_workers} workers for parallel histogram processing")

        # Initialize workers immediately
        futures = []
        for worker_id, hist_path in self._init_params:
            future = self.executor.submit(init_worker, worker_id, hist_path, self.contiguous)
            futures.append(future)

        # Wait for all workers to initialize
        for future in futures:
            future.result()

        logger.info(f"Parallel histogram processor initialized with {self.num_workers} workers")

    def _init_worker_wrapper(self) -> None:
        """Wrapper to initialize worker with proper parameters."""
        # This will be called once per worker process

    def query(
        self, query: PercentileQuery, id_filter: NDArray[np.uint32] | None = None
    ) -> NDArray[np.uint32]:
        """Query histograms in parallel."""
        futures = [
            self.executor.submit(process_hist_chunk, query, id_filter)
            for _ in range(self.num_workers)
        ]

        # Collect results from all workers
        results = [future.result() for future in futures]

        # Combine results from all workers
        combined_result = (
            np.concatenate([r for r in results if r.size > 0], axis=None)
            if results
            else np.array([], dtype=np.uint32)
        )
        logger.info(f"Combined result size: {combined_result.size} from {len(results)} workers")
        return combined_result

    def shutdown(self) -> None:
        """Shutdown the executor."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

    def __del__(self) -> None:
        """Ensure the executor is properly shutdown on deletion."""
        self.shutdown()
        logger.debug("ParallelHistogramProcessor shutdown complete")
