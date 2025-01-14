import argparse
import os
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, MiniBatchKMeans  # type: ignore
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from fainder.typing import F64Array, Histogram
from fainder.utils import (
    ROUNDING_PRECISION,
    configure_run,
    load_input,
    predict_index_size,
    save_output,
)


def compute_features(
    hists: list[tuple[np.uint32, Histogram]],
    transform: Literal["standard", "robust", "quantile", "power"] | None,
    quantile_range: tuple[float, float] | None = (0.25, 0.75),
    seed: int | None = None,
) -> F64Array:
    features = np.zeros((len(hists), 3), dtype=np.float64)
    for id_, hist in hists:
        # Clipping is necessary to avoid invalid values during the clustering
        features[id_] = np.clip(
            (hist[1].min(), hist[1].max(), np.diff(hist[1]).mean()), -1e31, 1e31
        )
    if transform == "standard":
        features = StandardScaler().fit_transform(features)
    elif transform == "robust" and quantile_range:
        features = RobustScaler(quantile_range=quantile_range).fit_transform(features)
    elif transform == "quantile":
        features = QuantileTransformer(
            n_quantiles=min(10000, len(hist)), subsample=100000, random_state=seed
        ).fit_transform(features)
    elif transform == "power":
        features = PowerTransformer().fit_transform(features)
    else:
        pass

    return features


def compute_agglomerative_clustering(
    n_clusters: int, features: NDArray[np.float64]
) -> tuple[NDArray[np.int32], float]:
    # NOTE: Linkage methods other than single have a prohibitive memory footprint for large
    # datasets
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="euclidean",
        linkage="single",
    )
    clustering = clusterer.fit_predict(features)
    score = calinski_harabasz_score(features, clustering)
    return clustering, float(score)


def compute_hdbscan_clustering(
    n_clusters: int, features: NDArray[np.float64], n_jobs: int
) -> tuple[NDArray[np.int32], float]:
    clusterer = HDBSCAN(
        min_cluster_size=5,
        min_samples=1,
        metric="euclidean",
        n_jobs=n_jobs,
    )
    clustering = clusterer.fit_predict(features)
    score = calinski_harabasz_score(features, clustering)
    return clustering, float(score)


def compute_kmeans_clustering(
    n_clusters: int,
    features: NDArray[np.float64],
    seed: int | None,
    n_threads: int,
    verbose: bool = False,
) -> tuple[NDArray[np.int32], float]:
    clusterer = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=max(256 * n_threads, 1024),
        verbose=verbose,
        random_state=seed,
        n_init="auto",
    )
    clustering = clusterer.fit_predict(features)
    # NOTE: The silhouette score or Davies-Bouldin index could also be used instead
    score = calinski_harabasz_score(features, clustering)
    return clustering, float(score)


def compute_clustering(
    algorithm: Literal["agglomerative", "hdbscan", "kmeans"],
    features: F64Array,
    n_clusters: tuple[int, int],
    seed: int | None,
    n_workers: int | None,
    verbose: bool = False,
) -> NDArray[np.int32]:
    if n_clusters == (1, 1):
        return np.zeros(len(features), dtype=np.int32)
    if n_clusters[0] == 1:
        logger.warning(
            "Lower bound for n_cluster is set to 1. To create an index without clustering, you"
            " must pass n_cluster_range=(1,1) since a clustering cannot be compared with no"
            " clustering. Raising the lower limit to 2."
        )
        n_clusters = (2, n_clusters[1])

    cluster_range = range(n_clusters[0], n_clusters[1] + 1)
    n_jobs = len(cluster_range)
    if n_workers is None:
        n_workers = 1
    elif n_workers < 1:
        raise ValueError("Number of workers must be at least 1")

    if algorithm == "agglomerative":
        fn = partial(compute_agglomerative_clustering, features=features)
    elif algorithm == "hdbscan":
        fn = partial(compute_hdbscan_clustering, features=features, n_jobs=n_workers)
        logger.debug("Ignoring n_cluster_range parameter for HDBSCAN")
        n_jobs = 1
    elif algorithm == "kmeans":
        fn = partial(
            compute_kmeans_clustering,
            features=features,
            seed=seed,
            n_threads=int(n_workers / n_jobs),
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")

    if n_jobs == 1:
        clusterings = [fn(n_clusters[0])]
    else:
        with Pool(processes=min(n_workers, n_jobs)) as pool:
            clusterings = pool.map(fn, cluster_range)

    for i, (_, score) in enumerate(clusterings):
        logger.debug(f"Score for {i + n_clusters[0]:>2} clusters: {score}")

    return max(clusterings, key=lambda t: t[1])[0]


def assign_histograms(
    hists: list[tuple[np.uint32, Histogram]],
    clustering: NDArray[np.int32],
) -> list[list[tuple[np.uint32, Histogram]]]:
    clustered_hists: list[list[tuple[np.uint32, Histogram]]] = []
    hash_map: dict[int, int] = {}
    for i, id_ in enumerate(np.unique(clustering)):
        clustered_hists.append([])
        hash_map[id_] = i
    for i, (id_, hist) in enumerate(hists):
        clustered_hists[hash_map[clustering[i]]].append((id_, hist))

    return clustered_hists


def compute_cluster_bins(
    hists: list[list[tuple[np.uint32, Histogram]]],
    n_global_bins: int,
    alpha: float = 0.0,
) -> list[F64Array]:
    n_clusters = len(hists)
    min_bins = np.zeros(n_clusters, dtype=np.float64)
    max_bins = np.zeros(n_clusters, dtype=np.float64)
    cluster_bins: list[F64Array] = []
    n_hists = sum([len(cluster) for cluster in hists])
    assert alpha >= 0.0

    for i, cluster in enumerate(hists):
        min_max = np.array([(hist[1].min(), hist[1].max()) for _, hist in cluster])
        min_bins[i] = min_max[:, 0].min()
        max_bins[i] = min_max[:, 1].max()

        if alpha == np.inf:
            budget_share = 1 / n_clusters
        else:
            budget_share = (len(cluster) + alpha) / (n_hists + alpha * n_clusters)

        cluster_bins.append(
            np.round(
                np.linspace(  # type: ignore
                    start=min_bins[i],
                    stop=max_bins[i],
                    num=max(int(budget_share * n_global_bins), 1) + 1,
                    retstep=False,
                    dtype=np.float64,
                ),
                ROUNDING_PRECISION,
            )
        )
        try:
            assert np.allclose(cluster_bins[-1][0], min_bins[i], atol=1e-4)
            assert np.allclose(cluster_bins[-1][-1], max_bins[i], atol=1e-4)
        except AssertionError:
            logger.warning(f"Bins for cluster {i} do not match the histogram min/max values")
            logger.warning(f"min_bin: {min_bins[i]}, max_bin: {max_bins[i]}")
            logger.warning(f"cluster bin range: {cluster_bins[-1][0]}, {cluster_bins[-1][-1]}")

    return cluster_bins


def parse_args() -> argparse.Namespace:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    parser = argparse.ArgumentParser(
        description="Cluster a collection of histograms.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=lambda s: Path(os.path.expandvars(s)),
        required=True,
        help="path to compressed histogram collection",
        metavar="SRC",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=lambda s: os.path.expandvars(s),
        required=True,
        help="path to output file",
        metavar="DEST",
    )
    parser.add_argument(
        "-f",
        "--feature-output",
        default=None,
        type=lambda s: Path(os.path.expandvars(s)),
        help="path to feature output file",
        metavar="DEST",
    )

    # Clustering parameters
    parser.add_argument(
        "-a",
        "--algorithm",
        default="kmeans",
        type=str,
        choices=["agglomerative", "hdbscan", "kmeans"],
        help="clustering algorithm (default: %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--n-cluster-range",
        nargs=2,
        default=[2, 5],
        type=int,
        help="interval of n_cluster to consider (default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--bin-budget",
        default=1000,
        type=int,
        help="total number of bins across clusters (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--transform",
        default=None,
        type=str,
        choices=["standard", "robust", "quantile", "power"],
        help="feature preprocessing method (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quantile-range",
        nargs=2,
        default=(0.25, 0.75),
        type=float,
        help=(
            "quantile range for feature scaling, only used if transform=robust (default:"
            " %(default)s)"
        ),
    )
    parser.add_argument(
        "--alpha",
        default=0.0,
        type=float,
        help="additive smoothing parameter for bin budgeting (default: %(default)s)",
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
        "--seed",
        default=None,
        type=int,
        help="random seed (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        choices=["INFO", "DEBUG", "TRACE"],
        help="verbosity of STDOUT logs (default: %(default)s)",
    )
    parser.add_argument(
        "--log-file",
        type=lambda s: Path(os.path.expandvars(s)),
        default=Path(f"logs/clustering_{timestamp}.log"),
        help="path to log file (default: %(default)s)",
        metavar="LOG",
    )

    return parser.parse_args()


def cluster_histograms(
    hists: list[tuple[np.uint32, Histogram]],
    transform: Literal["standard", "robust", "quantile", "power"] | None,
    quantile_range: tuple[float, float] | None,
    algorithm: Literal["agglomerative", "hdbscan", "kmeans"],
    n_cluster_range: tuple[int, int],
    n_global_bins: int,
    alpha: float,
    seed: int | None,
    workers: int | None,
    verbose: bool = False,
) -> tuple[list[list[tuple[np.uint32, Histogram]]], list[F64Array], F64Array]:
    start = time.perf_counter()

    logger.debug("Computing features")
    features = compute_features(
        hists,
        transform=transform,
        quantile_range=quantile_range,
        seed=seed,
    )

    logger.debug("Computing feature clustering")
    clustering = compute_clustering(
        algorithm,
        features,
        n_cluster_range,
        seed,
        workers,
        verbose,
    )

    logger.debug("Assigning histograms to clusters")
    clustered_hists = assign_histograms(hists, clustering)

    logger.debug("Computing cluster bins")
    cluster_bins = compute_cluster_bins(clustered_hists, n_global_bins, alpha)
    n_clusters = len(cluster_bins)
    logger.debug(f"Number of clusters: {n_clusters}")
    logger.debug(f"Histograms per cluster: {[len(cluster) for cluster in clustered_hists]}")
    logger.debug(f"Bins per cluster: {[len(bins) - 1 for bins in cluster_bins]}")
    logger.debug(
        "Cluster ranges:"
        f" {', '.join([f'({bins[0]:.5g}, {bins[-1]:.5g})' for bins in cluster_bins])}"
    )
    logger.debug(
        f"Uniform bin widths: {', '.join([f'{bins[1] - bins[0]:.5g}' for bins in cluster_bins])}"
    )
    logger.debug(
        "Predicted (rebinning) index size:"
        f" {predict_index_size(clustered_hists, cluster_bins) / 1000**2:.2f} MB"
    )

    end = time.perf_counter()
    logger.info(
        f"Clustered {len(hists)} histograms into {n_clusters} clusters in {end - start:.2f}s"
    )
    logger.trace(f"total_time, {end - start}")

    return clustered_hists, cluster_bins, features


def main() -> None:
    args = parse_args()
    configure_run(args.log_level, args.log_file)
    logger.debug(vars(args))

    clustered_hists, cluster_bins, features = cluster_histograms(
        load_input(args.input, name="histograms"),
        args.transform,
        tuple(args.quantile_range),  # type: ignore
        args.algorithm,
        tuple(args.n_cluster_range),  # type: ignore
        args.bin_budget,
        args.alpha,
        args.seed,
        args.workers,
        args.log_level == "TRACE",
    )

    logger.debug("Saving output")
    output = Path(args.output.replace("k%", f"k{len(cluster_bins):0>3}"))
    save_output(output, (clustered_hists, cluster_bins), "clustering")
    if args.feature_output:
        save_output(args.feature_output, features, name="features")


if __name__ == "__main__":
    main()
