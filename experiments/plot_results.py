"""This script loads experiment results and plots them."""

import copy
import os
import shutil
import subprocess
from itertools import chain
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox, BboxBase

from fainder.utils import configure_run, load_input

OUTPUT_PATH = Path("tex/figures")


def change_to_git_root() -> None:
    """Change the current working directory to the root of the git repository."""
    git_root = (
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip().decode("utf-8")
    )
    os.chdir(git_root)


def delete_tex_cache() -> None:
    """Delete the matplotlib tex cache."""
    shutil.rmtree(Path.home() / ".cache" / "matplotlib" / "tex.cache", ignore_errors=True)


def set_style(font_scale: float = 6.0) -> None:
    """Configure seaborn and matplotlib."""
    sns.set_theme(
        context="paper",
        style="ticks",
        palette="colorblind",
        color_codes=True,
        rc={
            "axes.labelpad": 1.5,  # 4
            "axes.labelsize": font_scale,
            "axes.linewidth": 0.4,  # 0.8
            "axes.titlesize": font_scale,
            "font.family": "Computer Modern",
            "font.size": font_scale,
            "grid.linewidth": 0.4,  # 0.8
            "hatch.linewidth": 0.5,  # 1.0
            "legend.borderpad": 0.4,  # 0.4
            "legend.borderaxespad": 0.25,  # 0.5
            "legend.columnspacing": 1.0,  # 2.0
            "legend.fontsize": font_scale * 0.9,
            "legend.frameon": False,  # True
            "legend.handleheight": 0.7,  # 0.7
            "legend.handlelength": 1.5,  # 2.0
            "legend.handletextpad": 0.7,  # 0.8
            "legend.labelspacing": 0.4,  # 0.5
            "legend.title_fontsize": font_scale,
            "lines.linewidth": 1.0,  # 1.5
            "lines.markersize": 3.0,  # 6.0
            "patch.linewidth": 0.5,  # 1.0
            "xtick.labelsize": font_scale * 0.9,
            "xtick.major.pad": 2.0,  # 3.5
            "xtick.major.size": 3.5,  # 2.5
            "xtick.major.width": 0.4,  # 0.8
            "xtick.minor.pad": 2.0,  # 3.4
            "xtick.minor.size": 2.0,  # 2.0
            "xtick.minor.width": 0.3,  # 0.6
            "ytick.labelsize": font_scale * 0.9,
            "ytick.major.pad": 2.0,  # 3.5
            "ytick.major.size": 3.5,  # 3.5
            "ytick.major.width": 0.4,  # 0.8
            "ytick.minor.pad": 2.0,  # 3.4
            "ytick.minor.size": 2.0,  # 2.0
            "ytick.minor.width": 0.3,  # 0.6
            "text.usetex": True,
            "text.latex.preamble": (
                r"\newcommand{\system}[0]{\textsc{Fainder}}"
                r"\newcommand{\exact}[0]{\textsc{Fainder Exact}}"
                r"\newcommand{\approximate}[0]{\textsc{Fainder Approx}}"
                r"\newcommand{\pscan}[0]{\texttt{profile-scan}}"
                r"\newcommand{\binsort}[0]{\texttt{binsort}}"
                r"\newcommand{\ndist}[0]{\texttt{normal-dist}}"
            ),
        },
    )


def parse_wide_logs(path: Path) -> dict[str, Any]:
    """Parse logs in wide format."""
    data: dict[str, Any] = {}
    data["query_times"] = []
    with path.open("r") as file:
        for line in file.readlines():
            log = line.split("|")
            if len(log) <= 1 or log[1].strip() != "TRACE":
                continue
            log = log[2].split(",")
            log = [x.strip() for x in log]
            if log[0] == "query_time":
                data["query_times"].append(float(log[1]))
            elif log[0] in data:
                raise ValueError(f"Duplicate log entry: {log[0]}")
            else:
                data[log[0]] = float(log[1])
    data["query_times"] = np.array(data["query_times"])

    return data


def parse_long_logs(path: Path) -> list[dict[str, Any]]:
    """Parse logs in long format."""
    data: list[dict[str, str | float]] = []
    with path.open("r") as file:
        for line in file.readlines():
            log = line.split("|")
            if len(log) <= 1 or log[1].strip() != "TRACE":
                continue
            log = log[2].split(",")
            log = [x.strip() for x in log]

            entry: dict[str, str | float] = {}
            entry["metric"] = log[0]
            entry["value"] = float(log[1])
            data.append(entry)

    return data


def parse_execution_times(path: Path) -> list[float]:
    """Parse the execution times from the given log file."""
    times: list[float] = []
    with path.open("r") as file:
        for line in file.readlines():
            log = line.split("|")
            if log[1].strip() != "TRACE":
                continue
            log = log[2].split(",")
            log = [x.strip() for x in log]
            if log[0] != "query_collection_time":
                continue
            times.append(float(log[1]))

    if len(times) != 2:
        raise ValueError(f"Expected two query collection times, got {len(times)}")

    return times


def load_runtime_logs(path: Path) -> pd.DataFrame:
    """Load runtime logs from the given path."""
    log_list = []
    for file in path.iterdir():
        config = file.stem.split("-")
        data = parse_wide_logs(file)
        data["dataset"] = config[0]
        data["query_set"] = config[1]
        data["approach"] = config[2]
        data["execution"] = config[3]
        log_list.append(data)

    logs = pd.DataFrame(
        log_list,
        columns=["dataset", "query_set", "approach", "execution", "query_collection_time"],
    )
    return logs[(logs["execution"] == "single") | (logs["execution"] == "single_suppressed")]


def load_execution_trace(path: Path) -> pd.DataFrame:
    """Load execution trace logs from the given path."""
    trace_list = []
    for file in path.iterdir():
        config = file.stem.split("-")
        data = parse_long_logs(file)

        for entry in data:
            entry["dataset"] = config[0]
            entry["index_type"] = config[1]

        trace_list += data

    return pd.DataFrame(trace_list, columns=["dataset", "index_type", "metric", "value"])


def load_indexing_logs(path: Path) -> pd.DataFrame:
    """Load indexing logs from the given path."""
    log_list = []
    for file in path.iterdir():
        config = file.stem.split("-")
        if len(config) != 4:
            continue
        data = parse_wide_logs(file)
        data["dataset"] = config[0]
        data["phase"] = config[1]
        data["parameter"] = config[2][0]
        data["parameter_value"] = int(config[2][1:])
        log_list.append(data)

    return pd.DataFrame(
        log_list, columns=["dataset", "phase", "parameter", "parameter_value", "total_time"]
    )


def load_scalability_logs(path: Path) -> pd.DataFrame:
    """Loads runtime logs for a scalability analysis from the given path."""
    log_list = []
    for file in path.iterdir():
        config = file.stem.split("-")
        data = parse_wide_logs(file)
        data["dataset"] = config[0]
        data["query_set"] = config[1]
        data["index_type"] = config[2]
        data["execution"] = config[3]
        data["scaling_factor"] = int(config[4][2:]) / 100
        log_list.append(data)

    logs = pd.DataFrame(
        log_list,
        columns=[
            "dataset",
            "query_set",
            "index_type",
            "execution",
            "scaling_factor",
            "query_collection_time",
            "query_times",
            "avg_result_size",
        ],
    )
    logs["query_times_sum"] = logs["query_times"].apply(np.sum)
    return logs


def load_exact_runtime_logs(approx_path: Path, exact_path: Path) -> pd.DataFrame:
    """Load runtime logs for the exact and approximate approaches."""
    log_list = []
    for logfile in chain(
        approx_path.glob("*collection-binsort-single*"),
        approx_path.glob("*collection-iterative-single*"),
    ):
        config = logfile.stem.split("-")
        data = parse_wide_logs(logfile)
        data["dataset"] = config[0]
        data["query_set"] = config[1]
        data["approach"] = config[2]
        data["execution"] = config[3]
        data["iteration"] = config[4]
        log_list.append(data)

    approx_logs = pd.DataFrame(
        log_list,
        columns=["dataset", "approach", "iteration", "query_collection_time"],
    )
    approx_logs = approx_logs.rename(columns={"query_collection_time": "baseline_time"})
    approx_logs = approx_logs.replace({"approach": {"iterative": "pscan"}})

    log_list = []
    for logfile in exact_path.glob("*.zst"):
        config = logfile.stem.split("-")
        data = load_input(logfile)
        data["dataset"] = config[0]
        data["approach"] = config[1]
        data["iteration"] = config[2]
        log_list.append(data)

    exact_logs = pd.DataFrame(
        log_list,
        columns=[
            "dataset",
            "approach",
            "iteration",
            "precision_time",
            "recall_time",
            "iterative_time",
            "avg_reduction",
        ],
    )
    runtime = exact_logs.merge(approx_logs, on=["dataset", "approach", "iteration"])
    runtime["exact_time"] = (
        runtime["precision_time"] + runtime["recall_time"] + runtime["iterative_time"]
    )
    return runtime.groupby(["dataset", "approach"]).mean(numeric_only=True)


def load_accuracy_benchmark_logs(path: Path) -> pd.DataFrame:
    """Load accuracy benchmark logs from the given path."""
    log_list = []
    for dataset in ["sportstables", "open_data_usa", "gittables"]:
        for query_set in ["low_selectivity", "mid_selectivity", "high_selectivity"]:
            # Baselines
            for approach, metric_name in [
                ("pscan", "hist"),
                ("ndist", "dist"),
                ("binsort", "binsort"),
            ]:
                acc_logs = load_input(path / f"{dataset}-{approach}-{query_set}.zst")
                perf_logs = parse_wide_logs(path / f"{dataset}-{approach}-{query_set}.log")
                log_list.append(
                    [
                        dataset,
                        approach,
                        query_set,
                        perf_logs["query_collection_time"],
                        np.mean(acc_logs[f"{metric_name}_metrics"][0]),
                        np.mean(acc_logs[f"{metric_name}_metrics"][1]),
                        np.mean(acc_logs[f"{metric_name}_metrics"][2]),
                        np.mean(acc_logs[f"{metric_name}_metrics"][3]),
                    ]
                )

            # Fainder Approx
            for index_mode in ["rebinning", "conversion"]:
                acc_logs = load_input(path / f"{dataset}-{index_mode}-{query_set}.zst")
                exec_time = parse_execution_times(path / f"{dataset}-{index_mode}-{query_set}.log")
                for i, metric_mode in enumerate(["precision", "recall"]):
                    log_list.append(
                        [
                            dataset,
                            f"{index_mode}-{metric_mode}",
                            query_set,
                            exec_time[i],
                            np.mean(acc_logs[f"{metric_mode}_mode_metrics"][0]),
                            np.mean(acc_logs[f"{metric_mode}_mode_metrics"][1]),
                            np.mean(acc_logs[f"{metric_mode}_mode_metrics"][2]),
                            np.mean(acc_logs[f"{metric_mode}_mode_metrics"][3]),
                        ]
                    )

            # Fainder Exact
            logs = load_input(path / f"{dataset}-exact-{query_set}.zst")
            log_list.append(
                [
                    dataset,
                    "exact",
                    query_set,
                    logs["precision_time"] + logs["recall_time"] + logs["iterative_time"],
                    1,  # Metrics not logged because approach is exact
                    1,
                    1,
                    None,
                ]
            )

    return pd.DataFrame(
        log_list,
        columns=[
            "dataset",
            "approach",
            "queries",
            "time",
            "precision",
            "recall",
            "f1",
            "pruning_factor",
        ],
    )


def load_microbenchmark(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load microbenchmark logs from the given path."""
    log_list = []
    for file in (path / "runtime").iterdir():
        config = file.stem.split("-")
        data = parse_wide_logs(file)
        data["dataset"] = config[0]
        data["index_type"] = config[1]
        data["parameter"] = config[2][0]
        data["parameter_value"] = int(config[2][1:])
        data["execution"] = config[3]
        log_list.append(data)

    runtime = (
        pd.DataFrame(
            log_list,
            columns=[
                "dataset",
                "index_type",
                "parameter",
                "parameter_value",
                "execution",
                "query_collection_time",
                "avg_result_size",
            ],
        )
        .groupby(["dataset", "index_type", "parameter", "parameter_value", "execution"])
        .mean()
        .reset_index()
    )

    log_list = []
    for file in (path / "indexing").iterdir():
        config = file.stem.split("-")
        if config[1] != "rebinning":
            continue
        data = parse_wide_logs(file)
        data["dataset"] = config[0]
        data["phase"] = config[1]
        data["parameter"] = config[2][0]
        data["parameter_value"] = int(config[2][1:])
        log_list.append(data)

    index_size = (
        pd.DataFrame(
            log_list,
            columns=[
                "dataset",
                "phase",
                "parameter",
                "parameter_value",
                "index_size",
            ],
        )
        .groupby(["dataset", "phase", "parameter", "parameter_value"])
        .mean()
        .reset_index()
    )

    log_list = []
    for file in (path / "accuracy").iterdir():
        logs = load_input(file)
        config = file.stem.split("-")
        for mode, mode_data in [
            ("recall", logs["recall_mode_metrics"]),
            ("precision", logs["precision_mode_metrics"]),
        ]:
            for i, values in enumerate(mode_data):
                for value in values:
                    log_list.append(
                        {
                            "dataset": config[0],
                            "index_type": config[1],
                            "parameter": config[2][0],
                            "parameter_value": int(config[2][1:]),
                            "index_mode": mode,
                            "metric": ["precision", "recall", "f1", "pruning_factor"][i],
                            "value": value,
                        }
                    )

    accuracy = (
        pd.DataFrame(log_list)
        .groupby(["dataset", "index_type", "index_mode", "parameter", "parameter_value", "metric"])
        .agg({"value": "mean"})
        .reset_index()
    )

    return runtime, index_size, accuracy


def autolabel_bars(
    ax: Axes,
    offsets: list[float] | None = None,
    precision: int = 1,
    decimal_precision: int | None = None,
    rotation: int = 0,
    y_offset: float = 1,
    unrotate_first: bool = False,
) -> None:
    """Attach a text label above each bar in `ax`, displaying its height."""
    if decimal_precision is None:
        decimal_precision = precision

    for i, patch in enumerate(ax.patches):
        height = patch.get_height() + offsets[i] if offsets else patch.get_height()  # type: ignore
        height_str = f"{height:.{precision}g}"
        ha = "center"
        x = patch.get_x() + patch.get_width() / 2  # type: ignore
        current_rotation = rotation
        if i == 0 and unrotate_first:
            current_rotation = 0

        # Special cases for large numbers in GitTables
        if height < 1:
            height_str = f"{height:.{decimal_precision}f}"
        elif height > 1000:
            height_str = f"{height:.0f}"
            if height > 10000:
                ha = "left"
                x = patch.get_x()  # type: ignore
                current_rotation = 0

        ax.annotate(
            height_str,
            xy=(x, height),
            xytext=(0, y_offset),
            fontsize=mpl.rcParams["font.size"] * 0.8,
            textcoords="offset points",
            ha=ha,
            va="bottom",
            rotation=current_rotation,
        )


def plot_legend(
    path: Path, handles: list[Artist], labels: list[str] | None, ncol: int = 1
) -> None:
    """Plot a legend with the given handles and labels."""
    fig = plt.figure(figsize=(10, 3))
    fig.legend(handles=handles, labels=labels, loc="center", ncol=ncol)
    plt.tight_layout(pad=1.02)
    plt.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def plot_label(
    path: Path, label: str, bbox: BboxBase, height: float, width: float, xlabel: str | None = None
) -> None:
    """Plot a separate y-label figure with the given label."""
    fig, ax = plt.subplots(1, 1, figsize=(3, height))
    ax.set_ylabel(label)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    plt.tight_layout(pad=1.02)
    label_bbox = fig.get_tightbbox()
    adjusted_bbox = Bbox(((label_bbox.x0, bbox.y0), (width, bbox.y1)))
    plt.savefig(path, bbox_inches=adjusted_bbox)
    plt.close()


def plot_runtime(runtime: pd.DataFrame, fig_num: int) -> None:
    """Plot the results for the runtime comparison."""
    height = 0.8
    for dataset, suffix in [("sportstables", "a"), ("open_data_usa", "b"), ("gittables", "c")]:
        fig, ax = plt.subplots(1, 1, figsize=(1.2, height))
        data = (
            runtime[(runtime["dataset"] == dataset) & (runtime["query_set"] == "collection")]
            .groupby(["approach", "execution"])
            .agg({"query_collection_time": ["mean", "std"]})
        )

        ax.bar(
            x=0,
            height=data.loc[("iterative", "single"), ("query_collection_time", "mean")],  # type: ignore
            width=0.5,
            color=sns.color_palette()[0],
            edgecolor="black",
            label=r"\pscan{}",
        )
        ax.bar(
            x=0.75,
            height=data.loc[("binsort", "single"), ("query_collection_time", "mean")],  # type: ignore
            width=0.5,
            color=sns.color_palette()[1],
            edgecolor="black",
            label=r"\binsort{}",
        )
        ax.bar(
            x=1.5,
            height=data.loc[("rebinning", "single"), ("query_collection_time", "mean")],  # type: ignore
            width=0.5,
            color=sns.color_palette()[2],
            edgecolor="black",
            hatch="////",
            label=r"\system{} w/ results",
        )
        ax.bar(
            x=2.05,
            height=data.loc[("rebinning", "single_suppressed"), ("query_collection_time", "mean")],  # type: ignore
            width=0.5,
            color=sns.color_palette()[2],
            edgecolor="black",
            hatch="oooo",
            label=r"\system{} w/o results",
        )

        ax.set_xticks([])
        ax.set_yscale("log")
        if dataset == "gittables":
            ax.set_ylim(top=ax.get_ylim()[1] * 4)
        else:
            ax.set_ylim(top=ax.get_ylim()[1] * 2.5)
        autolabel_bars(ax, precision=3, decimal_precision=2)

        plt.tight_layout(pad=1.02)
        bbox = fig.get_tightbbox()
        plt.savefig(
            OUTPUT_PATH / f"figure_{fig_num}_{suffix}.pdf", bbox_inches="tight", pad_inches=0.01
        )

        ax.set_ylabel("Time (s)")
        bbox = fig.get_tightbbox()
        label_bbox = Bbox(((bbox.x0, bbox.y0), (0.07, bbox.y1)))
        plt.savefig(OUTPUT_PATH / f"figure_{fig_num}_label.pdf", bbox_inches=label_bbox)
        plt.close()

    handles, labels = ax.get_legend_handles_labels()  # type: ignore
    handles.append(copy.deepcopy(handles[-1]))
    handles[2].patches[0].set_hatch("")  # type: ignore
    handles[3].patches[0].set_facecolor("white")  # type: ignore
    handles[3].patches[0].set_hatch("////")  # type: ignore
    handles[4].patches[0].set_facecolor("white")  # type: ignore
    labels = labels[:2] + [r"\system{}", "w/ results", "w/o results"]
    plot_legend(
        OUTPUT_PATH / f"figure_{fig_num}_legend.pdf", handles=handles, labels=labels, ncol=5
    )


def plot_scalability(runtime: pd.DataFrame, fig_num: int) -> None:
    """Plot the results for the scalability analysis."""
    _, ax = plt.subplots(1, 1, figsize=(1.72, 1.2))
    data = (
        runtime[(runtime["index_type"] == "conversion") & (runtime["query_set"] == "collection")]
        .groupby(["execution", "scaling_factor"])
        .agg({"query_collection_time": ["mean", "std"]})
        .values
    )

    x = [0.25, 0.5, 1, 2]
    ax.plot(x, data[: len(x), 0], color=sns.color_palette()[0], label="w/ results")
    ax.fill_between(
        x,
        data[: len(x), 0] - data[: len(x), 1],
        data[: len(x), 0] + data[: len(x), 1],
        alpha=0.1,
        edgecolor="white",
        color=sns.color_palette()[0],
    )
    ax.plot(
        x,
        data[len(x) :, 0],
        color=sns.color_palette()[1],
        label="w/o results",
    )
    ax.fill_between(
        x,
        data[len(x) :, 0] - data[len(x) :, 1],
        data[len(x) :, 0] + data[len(x) :, 1],
        alpha=0.1,
        edgecolor="white",
        color=sns.color_palette()[1],
    )

    ax.set_xticks(x, labels=[".25", ".5", "1", "2"])
    ax.set_ylabel("Time (s)")
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 3)
    ax.legend(loc="center right")
    sns.despine()
    plt.tight_layout(pad=1.02)
    plt.savefig(OUTPUT_PATH / f"figure_{fig_num}.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close()


def plot_execution_trace(trace: pd.DataFrame, fig_num: int = 13) -> None:
    """Plot the results for the execution time breakdown."""
    _, ax = plt.subplots(1, 1, figsize=(1.79, 1.2))

    for i, dataset in enumerate(["sportstables", "open_data_usa", "gittables"]):
        data = trace[trace.dataset == dataset].groupby(["metric"]).agg({"value": ["mean", "std"]})

        bottom = 0
        for j, column in enumerate(
            [
                "query_boostrap_time",
                "query_bin_search_time",
                "query_hist_search_time",
                "query_result_update_time",
                "query_cluster_skip_time",
            ]
        ):
            ax.bar(
                x=i * 0.75,
                height=data.loc[column, ("value", "mean")],  # type: ignore
                width=0.5,
                bottom=bottom,
                color=sns.color_palette()[j],
                edgecolor="black",
                label=(
                    [
                        "Bootstrap",
                        "Bin search",
                        "Histogram search",
                        "Result update",
                        "Cluster skip",
                    ][j]
                    if i == 0
                    else ""
                ),
            )
            bottom += data.loc[column, ("value", "mean")]  # type: ignore

        ax.annotate(
            f"{bottom:.4f}",
            xy=(i * 0.75, bottom),  # type: ignore
            xytext=(0, 1),  # 1 point vertical offset
            textcoords="offset points",
            fontsize=6 * 0.8,
            ha="center",
            va="bottom",
        )

    ax.set_xticks([0, 0.75, 1.5], ["ST", "OD", "GT"])
    ax.set_ylim(
        (trace.groupby(["dataset", "metric"]).mean(numeric_only=True).min()).item() / 2,
        ax.get_ylim()[1] * 5,
    )
    ax.set_yscale("log")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", fontsize="x-small")
    sns.despine()

    plt.tight_layout(pad=1.02)
    plt.savefig(OUTPUT_PATH / f"figure_{fig_num}.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close()


def plot_exact_runtime(runtime: pd.DataFrame, fig_num: int = 14) -> None:
    """Plot the runtime breakdown of Fainder Exact."""
    height = 1.1
    handles: list[Artist] = []
    for dataset, suffix in [("sportstables", "a"), ("open_data_usa", "b"), ("gittables", "c")]:
        fig, ax = plt.subplots(figsize=(1.2, height))
        colors = [sns.color_palette()[i] for i in range(4)]
        hatches = ["xxx", "ooo", "///", "\\\\\\"]

        for i, baseline in enumerate(["pscan", "binsort"]):
            data = runtime.query(f"dataset == '{dataset}' & approach == '{baseline}'")
            handles += ax.bar(
                i * 0.75,
                data["baseline_time"],
                width=0.5,
                color=colors[i],
                edgecolor="black",
                hatch=hatches[i],
            )

            bottom = 0
            for j, time in [
                (3, data["recall_time"]),
                (2, data["precision_time"]),
                (i, data["iterative_time"]),
            ]:
                handles += ax.bar(
                    1.5 + i * 0.55,
                    time,
                    bottom=bottom,
                    width=0.5,
                    color=colors[j],
                    edgecolor="black",
                    hatch=hatches[j],
                )
                bottom += time.item()

            for time, x in [(data["baseline_time"].item(), i * 0.75), (bottom, 1.5 + i * 0.55)]:
                label = f"{time:.0f}" if time > 100 else f"{time:.1f}"
                ax.annotate(
                    label,
                    xy=(x, float(time)),
                    xytext=(0, 1),
                    fontsize=mpl.rcParams["font.size"] * 0.8,
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        ax.set_xticks([0, 0.75, 1.775])
        ax.set_xticklabels(
            [
                "Full\nscan",
                r"\texttt{bin}-" "\n" r"\texttt{sort}",
                r"\textsc{Fainder}" "\n" r"\textsc{ Exact}",
            ]
        )
        if dataset == "gittables":
            ax.set_ylim(200, 90000)
        else:
            ax.set_ylim(
                runtime.query(f"dataset == '{dataset}' & approach == 'pscan'")[
                    "recall_time"
                ].item()
                / 2
                % 10,
                ax.get_ylim()[1] * 2,
            )
        ax.set_yscale("log")

        sns.despine()
        plt.tight_layout(pad=1.02)
        bbox = fig.get_tightbbox()
        plt.savefig(
            OUTPUT_PATH / f"figure_{fig_num}_{suffix}.pdf", bbox_inches="tight", pad_inches=0.01
        )
        plt.close()

    plot_label(
        OUTPUT_PATH / f"figure_{fig_num}_label.pdf",
        "Time (s)",
        bbox,
        height=height,
        width=0.12,
        xlabel="Approach",
    )
    plot_legend(
        OUTPUT_PATH / f"figure_{fig_num}_legend.pdf",
        [handles[0], handles[4], handles[2], handles[1]],
        [
            r"\pscan{}",
            r"\binsort{}",
            r"\textsc{F. Approx} full prec.",
            r"\textsc{F. Approx} full rec.",
        ],
        ncol=4,
    )


def plot_indexing_time(indexing: pd.DataFrame, fig_num: int = 15) -> None:
    """Plot the index construction time analysis."""
    height = 1.1
    dataset = "gittables"
    for param, suffix, xlabel, x in [
        (
            "k",
            "a",
            "Number of clusters",
            list(range(50, 250, 10)) + list(range(250, 1001, 50)),
        ),
        (
            "b",
            "b",
            "Bin budget",
            [10000, 50000, 100000, 500000, 1000000],
        ),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(1.77, height))
        for i, (phase, label) in enumerate(
            [
                ("clustering", "Clustering"),
                ("rebinning", "Rebinning"),
                ("conversion", "Conversion"),
            ]
        ):
            data = (
                indexing[
                    (indexing["dataset"] == dataset)
                    & (indexing["parameter"] == param)
                    & (indexing["phase"] == phase)
                ]
                .groupby(["parameter_value"])
                .agg({"total_time": ["mean", "std"]})
                .values
            )

            ax.plot(
                x,
                data[: len(x), 0],
                color=sns.color_palette()[i],
                label=label,
            )
            ax.fill_between(
                x,
                data[: len(x), 0] - data[: len(x), 1],
                data[: len(x), 0] + data[: len(x), 1],
                alpha=0.1,
                edgecolor="white",
                color=sns.color_palette()[i],
            )

        if param == "b":
            ax.set_xscale("log")
            ax.set_yscale("log")
        if param == "k":
            ax.set_xticks([50, 500, 1000])
        ax.set_xlabel(xlabel)
        ax.set_xlim(min(x), max(x))

        sns.despine()
        plt.tight_layout(pad=1.02)
        bbox = fig.get_tightbbox()
        plt.savefig(
            OUTPUT_PATH / f"figure_{fig_num}_{suffix}.pdf", bbox_inches="tight", pad_inches=0.01
        )
        plt.close()

    handles, labels = ax.get_legend_handles_labels()
    plot_legend(
        OUTPUT_PATH / f"figure_{fig_num}_legend.pdf", handles=handles, labels=labels, ncol=5
    )
    plot_label(
        OUTPUT_PATH / f"figure_{fig_num}_label.pdf",
        "Time (s)",
        bbox,
        height=height,
        width=0.12,
        xlabel="Number of clusters",
    )


def plot_f1_comparison(logs: pd.DataFrame, fig_num: int = 16) -> None:
    """Plot the F1-score comparison."""
    for dataset, suffix in [("sportstables", "a"), ("open_data_usa", "b"), ("gittables", "c")]:
        width = 0.5
        fig, ax = plt.subplots(1, 1, figsize=(2.4, 1.1))

        ax.bar(
            x=0,
            height=logs[(logs["dataset"] == dataset) & (logs["approach"] == "pscan")]["f1"].mean()
            * 100,
            width=width,
            color=sns.color_palette()[0],
            edgecolor="black",
            clip_on=False,
        )

        for i, approach in enumerate(["ndist", "rebinning-recall", "conversion-recall"]):
            for j, query_set in enumerate(
                ["low_selectivity", "mid_selectivity", "high_selectivity"]
            ):
                ax.bar(
                    x=1.5 * width + i * 3.8 * width + j * 1.1 * width,
                    height=logs[
                        (logs["dataset"] == dataset)
                        & (logs["approach"] == approach)
                        & (logs["queries"] == query_set)
                    ]["f1"].mean()
                    * 100,
                    width=width,
                    color=sns.color_palette()[j + 1],
                    edgecolor="black",
                    clip_on=False,
                )

        height = logs[(logs["dataset"] == dataset) & (logs["approach"] == "exact")]["f1"].mean()
        ax.bar(
            x=12.9 * width,
            height=height * 100,
            width=width,
            color=sns.color_palette()[0],
            edgecolor="black",
            clip_on=False,
        )

        ax.set_xticks(
            [0, 2.6 * width, 6.4 * width, 10.2 * width, 12.9 * width],
            [
                r"\texttt{profile-}" "\n" r"\texttt{scan}",
                r"\texttt{normal-}" "\n" r"\texttt{dist}",
                r"\textsc{F. Approx}" "\n" r"low mem.",
                r"\textsc{F. Approx}" "\n" r"full rec.",
                r"\textsc{Fainder}" "\n" r"\textsc{Exact}",
            ],
        )
        ax.set_xlim(-0.6 * width, 13.5 * width)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        autolabel_bars(ax, precision=3, decimal_precision=1)

        sns.despine()
        plt.tight_layout(pad=1.02)
        bbox = fig.get_tightbbox()
        plt.savefig(
            OUTPUT_PATH / f"figure_{fig_num}_{suffix}.pdf", bbox_inches="tight", pad_inches=0.01
        )

        ax.set_ylabel(r"$F_1$ score (\%)")
        bbox = fig.get_tightbbox()
        label_bbox = Bbox(((bbox.x0, bbox.y0), (0.07, bbox.y1)))
        plt.savefig(OUTPUT_PATH / f"figure_{fig_num}_label.pdf", bbox_inches=label_bbox)
        plt.close()

    plot_legend(
        OUTPUT_PATH / f"figure_{fig_num}_legend.pdf",
        handles=[
            Patch(facecolor=sns.color_palette()[0], edgecolor="black", label="All queries"),
            Patch(facecolor=sns.color_palette()[1], edgecolor="black", label="Low selectivity"),
            Patch(facecolor=sns.color_palette()[2], edgecolor="black", label="Mid selectivity"),
            Patch(facecolor=sns.color_palette()[3], edgecolor="black", label="High selectivity"),
        ],
        labels=["All queries", "Low selectivity", "Mid selectivity", "High selectivity"],
        ncol=4,
    )


def plot_approx_comparison(logs: pd.DataFrame, fig_num: int = 17) -> None:
    """Plot the precision and pruning factor comparison."""
    for metric, ylabel, precision, suffix in [
        ("precision", r"Precision (\%)", 2, "a"),
        ("pruning_factor", r"Pruning factor (\%)", 3, "b"),
    ]:
        width = 0.5
        _, ax = plt.subplots(1, 1, figsize=(1.95, 1.15))

        for i, approach in enumerate(["ndist", "rebinning-recall", "conversion-recall"]):
            for j, query_set in enumerate(
                ["low_selectivity", "mid_selectivity", "high_selectivity"]
            ):
                ax.bar(
                    x=i * 3.8 * width + j * 1.1 * width,
                    height=logs[
                        (logs["dataset"] == "gittables")
                        & (logs["approach"] == approach)
                        & (logs["queries"] == query_set)
                    ][metric].mean()
                    * 100,
                    width=width,
                    color=sns.color_palette()[j + 1],
                    edgecolor="black",
                )

        ax.set_xticks(
            [1.1 * width, 4.9 * width, 8.7 * width],
            [
                r"\texttt{normal-}" "\n" r"\texttt{dist}",
                r"\textsc{F. Approx}" "\n" r"low mem.",
                r"\textsc{F. Approx}" "\n" r"full rec.",
            ],
        )
        ax.set_xlim(-0.6 * width, 10.4 * width)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
        autolabel_bars(ax, precision=precision, decimal_precision=2)

        sns.despine()
        plt.tight_layout(pad=1.02)
        plt.savefig(
            OUTPUT_PATH / f"figure_{fig_num}_{suffix}.pdf", bbox_inches="tight", pad_inches=0.01
        )
        plt.close()

    plot_legend(
        OUTPUT_PATH / f"figure_{fig_num}_legend.pdf",
        handles=[
            Patch(facecolor=sns.color_palette()[1], edgecolor="black", label="Low selectivity"),
            Patch(facecolor=sns.color_palette()[2], edgecolor="black", label="Mid selectivity"),
            Patch(facecolor=sns.color_palette()[3], edgecolor="black", label="High selectivity"),
        ],
        labels=["Low selectivity", "Mid selectivity", "High selectivity"],
        ncol=3,
    )


def plot_acc_runtime_scatter(logs: pd.DataFrame, fig_num: int = 18) -> None:
    """Plot the accuracy-runtime scatter plot."""
    height = 1.1
    for dataset, suffix in [("sportstables", "a"), ("open_data_usa", "b"), ("gittables", "c")]:
        fig, ax = plt.subplots(1, 1, figsize=(1.23, height))

        data = (
            logs[
                (logs["dataset"] == dataset)
                & ~logs["approach"].isin(["conversion-precision", "rebinning-precision"])
            ]
            .groupby(["approach"])
            .agg({"time": "mean", "f1": "mean"})
            .reindex(
                ["pscan", "ndist", "binsort", "exact", "rebinning-recall", "conversion-recall"]
            )
        )

        ax.scatter(
            data["time"],
            data["f1"] * 100,
            c=sns.color_palette()[:6],
            clip_on=False,
        )
        ax.grid(
            True, which="major", axis="y", linestyle="--", linewidth=0.5, alpha=0.3, color="gray"
        )
        ax.set_xlabel("Time (s)")
        ax.set_xscale("log")
        ax.set_ylim(0, 100)

        sns.despine()
        plt.tight_layout(pad=1.02)
        bbox = fig.get_tightbbox()
        plt.savefig(
            OUTPUT_PATH / f"figure_{fig_num}_{suffix}.pdf", bbox_inches="tight", pad_inches=0.01
        )

    plot_label(
        OUTPUT_PATH / f"figure_{fig_num}_label.pdf",
        r"$F_1$ score (\%)",
        bbox,
        height=height,
        width=0.12,
        xlabel="Time (s)",
    )
    plot_legend(
        OUTPUT_PATH / f"figure_{fig_num}_legend.pdf",
        handles=[Patch(facecolor=sns.color_palette()[i], edgecolor="black") for i in range(6)],
        labels=[
            r"\pscan{}",
            r"\ndist{}",
            r"\binsort{}",
            r"\exact{}",
            r"\approximate{} low mem.",
            r"\approximate{} full rec.",
        ],
        ncol=3,
    )


def plot_microbenchmark(
    runtime: pd.DataFrame,
    index_size: pd.DataFrame,
    accuracy: pd.DataFrame,
    param: Literal["b", "k"],
    fig_num: int,
) -> None:
    """Plot microbenchmark results for k or b."""

    def prep_fig(dataset: str, size_scale: float = 1.0) -> tuple[Figure, tuple[Axes, Axes, Axes]]:
        r = runtime[(runtime["dataset"] == dataset) & (runtime["parameter"] == param)]
        i = index_size[
            (index_size["dataset"] == dataset)
            & (index_size["parameter"] == param)
            & (index_size["phase"] == "rebinning")
        ]
        a = accuracy[
            (accuracy["dataset"] == dataset)
            & (accuracy["parameter"] == param)
            & (accuracy["index_mode"] == "recall")
            & (accuracy["metric"] == "f1")
        ]

        fig, ax1 = plt.subplots(figsize=(3.5, 1.2), layout="constrained")
        ax2: Axes = ax1.twinx()  # type: ignore
        ax3: Axes = ax1.twinx()  # type: ignore
        ax3.spines.right.set_position(("axes", 1.2))

        # Runtime
        ax1.plot(
            r[(r["execution"] == "single")]["parameter_value"],
            r[(r["execution"] == "single")]["query_collection_time"],
            color=sns.color_palette()[0],
            label="w/ results",
        )
        ax1.plot(
            r[(r["execution"] == "single_suppressed")]["parameter_value"],
            r[(r["execution"] == "single_suppressed")]["query_collection_time"],
            color=sns.color_palette()[0],
            label="w/o results",
            linestyle="--",
        )

        # Index size
        ax2.plot(
            i["parameter_value"],
            i["index_size"] * size_scale,
            color=sns.color_palette()[1],
            label="Index size",
        )

        # Accuracy
        ax3.plot(
            a[(a["index_type"] == "rebinning")]["parameter_value"],
            a[(a["index_type"] == "rebinning")]["value"] * 100,
            color=sns.color_palette()[2],
            label=r"Low mem.",
        )
        ax3.plot(
            a[(a["index_type"] == "conversion")]["parameter_value"],
            a[(a["index_type"] == "conversion")]["value"] * 100,
            color=sns.color_palette()[2],
            label=r"Full rec.",
            linestyle="--",
        )

        return fig, (ax1, ax2, ax3)

    fig, (ax1, ax2, ax3) = prep_fig("open_data_usa")

    if param == "k":
        ax1.set(xlabel="Number of clusters", xlim=(1, 1000), ylabel="Time (s)", yscale="log")
        ax1.set_xticks([1, 100, 200, 400, 600, 800, 1000])
        ax1.set_yticks([0.1, 1], ["0.1", "1"])
    elif param == "b":
        ax1.set(
            xlabel="Bin Budget", xlim=(1e2, 1e6), ylabel="Time (s)", xscale="log", yscale="log"
        )
        ax1.set_yticks([1], ["1"])
    ax2.set(ylabel="Index size (MB)", yscale="log")
    ax3.set(ylabel=r"$F_1$ score (\%)", ylim=(0, 100))
    ax1.yaxis.label.set_color(sns.color_palette()[0])
    ax2.yaxis.label.set_color(sns.color_palette()[1])
    ax3.yaxis.label.set_color(sns.color_palette()[2])

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=5)

    plt.savefig(OUTPUT_PATH / f"figure_{fig_num}.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close()


if __name__ == "__main__":
    ### Setup
    change_to_git_root()
    delete_tex_cache()
    configure_run("WARNING")
    set_style()
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    ### Load data
    runtime = load_runtime_logs(Path("logs/runtime_benchmark/execution/"))
    runtime_low_sel = load_runtime_logs(Path("logs/runtime_benchmark/low_selectivity/"))
    runtime_exact = load_exact_runtime_logs(
        Path("logs/runtime_benchmark/execution/"), Path("logs/exact_results/")
    )
    execution_trace = load_execution_trace(Path("logs/runtime_benchmark/index_trace/"))
    indexing = load_indexing_logs(Path("logs/runtime_benchmark/indexing/"))
    scalability = load_scalability_logs(Path("logs/scalability_benchmark/execution/"))
    accuracy_benchmark = load_accuracy_benchmark_logs(
        Path("logs/accuracy_benchmark/baseline_comp/")
    )
    runtime_mb, index_size_mb, accuracy_mb = load_microbenchmark(Path("logs/microbenchmarks/"))

    ### Plotting
    plot_runtime(runtime, 10)
    plot_runtime(runtime_low_sel, 11)
    plot_scalability(scalability, 12)
    plot_execution_trace(execution_trace, 13)
    plot_exact_runtime(runtime_exact, 14)
    plot_indexing_time(indexing, 15)
    plot_f1_comparison(accuracy_benchmark, 16)
    plot_approx_comparison(accuracy_benchmark, 17)
    plot_acc_runtime_scatter(accuracy_benchmark, 18)
    plot_microbenchmark(runtime_mb, index_size_mb, accuracy_mb, "k", 19)
    plot_microbenchmark(runtime_mb, index_size_mb, accuracy_mb, "b", 20)

    print(f"Succesfully generated all plots and saved them to {OUTPUT_PATH}.")
