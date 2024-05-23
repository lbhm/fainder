from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox, BboxBase


def parse_logs_wide(logfile: Path | str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    data["query_times"] = []
    with open(logfile, "r") as file:
        lines = file.readlines()

    for line in lines:
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


def parse_logs_long(logfile: Path | str) -> list[dict[str, Any]]:
    data: list[dict[str, str | float]] = []
    with open(logfile, "r") as file:
        lines = file.readlines()

    for line in lines:
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


def parse_logs_special(logfile: Path | str) -> list[float]:
    with open(logfile, "r") as file:
        lines = file.readlines()

    times: list[float] = []
    for line in lines:
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
        if offsets:
            height = patch.get_height() + offsets[i]  # type: ignore
        else:
            height = patch.get_height()  # type: ignore

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


def plot_legend(path: str, handles: list[Artist], labels: list[str] | None, ncol: int = 1) -> None:
    fig = plt.figure(figsize=(10, 3))

    fig.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=ncol,
    )

    plt.tight_layout(pad=1.02)
    plt.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def plot_ylabel(
    path: str, label: str, bbox: BboxBase, height: float, width: float, xlabel: str | None = None
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(3, height))
    ax.set_ylabel(label)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    plt.tight_layout(pad=1.02)
    label_bbox = fig.get_tightbbox()
    adjusted_bbox = Bbox(((label_bbox.x0, bbox.y0), (width, bbox.y1)))
    plt.savefig(path, bbox_inches=adjusted_bbox)
    plt.close()


def set_style(font_scale: float = 6.0) -> None:
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
