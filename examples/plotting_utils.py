"""
Visualization utilities for bayalign. To visualize, better convert
the jnp array to np array and then use the functions here.
"""

import logging
import os
from typing import Dict, List

import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial import KDTree


def show_3d_pointcloud(
    positions: np.ndarray,
    pc_size: int = 3,
    opacity: float = 0.6,
    show_axis: bool = False,
    k: int = 200,
    color_by_neighbours: bool = True,
):
    """
    Visualizes a 3D point cloud using Plotly's scatter_3d function.

    Parameters
    ----------
    positions : array-like
        A 2D array or list of shape (n, 3) where each row represents the
        (x, y, z) coordinates of a point in the point cloud.
    pc_size : int, optional
        The size of the points in the point cloud visualization, by default 3.
    opacity : float, optional
        The opacity of the points in the point cloud visualization, by default 0.6.
    show_axis : bool, optional
        A flag to control the visibility of the axes. If False, the axes are
        hidden, by default False.
    k : int, optional
        The number of nearest neighbors to consider when assigning groups, by default 200.
    color_by_group : bool, optional
        A flag to control whether the points are colored by group or not, by default True.

    Returns
    -------
    None
        This function does not return any value. It directly displays the
        3D scatter plot using Plotly.
    """

    # Convert positions to a Pandas DataFrame
    df = pd.DataFrame(positions, columns=["x", "y", "z"])

    # Initialize an array to hold group IDs
    if color_by_neighbours:
        group_ids = np.full(len(positions), -1)  # Initialize all as -1 (unassigned)
        current_group = 0

        # Iterate through each point and assign groups
        for i in range(len(positions)):
            if group_ids[i] == -1:  # Only assign a new group if the point is unassigned
                # Find K-nearest neighbors
                tree = KDTree(positions)
                distances, indices = tree.query(positions[i], k=k)

                # Assign the same group ID to this point and its neighbors
                group_ids[indices] = current_group
                current_group += 1

        df["group"] = group_ids.astype(str)

        # Plot with Plotly, coloring by group
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color="group",
            opacity=opacity,
            width=500,
            height=500,
        )
    else:
        # Create a 3D scatter plot using Plotly
        fig = px.scatter_3d(
            df, x="x", y="y", z="z", opacity=opacity, width=500, height=500
        )

    # Update the traces to set the size of the points in the point cloud
    fig.update_traces(marker=dict(size=pc_size))
    fig.update_layout(showlegend=False)

    # Turn off the axes if show_axis is False
    if not show_axis:
        dict_kwargs = dict(
            showbackground=False,
            showticklabels=False,
            title="",
            visible=False,
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict_kwargs,
                yaxis=dict_kwargs,
                zaxis=dict_kwargs,
            )
        )

    # Show the plot
    fig.show()


def scatter_matrix(
    n_dim: int,
    samples: Dict[str, np.ndarray],
    methods: List[str],
    algos: Dict[str, str],
    loadpath: str,
    filename: str,
    savefig: bool = False,
):
    """
    Plotting scatter matrix with the corner library and adjusted label sizes

    Parameters
    ----------
    n_dim : int
        Number of dimensions in the data
    samples : dict
        Dictionary with method names as keys and sample arrays as values
    methods : list
        List of method names to include in the plot
    algos : dict
        Dictionary mapping method names to display names for the legend
    loadpath : str
        Path where to save the figure if savefig is True
    filename : str
        Name of the file to save
    savefig : bool, optional
        Whether to save the figure, by default False

    Raises
    ------
    ImportError
        If required optional dependencies (matplotlib, corner) are not installed.
    """
    # Check for required dependencies

    # Define font sizes
    label_size = 32  # Size for axis labels
    tick_size = 20  # Size for tick labels
    legend_size = 24  # Size for legend

    # create dir to save scatter matrices
    labels = [rf"$x_{i}$" for i in range(n_dim)]

    # Set default font sizes for matplotlib
    plt.rcParams.update(
        {
            "font.size": tick_size,
            "axes.labelsize": label_size,
            "axes.titlesize": label_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
        }
    )

    # Create custom labels for each dataset
    colors = ["tab:blue", "tab:orange", "tab:green", "indianred"]

    figure = plt.figure(figsize=(18, 18))

    for method, color in zip(methods, colors):
        # samples for every method (draws, dimensions)
        samples_per_method = samples[method][: int(1e6)]

        # First corner plot for contours and 1D histograms using all samples
        figure = corner.corner(
            samples_per_method,
            bins=250,
            color=color,
            labels=labels,
            fig=figure,
            plot_density=False,
            plot_contours=True,  # shows the 2D histograms with contours
            contour_kwargs={"alpha": 0.6},
            plot_datapoints=False,
            levels=[0.68, 0.95],
            labelsize=label_size,
            label_kwargs={"fontsize": label_size, "labelpad": 10},
            tick_labels_size=tick_size,
            hist_kwargs={"alpha": 1.0},  # 1D histogram
            smooth1d=2,  # smoothens the 1D histogram
        )

        # Second corner plot for showing fewer scatter points
        figure = corner.corner(
            samples_per_method[::50],
            bins=50,
            color=color,
            plot_density=False,
            plot_contours=False,
            fig=figure,
            plot_datapoints=True,  # only shows the scatter points
            data_kwargs={"alpha": 0.1},
            labels=labels,
            labelsize=label_size,
            label_kwargs={"fontsize": label_size, "labelpad": 10},
            tick_labels_size=tick_size,
            hist_kwargs={"alpha": 0.0},  # 1D histogram disabled
        )

    # Create custom legend with the figure instance
    legend_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    figure.legend(
        legend_handles,
        [algos[method] for method in methods],
        loc="upper right",
        fontsize=legend_size,
    )

    # Adjust tick label sizes for all axes
    axes = np.array(figure.axes).reshape((n_dim, n_dim))
    for ax in axes.flat:
        if ax is not None:
            ax.tick_params(labelsize=tick_size)

    # save corner plot
    if savefig:
        savedir = f"{loadpath}/corner_plots"
        os.makedirs(savedir, exist_ok=True)
        logging.info(f"Saving corner plot to {savedir}/{filename}.pdf")
        figure.savefig(f"{savedir}/{filename}.pdf", bbox_inches="tight", dpi=150)
