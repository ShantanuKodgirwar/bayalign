import os

import mrcfile
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import KDTree
from scipy.special import logsumexp
from skimage import measure

from bayalign.pointcloud import PointCloud


def load_class_average(index=0, data_path="data/ribosome_80S/pfrib80S_cavgs.mrc"):
    """Loads a projection image (class average) of the 80S ribosome.

    Parameters
    ----------
    index : int, optional
        index for one of the images in the file., by default 0
    data_path : str, optional
        Path to the MRC file.

    Returns
    -------
    np.ndarray
        single image
    """
    assert 0 <= index < 400, f"Index {index} out of bounds (0-399)"

    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    data = mrcfile.open(data_path).data
    return data[index]


def pointcloud_from_class_avg(image, pixelsize=2.68, threshold_method="mean"):
    """Creates a pointcloud from class average projection image

    Parameters
    ----------
    image : np.ndarray
        class average projection image
    pixelsize : float, optional
        Pixel size in Angstrom, by default 2.68
    threshold_method : str, optional
        Method to determine threshold ('mean', 'median', or 'otsu'), by default 'mean'

    Returns
    -------
    target: PointCloud
        Target point cloud
    mask: np.ndarray
        Binary mask of the selected region
    """
    # Determine threshold based on method
    if threshold_method == "mean":
        threshold = np.mean(image)
    elif threshold_method == "median":
        threshold = np.median(image)
    elif threshold_method == "otsu":
        from skimage.filters import threshold_otsu

        threshold = threshold_otsu(image)
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")

    mask = image > threshold

    # Find connected regions
    labels = ndi.label(mask)[0]

    # Keep the largest connected region
    props = measure.regionprops(labels, intensity_image=image)
    if not props:
        raise ValueError("No connected regions found above threshold")

    _centers = np.array([prop.weighted_centroid for prop in props])
    index = np.argmax([prop.area for prop in props])
    mask = labels == props[index].label

    # Create a point cloud representing the projected ribosome
    positions = np.transpose(np.nonzero(mask)) * pixelsize
    weights = image[mask] - threshold

    # Create point cloud
    target = PointCloud(positions, weights)

    # Shift point cloud's center such that the center of mass is at (0, 0)
    target.transformed(np.eye(2), -target.center_of_mass)

    return target, mask


def fit_model2d(target, K, n_iter=1000, k=100, verbose=False):
    """
    Fits a 2D model (mixture of gaussians) with fixed number of points (RBF kernels) K to weighted 2D point cloud using
    Expectation-Maximization (EM) algorithm. This returns a fixed bandwidth `sigma`.

    Parameters
    ----------
    target : PointCloud
        Point cloud that will be fitted with a smaller (unweighted) point cloud
        by using Expectation Maximization.
    K : int
        Desired number of points or clusters
    n_iter : int, optional
        Number of iterations in expectation maximization, by default 1000
    k : int, optional
        Number of nearest neighbours for assignment, by default 100
    verbose : bool, optional
        Whether to print progress information, by default False

    Returns
    -------
    x : np.ndarray
        Fitted positions
    sigma : float
        Standard deviation of the associated fit
    """
    # Normalize weights
    w = np.asarray(target.weights / target.weights.sum())
    y = np.asarray(target.positions)

    sigma = None

    # Initialize centers by sampling from the target positions based on weights
    x = y[np.random.choice(np.arange(target.size), K, replace=False, p=w)]

    for i in range(n_iter):
        # E-step: softly assign points to centers using KDTree
        tree = KDTree(x)
        d, indices = tree.query(y, k=min(k, x.shape[0]))

        # Initialize sigma if not already set
        if sigma is None:
            sigma = np.sqrt(np.dot(w, np.square(np.min(d, axis=1))))

        # Compute assignment probabilities
        p = -0.5 * d**2 / sigma**2
        p -= logsumexp(p, axis=1)[:, None]
        np.exp(p, out=p)

        # M-step: update sigma
        sigma = np.sqrt(np.sum(w[:, None] * p * np.square(d)) / 2)

        # Update centers (x) based on assignment probabilities
        n = ndi.sum_labels(w[:, None] * p, indices, index=np.arange(K))
        x = np.array(
            [
                ndi.sum_labels((w * yy)[:, None] * p, indices, index=np.arange(K))
                for yy in y.T
            ]
        )
        x = (x / n).T

        if verbose and (i + 1) % (n_iter // 10) == 0:
            print(f"EM iteration {i + 1}/{n_iter}, sigma={sigma:.4f}")

    return x, sigma
