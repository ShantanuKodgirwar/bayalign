import mrcfile
import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import approx_fprime
from scipy.spatial import KDTree
from scipy.special import logsumexp
from skimage import measure

from bayalign.pointcloud import PointCloud, RotationProjection
from bayalign.score import KernelCorrelation, MixtureSphericalGaussians
from bayalign.sphere_utils import sample_sphere


def load_class_average(index=0):
    """Loads a projection image (class average) of the 80S ribosome.

    Parameters
    ----------
    index : int, optional
        index for one of the images in the file., by default 0

    Returns
    -------
    np.ndarray
        single image
    """

    assert 0 <= index < 400
    data = mrcfile.open("data/ribosome_80S/pfrib80S_cavgs.mrc").data
    return data[index]


def pointcloud_from_class_avg(image, pixelsize=2.68):
    """Creates a pointcloud from class average projection image

    Parameters
    ----------
    image : np.ndarray
        class average projection image
    return_image : bool, optional
        Flag indicating if image should also be returned, by default True
    pixelsize : float, optional
        Pixel size in Angstrom, by default 2.68

    Returns
    -------
    target: np.ndarry
        target image
    """

    # determine a threshold that will be used to convert the grayscale image to
    # a point cloud
    threshold = (np.median(image), np.mean(image))[1]
    mask = image > threshold

    # find connected regions
    labels = ndi.label(mask)[0]

    # keep the largest connected region
    props = measure.regionprops(labels, intensity_image=image)
    centers = np.array([prop.weighted_centroid for prop in props])  # noqa: F841
    index = np.argmax([prop.area for prop in props])
    mask = labels == props[index].label

    # create a point cloud representing the projected ribosome (shifted pixel
    # values will serve as weights)

    # 2D locations of selected pixels in Angstrom
    positions = np.transpose(np.nonzero(mask)) * pixelsize

    # pixel values of selected pixels shifted by threshold so as to make
    # these positive (such that they can be interpreted as weights/mass)
    weights = image[mask] - threshold

    # create point cloud
    target = PointCloud(positions, weights)

    # shift point cloud's center such that the center of mass is at (0, 0)
    # we do this in order to remove an additional in-plane translation
    target.transform(np.eye(2), -target.center_of_mass)

    return target


def fit_model2d(target, K, n_iter=1000, k=100):
    """
    Fits a 2D model (mixture of gaussians) with fixed number of points (RBF kernels) K to weighted 2D point cloud using
    Expectation-Maximization (EM) algorithm. This returns a fixed bandwidth `sigma`.

    :param target: Pointcloud
        Point cloud that will be fitted with a smaller (unweighted) point cloud
        by using Expectation Maximization.
    :param K: positive int
        Desired number of points or clusters
    :param n_iter: positive int
        Number of iterations in expectation maximization
    :param k: positive int
        Number of nearest neighbours that will be considered in the assignment
        of points in the target to points in the approximate (unweighted) point
        cloud.
    :return: x: array_like
        Returns the fitted positions
    :return: sigma: scalar
        Standard deviation of the associated fit.
    """

    # Normalize weights
    w = target.weights / target.weights.sum()
    y = target.positions

    sigma = None

    # Initialize centers by sampling from the target positions based on weights
    x = y[np.random.choice(np.arange(target.size), K, replace=False, p=w)]

    for _ in range(n_iter):
        # E-step: softly assign points to centers using KDTree
        tree = KDTree(x)
        d, i = tree.query(y, k=k)

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
        n = ndi.sum_labels(w[:, None] * p, i, index=np.arange(K))
        x = np.array(
            [ndi.sum_labels((w * yy)[:, None] * p, i, index=np.arange(K)) for yy in y.T]
        )
        x = (x / n).T

    return x, sigma


if __name__ == "__main__":
    class_avg_idx = 10  # default 1
    image = load_class_average(class_avg_idx)
    target_cloud = pointcloud_from_class_avg(image)

    # load 3D model
    n_particles = (1000, 2000)[1]
    model_3d = np.load(f"data/ribosome/model3d_{n_particles}.npz")

    # initialize it as source, an instance of ProjectionRotation
    source_cloud = RotationProjection(model_3d["positions"], model_3d["weights"])
    source_cloud.positions -= source_cloud.center_of_mass

    # best fit to estimate the points and a scalar sigma
    positions_fit2d, sigma = fit_model2d(target_cloud, n_particles, n_iter=100, k=50)

    # initialize correlation
    log_density_kc = KernelCorrelation(
        target_cloud, source_cloud, sigma, k=20, beta=20.0
    )

    # test unit quaternion
    test_q = sample_sphere(3, seed=1234)
    print("Test quaternion:", test_q)

    analytical_grad = log_density_kc.gradient(test_q)
    print("Analytical gradient:", analytical_grad)

    numerical_grad = approx_fprime(test_q, log_density_kc.log_prob)
    print("Numerical gradient:", numerical_grad)

    relative_error = np.linalg.norm(analytical_grad - numerical_grad) / np.linalg.norm(
        numerical_grad
    )
    print(f"Relative error: {relative_error:.3e}")
