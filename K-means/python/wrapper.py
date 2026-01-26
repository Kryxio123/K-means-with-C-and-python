import numpy as np
import kmeans_c


def kmeans_fit(X, k, max_iters=20):
    """
    Python wrapper for C-implemented K-Means.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        Must be float64 and C-contiguous
    k : int
        Number of clusters
    max_iters : int
        Maximum number of iterations

    Returns
    -------
    labels : np.ndarray (n_samples,)
    centroids : np.ndarray (k, n_features)
    """
    # Enforce correct memory layout before crossing Python → C
    X = np.asarray(X, dtype=np.float64, order="C")

    labels, centroids = kmeans_c.fit(X, k, max_iters)
    return labels, centroids
