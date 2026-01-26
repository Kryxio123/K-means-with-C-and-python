import time
import numpy as np
from wrapper import kmeans_fit


def numpy_kmeans_baseline(X, k, max_iters=20):
    """
    Simple NumPy baseline for benchmarking.
    Intentionally not optimized.
    """
    n_samples, _ = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for _ in range(max_iters):
        distances = np.linalg.norm(
            X[:, None, :] - centroids[None, :, :], axis=2
        )
        labels = np.argmin(distances, axis=1)

        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centroids[i] = X[mask].mean(axis=0)

    return labels, centroids


def benchmark():
    np.random.seed(0)

    X = np.random.rand(100_000, 50)

    # ---- NumPy baseline ----
    start = time.perf_counter()
    numpy_kmeans_baseline(X, k=10, max_iters=20)
    t_numpy = time.perf_counter() - start

    # ---- C extension ----
    start = time.perf_counter()
    kmeans_fit(X, k=10, max_iters=20)
    t_c = time.perf_counter() - start

    print(f"NumPy time: {t_numpy:.3f}s")
    print(f"C-extension time: {t_c:.3f}s")
    print(f"Speedup: {t_numpy / t_c:.2f}x")


if __name__ == "__main__":
    benchmark()
