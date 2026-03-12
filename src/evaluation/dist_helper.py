"""Distance computation utilities for graph metric evaluation.

This module provides functions for computing Maximum Mean Discrepancy (MMD)
and Earth Mover's Distance (EMD) between graph distributions.
"""

import concurrent.futures
from functools import partial

import numpy as np
from scipy.linalg import toeplitz
from tqdm import tqdm


def gaussian(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Gaussian kernel between two histograms.

    Args:
        x: First histogram.
        y: Second histogram.
        sigma: Kernel bandwidth.

    Returns:
        Gaussian kernel value.
    """
    support_size = max(len(x), len(y))
    x = x.astype(float)
    y = y.astype(float)

    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    dist = np.linalg.norm(x - y, 2)
    return np.exp(-dist * dist / (2 * sigma * sigma))


def gaussian_tv(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Gaussian kernel with total variation distance.

    Args:
        x: First histogram.
        y: Second histogram.
        sigma: Kernel bandwidth.

    Returns:
        Gaussian kernel value based on TV distance.
    """
    support_size = max(len(x), len(y))
    x = x.astype(float)
    y = y.astype(float)

    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))


def gaussian_emd(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 1.0,
    distance_scaling: float = 1.0,
) -> float:
    """Gaussian kernel with Earth Mover's Distance.

    Args:
        x: First histogram.
        y: Second histogram.
        sigma: Kernel bandwidth.
        distance_scaling: Scaling factor for distance matrix.

    Returns:
        Gaussian kernel value based on EMD.
    """
    try:
        import pyemd
    except ImportError:
        return gaussian_tv(x, y, sigma)

    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(float)
    distance_mat = d_mat / distance_scaling

    x = x.astype(float)
    y = y.astype(float)

    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd_val = pyemd.emd(x, y, distance_mat)
    return np.exp(-emd_val * emd_val / (2 * sigma * sigma))


def _kernel_worker(args: tuple) -> float:
    """Worker function for parallel kernel computation."""
    x, samples2, kernel_fn = args
    return sum(kernel_fn(x, s2) for s2 in samples2)


def disc(
    samples1: list[np.ndarray],
    samples2: list[np.ndarray],
    kernel: callable,
    is_parallel: bool = True,
    **kwargs,
) -> float:
    """Compute discrepancy between two sample sets.

    Args:
        samples1: First set of samples.
        samples2: Second set of samples.
        kernel: Kernel function.
        is_parallel: Whether to use parallel computation.
        **kwargs: Additional kernel arguments.

    Returns:
        Average kernel value between sample sets.
    """
    # Extract optional progress kwargs (do not pass them to the kernel)
    show_progress = kwargs.pop("show_progress", False)
    progress_desc = kwargs.pop("progress_desc", "")

    if not is_parallel:
        d = 0.0
        iterable = (
            tqdm(samples1, desc=progress_desc or "disc", unit="sample")
            if show_progress
            else samples1
        )
        for s1 in iterable:
            for s2 in samples2:
                d += kernel(s1, s2, **kwargs)
    else:
        kernel_fn = partial(kernel, **kwargs)
        args_list = [(s1, samples2, kernel_fn) for s1 in samples1]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results_iter = executor.map(_kernel_worker, args_list)
            if show_progress:
                d = 0.0
                for val in tqdm(
                    results_iter,
                    total=len(args_list),
                    desc=progress_desc or "disc",
                    unit="chunk",
                ):
                    d += val
            else:
                d = sum(results_iter)

    n1, n2 = len(samples1), len(samples2)
    if n1 * n2 > 0:
        return d / (n1 * n2)
    return 1e6


def compute_mmd(
    samples1: list[np.ndarray],
    samples2: list[np.ndarray],
    kernel: callable = gaussian,
    is_hist: bool = True,
    **kwargs,
) -> float:
    """Compute Maximum Mean Discrepancy between two sample sets.

    MMD = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    Args:
        samples1: First set of samples (reference).
        samples2: Second set of samples (generated).
        kernel: Kernel function.
        is_hist: Whether samples are histograms to normalize.
        **kwargs: Additional kernel arguments.

    Returns:
        MMD value (lower is better).
    """
    if is_hist:
        samples1 = [s1 / (np.sum(s1) + 1e-6) for s1 in samples1]
        samples2 = [s2 / (np.sum(s2) + 1e-6) for s2 in samples2]

    return (
        disc(samples1, samples1, kernel, **kwargs)
        + disc(samples2, samples2, kernel, **kwargs)
        - 2 * disc(samples1, samples2, kernel, **kwargs)
    )
