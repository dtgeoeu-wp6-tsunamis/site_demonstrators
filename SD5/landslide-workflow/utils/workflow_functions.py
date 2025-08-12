import os
import numpy as np

def exceedance_probability(probs, value, thresholds, weights):
    """
    Vectorized calculation of exceedance probabilities for an array of thresholds.
    Returns a numpy array of weighted exceedance probabilities for each threshold.

    probs:  2D numpy array of shape (num_clusters, num_scenarios) Probability of release by cluster and scenario.
    value: 1D numpy array of shape (num_clusters,) Value to be exceeded by cluster.
    thresholds: 1D numpy array of shape (num_thresholds,) Thresholds for exceedance.
    weights: 1D numpy array of shape (num_scenarios,) Weights for each scenario.
    """
    # Create a mask matrix: shape (num_thresholds, num_clusters)
    mask = value[None, :] >= thresholds[:, None]  # shape (T, C)

    # For each threshold, mask out clusters not exceeding the threshold
    # Set probabilities for clusters not exceeding threshold to 0 (so they don't affect the product)
    masked_probs = np.where(mask[:, :, None], probs[None, :, :], 0.0)  # shape (T, C, S)

    # Compute product over clusters (axis=1), for each threshold and scenario
    prod = np.prod(1 - masked_probs, axis=1)  # shape (T, S)
    exceed_by_scenario = 1 - prod  # shape (T, S)

    # Weighted sum over scenarios for each threshold
    weighted = np.dot(exceed_by_scenario, weights)  # shape (T,)

    return weighted