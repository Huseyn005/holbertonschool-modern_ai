#!/usr/bin/env python3
"""
This module contains a function to perform feature standardization on tabular data.
"""
from sklearn import preprocessing


def Standardize(X):
    """
    Standardizes a numpy.ndarray of tabular data using Scikit-learn's StandardScaler.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features)

    Returns:
        numpy.ndarray: The standardized version of the input data.
    """
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
