from typing import Tuple
import numpy as np
import pandas as pd


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion) -> Tuple[
	pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
	"""
    Split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
	# X = X.reset_index(drop=True)
	# y = y.reset_index(drop=True)
	train_x = X.sample(frac=train_proportion)
	train_y = y.iloc[train_x.index]
	test_indices = list(set(X.index) - set(train_x.index))
	test_x = X.iloc[test_indices]
	test_y = y.iloc[test_indices]
	return train_x.reset_index(drop=True), train_y.reset_index(drop=True), \
		test_x.reset_index(drop=True), test_y.reset_index(drop=True)


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	"""
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
	raise NotImplementedError()
