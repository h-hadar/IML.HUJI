from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    m = len(y)
    train_score = 0
    validation_score = 0
    sub_data = np.array_split(X, cv)
    sub_labels = np.array_split(y, cv)
    for fold_index, validation_X in enumerate(sub_data):
        # get the fold slices:
        train_X = np.concatenate(([sub_data[j] for j in set(range(fold_index)).union(range(fold_index+1, cv))]))
        train_y = np.concatenate(([sub_labels[j] for j in set(range(fold_index)).union(range(fold_index+1, cv))]))
        validation_y = sub_labels[fold_index]
        # score the performance on given fold
        estimator.fit(train_X, train_y)
        train_score += scoring(estimator.predict(train_X), train_y)
        validation_score += scoring(estimator.predict(validation_X), validation_y)
    return train_score / cv, validation_score / cv
