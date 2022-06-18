from __future__ import annotations

import math
from typing import Tuple, NoReturn

import numpy as np

from ...base import BaseEstimator
from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is above the threshold
    """
    
    def __init__(self):
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None
    
    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        best_loss = math.inf
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        best_loss = math.inf
        for sign in (-1, 1):
            # thr_array has two rows for each feature: row 0 has the thresholds and row 1 has the errors
            thr_array = np.apply_along_axis(lambda col: self._find_threshold(col, y, sign), axis=0, arr=X)
            best_threshold_index = np.argmin(thr_array[1, :])
            loss = thr_array[1, best_threshold_index]
            if loss < best_loss:
                self.j_ = best_threshold_index
                best_loss = loss
                self.sign_ = sign
                self.threshold_ = thr_array[0, best_threshold_index]
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -self.sign_)
    
    @staticmethod
    def _find_threshold(values: np.ndarray, weighted_true_labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        weighted_true_labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassification error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        best_loss = math.inf
        threshold = 0
        for current_val in values:
            # create a label vec with 1's (or -1's) where feature values > current value
            possible_labels = np.where(values >= current_val, sign, -sign)
            possible_loss = DecisionStump.weighted_error(weighted_true_labels, possible_labels)
            if possible_loss < best_loss:
                best_loss, threshold = possible_loss, current_val
        return threshold, best_loss
    
    @staticmethod
    def weighted_error(weighted_true_labels, predictions):
        return np.sum(abs(weighted_true_labels)[weighted_true_labels * predictions < 0])
    
    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under misclassification loss function
        """
        return misclassification_error(y, self.predict(X))
