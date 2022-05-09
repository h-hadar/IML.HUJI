from __future__ import annotations
from typing import Callable
from typing import NoReturn
from ...base import BaseEstimator
from ...metrics import loss_functions
import numpy as np


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
    """
    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to to given samples. Iterate over given data as long as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.fit_intercept_`
        """
        X = self.__expand_matrix(X)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.coefs_ = np.zeros((n_features,))
        for iteration in range(self.max_iter_):
            found_violation = False
            for i in range(n_samples):
                if np.sign(y[i]) != np.sign(X[i, :] @ self.coefs_):
                    found_violation = True
                    self.coefs_ += y[i] * X[i, :]
                    self.fitted_ = True
                    self.callback_(self, X[i, :], y[i])
                    break  # go on to the next t iteration
            # if we didn't break during the inner loop, it means that we didn't find any violation of the separation.
            # therefore, we can break from the outer loop as well
            if not found_violation:
                break
                
    def __expand_matrix(self, X):
        if self.include_intercept_:
            ones_vector = np.ones(X.shape[0]).reshape(-1, 1)
            X = np.concatenate((ones_vector, X), axis=1)
        return X
        
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return np.sign(self.__expand_matrix(X) @ self.coefs_)

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
        y_pred = self.predict(X)
        return loss_functions.misclassification_error(y_true=y, y_pred=y_pred)
