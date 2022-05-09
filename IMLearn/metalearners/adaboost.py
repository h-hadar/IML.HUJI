from typing import Callable, NoReturn

import numpy as np

from ..base import BaseEstimator
from ..metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """
    
    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_ = [None] * self.iterations_
        self.weights_, self.D_ = None, None
    
    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.weights_ = np.zeros((self.iterations_,))
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        m = X.shape[0]
        self.D_ = [1 / m] * m
        for t in range(self.iterations_):
            # sample m rows and labels with distribution D
            sample_indices = np.random.choice(m, size=m, p=self.D_)
            samples = X[sample_indices, :]
            labels = y[sample_indices]
            # fit a weak learner on the sampled data
            clf = self.wl_()
            clf.fit(samples, labels)
            self.models_[t] = clf
            error = clf.loss(samples, labels)  # loss on distribution D
            # calculate the weight of this learner
            alpha = .5 * np.log((1 - error) / error)
            self.weights_[t] = alpha
            # update the distribution for the next iteration
            predictions = clf.predict(X)
            self.D_ = self.D_ * np.exp(-alpha * y * predictions)
            self.D_ /= np.sum(self.D_)
    
    def _predict(self, X):
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
        return self.partial_predict(X, self.iterations_)
    
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
        return self.partial_loss(X, y, self.iterations_)
    
    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        answers_sum = np.zeros(X.shape[0])
        for t in range(T):
            answers_sum += self.weights_[t] * self.models_[t].predict(X)
        return np.sign(answers_sum)
    
    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under misclassification loss function
        """
        return misclassification_error(y, self.partial_predict(X, T))
