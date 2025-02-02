from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        if self.biased_:
            self.mu_ = np.mean(X) + 1/np.size(X)
            self.var_ = np.sum(np.power(X - self.mu_, 2)) / np.size(X)
        else:
            self.mu_ = np.mean(X)
            self.var_ = np.sum(np.power(X - self.mu_, 2)) / (np.size(X) - 1)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        norm_factor = 1 / (np.sqrt(np.pi*2 * self.var_))
        e_powers = -(X - self.mu_) ** 2 / (2 * self.var_)
        return norm_factor * np.exp(e_powers)

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        norm_factor = 1 / (np.sqrt(np.pi*2 * sigma))
        exponent = -(X - mu)**2 / (2 * sigma)
        return np.sum(np.log(norm_factor) + exponent)


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        # raise NotImplementedError()
        self.mu_ = np.mean(X, axis=0)
        n = (np.size(X, axis=1))
        self.cov_ = [[self.cov_helper(X, i, j) for i in range(n)] for j in range(n)]
        self.fitted_ = True
        return self
    
    def cov_helper(self, X, i, j):
        vec_i_diff = X[:, i] - self.mu_[i]
        vec_j_diff = X[:, j] - self.mu_[j]
        return np.sum(np.multiply(vec_i_diff, vec_j_diff)) / (np.size(X, axis=0) - 1)
    
    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        norm_fact = 1 / np.sqrt((np.pi * 2)**np.size(X, axis=1) * np.linalg.det(self.cov_))

        return [MultivariateGaussian._pdf_helper(X[i, :], norm_fact, self.mu_, self.cov_) for i in range(np.size(X, axis=0))]
        
        # raise NotImplementedError()
    
    @staticmethod
    def _pdf_helper(vec, norm_fact, mu, cov):
        centered = (vec - mu)
        power = -0.5 * np.linalg.multi_dot((centered, np.linalg.inv(cov), centered))
        return norm_fact * np.exp(power)
    
    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        
        # in this function i calculate the log-likelihood according to the derivation in q9 of the theoretical part
        n = X.shape[0]
        d = X.shape[1]

        norm_fact = 1 / np.sqrt(np.power((np.pi * 2), d) * np.linalg.det(cov))
        first_part = n * np.log(norm_fact)
        args_to_sum = np.apply_along_axis(lambda vec: -.5 * np.linalg.multi_dot(((vec - mu), np.linalg.inv(cov),
                                                                                (vec - mu))), arr=X, axis=1)
        return first_part + np.sum(args_to_sum)
