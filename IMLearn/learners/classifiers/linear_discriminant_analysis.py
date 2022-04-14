from typing import NoReturn

import numpy as np
from numpy.linalg import det

from ..gaussian_estimators import MultivariateGaussian
from .gaussian_naive_bayes import GaussianNaiveBayes
from ...base import BaseEstimator
from ...metrics import loss_functions


class LDA(BaseEstimator):
	"""
	Linear Discriminant Analysis (LDA) classifier

	Attributes
	----------
	self.classes_ : np.ndarray of shape (n_classes,)
		The different labels classes. To be set in `LDA.fit`

	self.mu_ : np.ndarray of shape (n_classes,n_features)
		The estimated features means for each class. To be set in `LDA.fit`

	self.cov_ : np.ndarray of shape (n_features,n_features)
		The estimated features covariance. To be set in `LDA.fit`

	self._cov_inv : np.ndarray of shape (n_features,n_features)
		The inverse of the estimated features covariance. To be set in `LDA.fit`

	self.pi_: np.ndarray of shape (n_classes)
		The estimated class probabilities. To be set in `LDA.fit`
	"""
	
	def __init__(self):
		"""
		Instantiate an LDA classifier
		"""
		super().__init__()
		self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None
	
	def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
		"""
		fits an LDA model.
		Estimates gaussian for each label class - Different mean vector, same covariance
		matrix with dependent features.

		Parameters
		----------
		X : ndarray of shape (n_samples, n_features)
			Input data to fit an estimator for

		y : ndarray of shape (n_samples, )
			Responses of input data to fit to
		"""
		self.classes_, class_sample_count = np.unique(y, return_counts=True)
		m = X.shape[0]
		self.pi_ = class_sample_count / m
		self.mu_ = [np.sum(X[y == k, :], axis=0) / class_sample_count[i] for i, k in enumerate(self.classes_)]
		n_features = X.shape[1]
		self.cov_ = np.zeros((n_features, n_features))
		for i, k in enumerate(self.classes_):
			mu_k = self.mu_[i]
			for x_j in X[y == k, :]:
				dist = (x_j - mu_k).reshape(-1, 1)
				self.cov_ += dist @ dist.T
		self.cov_ /= (m - len(self.classes_))
		self._cov_inv = np.linalg.inv(self.cov_)
	
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
		y_pred = np.zeros((X.shape[0],))
		n_classes = self.classes_.shape[0]
		linear_func_coeffs = [(self.__a_coeff(i), self.__b_coeff(i)) for i in range(n_classes)]
		for i, sample in enumerate(X):
			sample = sample.reshape(-1, 1)
			results = [((a.T @ sample) + b) for a, b in linear_func_coeffs]
			best_class_index = np.argmax(results)
			y_pred[i] = self.classes_[best_class_index]
		return y_pred
	
	# calculate sigma-1 * mu_k
	def __a_coeff(self, class_index: int):
		return self._cov_inv @ (self.mu_[class_index].reshape(-1, 1))
	
	# calculate log(pi_k) - 0.5 * mu_k * sigma-1 * mu_k
	def __b_coeff(self, class_index: int):
		mu_k = self.mu_[class_index].reshape(-1, 1)
		return np.log(self.pi_[class_index]) - 0.5 * mu_k.T @ self._cov_inv @ mu_k
	
	def likelihood(self, X: np.ndarray) -> np.ndarray:
		"""
		Calculate the likelihood of a given data over the estimated model

		Parameters
		----------
		X : np.ndarray of shape (n_samples, n_features)
			Input data to calculate its likelihood over the different classes.

		Returns
		-------
		likelihoods : np.ndarray of shape (n_samples, n_classes)
			The likelihood for each sample under each of the classes

		"""
		if not self.fitted_:
			raise ValueError("Estimator must first be fitted before calling `likelihood` function")
		n_classes = self.classes_.shape[0]
		
		n_classes = self.classes_.shape[0]
		pdfs_mat = np.array([])
		for k in range(n_classes):
			mu_k = self.mu_[k]
			likelihood_for_class_k = GaussianNaiveBayes.likelihood_helper(mu_k, self.cov_, X)
			pdfs_mat = likelihood_for_class_k if not pdfs_mat.size else np.c_[pdfs_mat, likelihood_for_class_k]
		return pdfs_mat
	
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
		return loss_functions.misclassification_error(y_true=y, y_pred=self.predict(X))
