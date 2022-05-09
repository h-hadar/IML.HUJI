from typing import NoReturn

import numpy as np

from ...base import BaseEstimator
from ...metrics import loss_functions


class GaussianNaiveBayes(BaseEstimator):
	"""
	Gaussian Naive-Bayes classifier
	"""
	
	def __init__(self):
		"""
		Instantiate a Gaussian Naive Bayes classifier

		Attributes
		----------
		self.classes_ : np.ndarray of shape (n_classes,)
			The different labels classes. To be set in `GaussianNaiveBayes.fit`

		self.mu_ : np.ndarray of shape (n_classes,n_features)
			The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

		self.vars_ : np.ndarray of shape (n_classes, n_features)
			The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

		self.pi_: np.ndarray of shape (n_classes)
			The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
		"""
		super().__init__()
		self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None
	
	def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
		"""
		fits a gaussian naive bayes model

		Parameters
		----------
		X : ndarray of shape (n_samples, n_features)
			Input data to fit an estimator for

		y : ndarray of shape (n_samples, )
			Responses of input data to fit to
		"""
		self.classes_, class_sample_count = np.unique(y, return_counts=True)
		m = X.shape[0]
		if len(X.shape) == 1:
			X = X.reshape(-1, 1)
		self.pi_ = class_sample_count / m
		self.mu_ = [np.sum(X[y == k, :], axis=0) / class_sample_count[i]
					for i, k in enumerate(self.classes_)]
		self.vars_ = [np.sum(((X - self.mu_[i])[y == k, :]) ** 2, axis=0) / class_sample_count[i]
					  for i, k in enumerate(self.classes_)]
		self.vars_ = np.array(self.vars_)
	
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
		likelihood_mat = self.likelihood(X)
		n_samples = X.shape[0]
		response = np.zeros((n_samples,))
		for i, x_i in enumerate(X):
			sample_probability = self.pi_ @ likelihood_mat[i, :]
			response[i] = np.argmax(likelihood_mat[i, :] * self.pi_ / sample_probability)
		return response
	
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
		pdfs_mat = np.array([])
		for k in range(n_classes):
			mu_k = self.mu_[k]
			cov_k = np.diag(self.vars_[k])
			likelihood_for_class_k = GaussianNaiveBayes.likelihood_helper(mu_k, cov_k, X)
			pdfs_mat = likelihood_for_class_k if not pdfs_mat.size else np.c_[pdfs_mat, likelihood_for_class_k]
		return pdfs_mat
	
	@staticmethod
	def likelihood_helper(mu, cov, X):
		return np.apply_along_axis(lambda x: GaussianNaiveBayes._pdf(mu, cov, x), axis=1, arr=X)
	
	@staticmethod
	def _pdf(mu, cov, sample) -> float:
		d = len(sample)
		coeff = 1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
		power = -0.5 * np.linalg.multi_dot(((sample - mu), np.linalg.inv(cov), (sample - mu)))
		return coeff * np.exp(power)
	
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
