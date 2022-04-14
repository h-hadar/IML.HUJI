from typing import Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from IMLearn.metrics.loss_functions import accuracy

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
	ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

	Parameters
	----------
	filename: str
		Path to .npy data file

	Returns
	-------
	X: ndarray of shape (n_samples, 2)
		Design matrix to be used

	y: ndarray of shape (n_samples,)
		Class vector specifying for each sample its class

	"""
	temp = np.load(filename)
	return temp[:, 0:2], temp[:, 2]


def run_perceptron():
	"""
	Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

	Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
	as a function of the training iterations (x-axis).
	"""
	for name, filename in [("Linearly Separable", "linearly_separable.npy"),
						   ("Linearly Inseparable", "linearly_inseparable.npy")]:
		# Load dataset
		features_mat, labels = load_dataset(f"../datasets/{filename}")
		
		# Fit Perceptron and record loss in each fit iteration
		losses = []
		perceptron = Perceptron(callback=lambda p, x_i, y_i: losses.append(p.loss(features_mat, labels)))
		perceptron.fit(features_mat, labels)
		
		# Plot figure
		fig = px.scatter(x=range(1, len(losses) + 1), y=losses)
		fig.update_traces(mode="lines", line=dict(color='mediumaquamarine'))
		fig.update_layout(
			xaxis_title='Iteration no.',
			yaxis_title='Loss recorded',
			title=f'Loss in the progress of fitting the perceptron for {name} data'
		)
		fig.show()


def compare_gaussian_classifiers():
	"""
	Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
	"""
	for filename in ["gaussian1.npy", "gaussian2.npy"]:
		# Load dataset
		features_mat, true_classes = load_dataset(f"../datasets/{filename}")
		
		# Fit models and predict over training set
		linear_analysis = LDA()
		linear_analysis.fit(features_mat, true_classes)
		lda_predictions = linear_analysis.predict(features_mat)
		lda_accuracy = accuracy(true_classes, lda_predictions)
		
		naive_bayes = GaussianNaiveBayes()
		naive_bayes.fit(features_mat, true_classes)
		bayes_predictions = naive_bayes.predict(features_mat)
		bayes_accuracy = accuracy(true_classes, bayes_predictions)
		
		# Plot a figure with two subplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
		# on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
		fig = make_subplots(rows=1, cols=2,
							subplot_titles=("LDA Classifier, accuracy: {:.3f}".format(lda_accuracy),
											"Naive Bayes Classifier, accuracy: {:.3f}".format(bayes_accuracy)))
		fig.add_trace(
			go.Scatter(x=features_mat[:, 0],
					   y=features_mat[:, 1],
					   mode='markers',
					   marker=dict(color=lda_predictions, symbol=true_classes)),
			row=1, col=1
		)
		fig.add_trace(
			go.Scatter(x=features_mat[:, 0],
					   y=features_mat[:, 1],
					   mode='markers',
					   marker=dict(color=bayes_predictions, symbol=true_classes)),
			row=1, col=2
		)
		fig.update_layout(
			title=f'Classification Performance Comparison on Dataset {filename}'
		)
		fig.show()


if __name__ == '__main__':
	np.random.seed(0)
	run_perceptron()
	compare_gaussian_classifiers()
