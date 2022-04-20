from typing import Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

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


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
	"""
	Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
	"""
	for filename in ["gaussian1.npy", "gaussian2.npy"]:
		# Load dataset
		features_mat, true_classes = load_dataset(f"../datasets/{filename}")
		
		# Fit models and predict over training set
		LDA_classifier = LDA()
		LDA_classifier.fit(features_mat, true_classes)
		lda_predictions = LDA_classifier.predict(features_mat)
		lda_accuracy = accuracy(true_classes, lda_predictions)
		
		naive_bayes_classifier = GaussianNaiveBayes()
		naive_bayes_classifier.fit(features_mat, true_classes)
		bayes_predictions = naive_bayes_classifier.predict(features_mat)
		bayes_accuracy = accuracy(true_classes, bayes_predictions)
		
		# Plot a figure with two subplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
		# on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
		symbols = np.array(["circle", "star", "triangle-up"])
		center_symbol = "x"
		fig = make_subplots(rows=1, cols=2,
							subplot_titles=("LDA Classifier, accuracy: {:.3f}".format(lda_accuracy),
											"Naive Bayes Classifier, accuracy: {:.3f}".format(bayes_accuracy)))
		lda_centers = np.array(LDA_classifier.mu_)
		fig.add_traces(
			[go.Scatter(x=features_mat[:, 0],
					   y=features_mat[:, 1],
					   mode='markers', showlegend=False,
					   marker=dict(size=10, color=lda_predictions, symbol=symbols[true_classes.astype(int)],
								   line=dict(color="gray", width=.5))),
			go.Scatter(x=lda_centers[:, 0], y=lda_centers[:, 1],
					   marker=dict(size=10, color='black', symbol=center_symbol, line=dict(color='white', width=1)),
					   mode='markers',
					   showlegend=False)],
			rows=1, cols=1
		)
		bayes_centers = np.array(naive_bayes_classifier.mu_)
		fig.add_traces(
			[go.Scatter(x=features_mat[:, 0],
					   y=features_mat[:, 1],
					   mode='markers', showlegend=False,
					   marker=dict(size=9, color=bayes_predictions, symbol=symbols[true_classes.astype(int)],
								   line=dict(color="gray", width=.5))),
			go.Scatter(x=bayes_centers[:, 0], y=bayes_centers[:, 1],
					   marker=dict(size=10, color='black', symbol=center_symbol, line=dict(color='white', width=1)),
					   mode='markers',
					   showlegend=False)],
			rows=1, cols=2
		)
		fig.update_layout(
			title=f'Classification Performance Comparison on Dataset {filename}'
		)
		fig.show()


if __name__ == '__main__':
	np.random.seed(0)
	# run_perceptron()
	compare_gaussian_classifiers()
