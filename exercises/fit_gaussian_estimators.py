from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.io as pio
import plotly.graph_objs as go
pio.templates.default = "plotly_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, 1000)
    estimator = UnivariateGaussian()
    estimations = estimator.fit(X)
    print('(', estimations.mu_, ',', estimations.var_, ')')

    # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = np.arange(10, 1001, 10)
    avg_distance_from_expectation = [abs(estimator.fit(X[1:n_samples]).mu_ - 10) for n_samples in sample_sizes]
    print(avg_distance_from_expectation)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample_sizes, y=avg_distance_from_expectation,
                             name='Absolute distance between estimated- and true expectation',
                             line=dict(color='firebrick', width=4)))
    fig.update_layout(title='Absolute distance between estimated- and true expectation',
                      xaxis_title='Number of samples',
                      yaxis_title='Distance from expectation')
    
    # # Question 3 - Plotting Empirical PDF of fitted model
    # raise NotImplementedError()


def test_multivariate_gaussian():
    pass
    # Question 4 - Draw samples and print fitted model
    # raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    # raise NotImplementedError()

    # Question 6 - Maximum likelihood
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
