from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.io as pio
import plotly.graph_objs as go
import plotly.express as px

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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample_sizes, y=avg_distance_from_expectation,
                             name='Absolute distance between estimated- and true expectation',
                             line=dict(color='firebrick', width=4)))
    fig.update_layout(title='Absolute distance between estimated- and true expectation',
                            xaxis_title='Number of samples',
                            yaxis_title='Distance from expectation')
    fig.show()
    
    # # Question 3 - Plotting Empirical PDF of fitted model
    fig = go.Figure()
    pdf_vector = estimator.pdf(X)
    fig.add_trace(go.Scatter(x=X, y=pdf_vector,
                             mode="markers", marker=dict(color="red", size=1.5)))
    fig.update_layout(title='PDF for each sample drawn',
                            xaxis_title='Sample Value',
                            yaxis_title='PDF')
    fig.show()
    
    # test the log-likelihood function:
    # print("log-likelihood of (10,1):", estimator.log_likelihood(10, 1, X))
    # print("log-likelihood of (10,5):", estimator.log_likelihood(10, 5, X))
    # print("log-likelihood of (10,2):", estimator.log_likelihood(10, 2, X))
    # print("log-likelihood of (0,1):", estimator.log_likelihood(0, 1, X))
    # print("log-likelihood of (5,2):", estimator.log_likelihood(5, 2, X))


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0,0,4,0])
    sigma = np.array(([[1,.2,0,.5], [.2,2,0,0], [0,0,1,0], [.5, 0,0,1]]))
    X = np.random.multivariate_normal(mu, sigma, 1000)
    estimator = MultivariateGaussian()
    estimations = estimator.fit(X)
    print(estimations.mu_)
    print(estimations.cov_)

    # # pdf test:
    # t = estimator.pdf(X)
    # print(t)
    # print(np.size(t))

    # Question 5 - Likelihood evaluation
    # raise NotImplementedError()
    # print("log-likelihood of real params:")
    # print(MultivariateGaussian.log_likelihood(mu, sigma, X))
    size_of_heatmap = 200
    f1 = np.linspace(-10, 10, size_of_heatmap)
    f3 = np.linspace(-10, 10, size_of_heatmap)
    likelihood_mat = np.zeros((np.size(f1), np.size(f1)))
    for (row_i, row_val) in enumerate(f1):
        for (col_i, col_val) in enumerate(f3):
            likelihood_mat[row_i, col_i] = MultivariateGaussian.log_likelihood(mu=np.array([row_val, 0, col_val, 0]),
                                                                              cov=sigma,
                                                                              X=X)
        
    fig = px.imshow(likelihood_mat, labels=dict(x='f3 - 3rd coordinate of mu', y='f1 - 1st coordinate of mu',
                                                color='log-likelihood'), x=f1, y=f3, origin='lower')
    fig.show()
    
    # Question 6 - Maximum likelihood
    max_f1_f3 = np.unravel_index(np.argmax(likelihood_mat, axis=None), likelihood_mat.shape)
    print("The f1, f3 values which yielded maximal likelihood:", "%.3f" % f1[max_f1_f3[0]], "%.3f" % f3[max_f1_f3[1]])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
