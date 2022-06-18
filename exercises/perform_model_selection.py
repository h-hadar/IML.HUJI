from __future__ import annotations

from sklearn import datasets
from sklearn.linear_model import Lasso

from IMLearn.learners.regressors import PolynomialFitting, RidgeRegression, LinearRegression
from IMLearn.metrics import mean_square_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test
from utils import *


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.random.uniform(low=-1.2, high=2, size=n_samples)
    epsilon = np.random.normal(scale=noise, size=n_samples)
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    y = np.apply_along_axis(f, axis=0, arr=X)
    y = y + epsilon
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion=2 / 3)
    X_train, y_train, X_test, y_test = X_train.to_numpy().reshape(1, -1)[0, :], y_train.to_numpy().reshape(1, -1)[0,
                                                                                :], \
                                       X_test.to_numpy().reshape(1, -1)[0, :], y_test.to_numpy().reshape(1, -1)[0, :]
    fig = go.Figure()
    x_model = np.linspace(-1.5, 2, 100)
    y_model = f(x_model)
    fig.add_traces([
        go.Scatter(x=X_train, y=y_train, mode='markers',
                   marker=dict(color='green', size=7, symbol='square'),
                   name='Training Data'),
        go.Scatter(x=X_test, y=y_test, mode='markers',
                   marker=dict(color='orange', size=7, symbol='x'),
                   name='Test Data'),
        go.Scatter(x=x_model, y=y_model, mode='lines',
                   line=dict(color='blue', width=1.2),
                   name='True model')
    ])
    fig.update_layout(title=f'Training data, Test Data and True Model (noise={noise})',
                      legend_title='Type'
                      ).show()
    
    # # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    validation_errors = []
    training_errors = []
    degree_axis = np.arange(0, 11)
    for k in degree_axis:
        model = PolynomialFitting(k)
        t_error, v_error = cross_validate(model, X_train, y_train, mean_square_error)
        validation_errors.append(v_error)
        training_errors.append(t_error)
    fig = go.Figure()
    fig.add_traces([
        go.Scatter(x=degree_axis, y=validation_errors, mode='lines', name='Validation'),
        go.Scatter(x=degree_axis, y=training_errors, mode='lines', name='Train')
    ])
    fig.update_layout(title=f'Validation & training errors against polynomial degree (noise={noise})',
                      xaxis_title='Polynom degree',
                      yaxis_title='Error rate',
                      legend_title='Error Type'
                      ).show()
    
    # # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(validation_errors)
    model = PolynomialFitting(best_k)
    model.fit(X_train, y_train)
    print(f"Q3 (noise={noise}) --- K={best_k} is the argmin of validation errors, with:")
    print(f"Test Error: {model.loss(X_test, y_test):.4f}")
    print(f"Validation Error: {validation_errors[best_k]:.4f}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    data, labels = datasets.load_diabetes(return_X_y=True, as_frame=True)
    train_data, train_labels, test_data, test_labels = split_train_test(data, labels, 50 / len(data))
    train_x = train_data.to_numpy()
    train_y = train_labels.to_numpy()
    test_x = test_data.to_numpy()
    test_y = test_labels.to_numpy()
    
    # # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam_range_for_lasso = np.linspace(0.005, 4.5, n_evaluations)
    lam_range_for_ridge = np.linspace(0, 0.9, n_evaluations)
    lasso_errors = np.apply_along_axis(lambda x: cross_validate(Lasso(alpha=x), train_x, train_y, mean_square_error),
                                       axis=1,
                                       arr=lam_range_for_lasso.reshape(-1, 1))
    ridge_errors = np.apply_along_axis(lambda x: cross_validate(RidgeRegression(lam=x), train_x, train_y,
                                                                mean_square_error), axis=1,
                                       arr=lam_range_for_ridge.reshape(-1, 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Ridge Regression', 'Lasso Regression'])
    errors = ['Train', 'Validation']
    colors = ['blue', 'crimson']

    for i in range(2):
        fig.add_traces([
            go.Scatter(x=lam_range_for_ridge, y=ridge_errors[:, i], mode='lines',
                       name=errors[i],
                       line=dict(color=colors[i]))],
            rows=1, cols=1)
        fig.add_traces([
            go.Scatter(x=lam_range_for_lasso, y=lasso_errors[:, i], mode='lines', name=errors[i],
                       line=dict(color=colors[i]))],
            rows=1, cols=2)
    fig.update_layout(title="train- and validation errors as a function of the regularization parameter value",
                      xaxis_title='Regularization Parameter',
                      yaxis_title='Error')
    fig.show()
    
    # # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge = lam_range_for_ridge[np.argmin(ridge_errors[:, 1])]
    ridge_error = RidgeRegression(best_ridge).fit(train_x, train_y).loss(test_x, test_y)
    best_lasso = lam_range_for_lasso[np.argmin(lasso_errors[:, 1])]
    lasso_error = mean_square_error(Lasso(best_lasso).fit(train_x, train_y).predict(test_x), test_y)
    MSE_error = LinearRegression().fit(train_x, train_y).loss(test_x, test_y)
    print(f"Q8 -- Error of best Ridge model with lam={best_ridge:.4f}: {ridge_error:.4f}")
    print(f"Q8 -- Error of best Lasso model with lam={best_lasso:.4f}: {lasso_error:.4f}")
    print(f"Q8 -- test error of MSE model: {MSE_error:.4f}")
    

if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
