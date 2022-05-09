from typing import Tuple

from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metalearners import AdaBoost
from utils import *


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y) = generate_data(train_size, noise)
    (test_X, test_y) = generate_data(test_size, noise)
    
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_boost = AdaBoost(DecisionStump, n_learners)
    ada_boost.fit(train_X, train_y)
    x_axis = list(range(1, n_learners + 1))
    train_error = [ada_boost.partial_loss(train_X, train_y, i) for i in x_axis]
    test_error = [ada_boost.partial_loss(test_X, test_y, i) for i in x_axis]
    fig = go.Figure()
    fig.add_traces(
        [
            go.Scatter(x=x_axis,
                       y=train_error,
                       mode='lines',
                       line=dict(color='blue', width=1.2),
                       name='Train Error'),
            go.Scatter(x=x_axis,
                       y=test_error,
                       mode='lines',
                       line=dict(color='springgreen', width=1.2),
                       name='Test Error')
        ])
    fig.update_layout(title=f'Train- and Test-Errors as a Function of Number of Fitted Weak Learners (noise={noise})',
                      xaxis_title='Fitted Learners',
                      yaxis_title='Error (normalized)',
                      legend_title='Error type'
                      )
    fig.show()
    
    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    symbols = np.array(["circle", "x"])
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f'Using {t} weak learners' for t in T])
    scatter_test_points = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                     marker=dict(color=test_y, colorscale=[custom[0], custom[-1]], size=4.5,
                                                 line=dict(color="black", width=.5)))
    for i, iteration_count in enumerate(T):
        fig.add_traces(
            [decision_surface(lambda data: ada_boost.partial_predict(data, iteration_count), lims[0], lims[1],
                              showscale=False),
             # add test set, colored by true labels
             scatter_test_points],
            rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(title=f'Decision Boundaries and Test Set Labels (noise={noise})')
    fig.show()
    
    # Question 3: Decision surface of best performing ensemble
    best_ensemble_size = np.argmin(test_error) + 1
    lowest_error = test_error[best_ensemble_size - 1]
    fig = go.Figure()
    fig.add_traces([decision_surface(lambda data: ada_boost.partial_predict(data, best_ensemble_size), lims[0], lims[1],
                                     showscale=False),
                    scatter_test_points])
    fig.update_layout(title=f"Best Ensemble size: {best_ensemble_size}, accuracy {1 - lowest_error} (noise={noise})")
    fig.show()
    #
    # # Question 4: Decision surface with weighted samples
    sizes = ada_boost.D_ / max(ada_boost.D_) * 12
    symbols = ['x', 'square']
    fig = go.Figure()
    fig.add_traces([decision_surface(ada_boost.predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                               mode="markers", showlegend=False,
                               marker=dict(color=train_y,
                                           colorscale=['darkmagenta', 'darkblue'],
                                           size=sizes,
                                           line=dict(color="black", width=.2))),
                    ])
    fig.update_layout(title=f'Final Decision Boundary, with training data point size depicting its weight in the '
                            f'distribution (noise={noise})').show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
    # raise NotImplementedError()
