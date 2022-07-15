import math
from typing import Tuple, List, Callable, Type

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule
from IMLearn.desent_methods import FixedLR, GradientDescent, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test

pio.templates.default = "plotly_white"


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])
    
    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []
    
    def my_callback(**kwargs):
        values.append(kwargs['val'])
        weights.append(kwargs['weights'])
    
    return my_callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for objective in [L1, L2]:
        convergence_fig = go.Figure()
        min_loss = np.inf
        for step_size in etas:
            lr = FixedLR(step_size)
            f = objective(init)
            callback, values, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(learning_rate=lr, callback=callback)
            solution = gd.fit(f, None, None)
            min_loss = min(min_loss, np.min(values))
            descent_path = np.r_[weights]
            fig = plot_descent_path(module=objective, descent_path=descent_path,
                                    title=f" - Objective: {objective.__name__}, Learning Rate: {step_size}")
            fig.write_image(f"ex6/q1/{objective.__name__}-{step_size}.png", width=1300, height=800)
            convergence_fig.add_trace(go.Scatter(x=np.arange(1, 1001), y=values, mode='lines',
                                                 name=f"eta={step_size}"))
        print(f"Minimal loss on objective {objective.__name__} using fixed LR: {min_loss:.4f}")
        convergence_fig.update_layout(title=f"{objective.__name__} - convergence rate",
                                      xaxis_title="iteration", yaxis_title="norm"
                                      ).write_image(f"ex6/q3/{objective.__name__}-convergence.png",
                                                    width=1300, height=800)


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    gamma_dict = dict()
    min_loss = math.inf
    for gamma in gammas:
        callback, values, weights = get_gd_state_recorder_callback()
        lr = ExponentialLR(base_lr=eta, decay_rate=gamma)
        f = L1(init)
        gd = GradientDescent(learning_rate=lr, callback=callback)
        gd.fit(f, None, None)
        gamma_dict[gamma] = (values, weights)
        print(f"Minimal loss on objective L1 using exponential LR, gamma={gamma}: {np.min(values):.4f}")
    
    # Plot algorithm's convergence for the different values of gamma
    convergence_fig = go.Figure()
    for gamma in gamma_dict:
        convergence_fig.add_trace(go.Scatter(x=np.arange(1, 1001), y=gamma_dict[gamma][0], mode='lines',
                                             line=dict(width=0.6),
                                             name=f"gamma={gamma}"))
    convergence_fig.update_layout(title="L1 - convergence rate with exponential decaying step size",
                                  xaxis_title="iteration", yaxis_title="norm"
                                  ).write_image("ex6/q5/L1-convergence.png",
                                                width=1300, height=800)
    
    # Plot descent path for gamma=0.95
    fig = plot_descent_path(module=L1, descent_path=np.r_[gamma_dict[0.95][1]],
                            title=f" - Objective: L1, Exponential Decaying Learning Rate: eta={eta}, gamma={0.95}")
    fig.write_image(f"ex6/q7/L1-{0.95}.png", width=1300, height=800)


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    solver = GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4))
    # Plotting convergence rate of logistic regression over SA heart disease data
    logistic = LogisticRegression(
        solver=solver
    )
    logistic.fit(X=X_train.to_numpy(), y=y_train.to_numpy())
    probs = logistic.predict_proba(X=X_train.to_numpy())
    fpr, tpr, thresholds = roc_curve(y_true=y_train, y_score=probs)
    roc_fig = go.Figure()
    roc_fig.add_traces([go.Scatter(x=fpr, y=tpr,
                                   mode='lines+markers', name='Logistic', text=thresholds,
                                   ),
                        go.Scatter(x=np.linspace(0, 1, 100),
                                   y=np.linspace(0, 1, 100),
                                   mode='lines', line=dict(color="black", dash='dash'),
                                   name='No skill')
                        ])
    roc_fig.layout = go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                               xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                               yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))
    roc_fig.write_image("ex6/q8-ROC.png", width=1300, height=800)
    roc_fig.show()
    # Q9
    best_alpha = thresholds[np.argmax(tpr - fpr)]
    pred_probs = logistic.predict_proba(X=X_test.to_numpy())
    pred_labels = np.where(pred_probs >= best_alpha, 1, 0)
    print(f"Q9 -- best threshold-alpha*: {best_alpha}")
    model = LogisticRegression(
        solver=solver, alpha=best_alpha
    )
    model.fit(X=X_train.to_numpy(), y=y_train.to_numpy())
    # error = misclassification_error(y_test.to_numpy(), pred_labels)
    error = model.loss(X_test.to_numpy(), y_test.to_numpy())
    print(f"Test error for best threshold: "
          f"{error}")
    
    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    for penalty_type in ["l1", "l2"]:
        best_gamma, best_validation = None, np.inf
        for gamma in {0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1}:
            # print(f"calculating gamma={gamma}")
            logistic_regularized = LogisticRegression(
                penalty=penalty_type, solver=solver, lam=gamma
            )
            t_score, v_score = \
                cross_validate(estimator=logistic_regularized, X=X_train.to_numpy(), y=y_train.to_numpy(),
                               scoring=misclassification_error, cv=5)
            if v_score < best_validation:
                best_gamma, best_validation = gamma, v_score
        test_error = LogisticRegression(
            penalty=penalty_type, solver=solver, lam=best_gamma
        ).fit(X=X_train.to_numpy(), y=y_train.to_numpy()).loss(X=X_test.to_numpy(), y=y_test.to_numpy())
        print(f"Q10 - {penalty_type}: "
              f"best regularization = {best_gamma} , "
              f"test error: {test_error}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
