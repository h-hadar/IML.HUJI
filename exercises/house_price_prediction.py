import copy
import random

import plotly.offline

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
 
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    features = copy.copy(full_data)
    del features["id"]
    
    # drop rows that contain invalid data:
    features.drop(features.loc[features['price'] <= 1000].index, inplace=True)
    features.drop(features.loc[features['bedrooms'] <= 0].index, inplace=True)
    features.drop(features.loc[features['floors'] < 1].index, inplace=True)

    features['floor'] = features['floors'].astype(int)
    
    # handle year_renovated, so it doesn't contain zeros, and add is_renovated binary column
    features['is_renovated'] = np.where(features["yr_renovated"] == 0, -1, 1)
    features["yr_renovated"] = np.where(features["yr_renovated"] != 0, features["yr_renovated"], features["yr_built"])
    year_sold = pd.to_datetime(features["date"], infer_datetime_format=True).dt.year
    features['age_when_sold'] = year_sold - features['yr_built']
    features['time_from_last_renovation'] = features['yr_renovated'] - features['yr_built']
    # features['date'] = pd.to_datetime(features["date"], infer_datetime_format=True).apply(lambda x: x.value)
    del features['date']
    
    # handle the basement zeros problem (they might skew the linear analysis of basement size)
    features["no_basement"] = np.where(features["sqft_basement"] == 0, 1, 0)
    features["1-500_basement"] = np.where(np.logical_and(0 < features["sqft_basement"], 500 >= features["sqft_basement"]), 1, 0)
    features["500-1000_basement"] = np.where(np.logical_and(500 < features["sqft_basement"], 1000 >= features["sqft_basement"]), 1, 0)
    features["1000-1500_basement"] = np.where(np.logical_and(1000 < features["sqft_basement"], 1500 >= features[
        "sqft_basement"]), 1, 0)
    features["1500-2000_basement"] = np.where(np.logical_and(1500 < features["sqft_basement"], 2000 >= features[
        "sqft_basement"]), 1, 0)
    features["2000+_basement"] = np.where(2000 < features["sqft_basement"], 1, 0)
    del features['sqft_basement']
    
    categorical_fields = ["zipcode"]
    for field in categorical_fields:
        features[field] = features[field].astype(str)
        features = pd.concat((features, pd.get_dummies(features[field], drop_first=True)), axis=1)
        del features[field]
        
    features.reset_index(drop=True, inplace=True)
    labels_vec = features['price']
    del features['price']
    return features, labels_vec


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    sigma_y = np.std(y)
    sigma_x = np.std(X, axis=0)  # the std for each feature
    for feature_name in X:
        try:
            cov = np.cov(X[feature_name], y)[0, 1]
            pearson_corr = cov / (sigma_x[feature_name] * sigma_y)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X[feature_name], y=y,
                                     name='Scatter of feature against response',
                                     mode='markers'))
            fig.update_layout(title=f'Scatter of response against {feature_name}, Pearson Correlation: {pearson_corr}',
                              xaxis_title=feature_name,
                              yaxis_title='Response (House price)')
            fig.update_traces(marker=dict(size=7,
                                          line=dict(width=1,
                                                    color='white'),
                                          color='slateblue'),
                              selector=dict(mode='markers'))
            plotly.offline.plot(fig, filename=output_path + f"\\{feature_name}.html", auto_open=False)
        except TypeError:
            print(f"problem with column {feature_name}")
        



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    features, labels = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(features, labels, output_path=".\ex2\q2_plots")

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(features, labels, train_proportion=0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    train_sizes = np.array(range(10, 101)) / 100

    def test_performance(percent):
        losses = []
        for i in range(10):
            sub_data = train_x.sample(frac=percent)
            sub_labels = train_y.iloc[sub_data.index]
            model = LinearRegression()
            model.fit(sub_data, sub_labels)
            loss = model.loss(test_x, test_y)
            losses.append(loss)
        return np.mean(losses), np.std(losses)
    
    results = [test_performance(size) for size in train_sizes]
    y = [r[0] for r in results]
    y_upper = [r[0] + 2 * r[1] for r in results]
    y_lower = [r[0] - 2 * r[1] for r in results]
    x = train_sizes*100

    fig = go.Figure([
        go.Scatter(
            name='Mean-Loss across 10 training sets',
            x=x,
            y=y,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='mean + 2*std',
            x=x,
            y=y_upper,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='mean - 2*std',
            x=x,
            y=y_lower,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(
        xaxis_title='Training set size (% of total training set)',
        yaxis_title='Mean LOSS on test set',
        title='Mean LOSS on test data with varying size of training set',
        hovermode="x"
    )
    fig.show()
    plotly.offline.plot(fig, filename=".\ex2\linear_regressin.html", auto_open=False)
        
        