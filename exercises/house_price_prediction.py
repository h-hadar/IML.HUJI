import copy

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
 
    full_data = pd.read_csv(filename).dropna()
    features = copy.copy(full_data)
    n_samples = full_data.shape[0]
    del features["id"]
    
    # drop rows that contain invalid data:
    features.drop(features.loc[features['price'] <= 0].index, inplace=True)

    features['date'] = pd.to_datetime(features["date"], infer_datetime_format=True).apply(lambda x: x.value)
    
    # handle year_renovated, so it doesn't contain zeros, and add is_renovated binary column
    features['is_renovated'] = np.where(features["yr_renovated"] == 0, 0, 1)
    features["yr_renovated"] = np.where(features["yr_renovated"] != 0, features["yr_renovated"], features["yr_built"])
    
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
        features = pd.concat((features, pd.get_dummies(features[field])), axis=1)
        del features[field]
        
    labels = full_data['price']
    del features['price']
    return features, labels


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
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, label = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
