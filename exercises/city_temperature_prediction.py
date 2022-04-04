import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    
    # filter invalid data:
    full_data = pd.read_csv(filename).dropna()
    data = full_data.loc[np.logical_and(full_data['Temp'] < 50, full_data['Temp'] > -50)]
    data = data.loc[np.logical_and(data['Day'] <= 31, data['Day'] > 0)]
    data = data.loc[np.logical_and(data['Month'] <= 12, data['Month'] > 0)]
    data = data.loc[data['Year'] > 0]
    data['DayOfYear'] = pd.to_datetime(data['Date']).dt.dayofyear
    data.reset_index(inplace=True, drop=True)
    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    df = data.loc[data['Country'] == 'Israel']
    x = df['DayOfYear']
    y = df['Temp']
    df['Year'] = df['Year'].astype(str)
    fig = px.scatter(df, x="DayOfYear", y="Temp", color="Year")
    fig.update_layout(
        xaxis_title='Day of Year',
        yaxis_title='Temperature',
        title='Temperature as a function of day of year, color coded by year'
    )
    fig.show()

    # # Question 3 - Exploring differences between countries
    # raise NotImplementedError()
    #
    # # Question 4 - Fitting model for different values of `k`
    # raise NotImplementedError()
    #
    # # Question 5 - Evaluating fitted model on different countries
    # raise NotImplementedError()