from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
# from IMLearn.utils import split_train_test
import sklearn

import numpy as np
import pandas as pd


def _to_date_number(features: pd.DataFrame, keys: list) -> None:
    for field in keys:
        features[field] = pd.to_datetime(features[field])
        features[field] = features[field].apply(lambda x: x.value)


def _to_day_of_week(features, keys):
    for field in keys:
        new_key = field + "_dayofweek"
        features[new_key] = pd.to_datetime(features[field])
        features[new_key] = features[new_key].apply(lambda x: x.dayofweek)


def _add_new_cols(features):
    _to_day_of_week(features, ["checkin_date", "checkout_date", "booking_datetime"])
    features['stay_days'] = (pd.to_datetime(features['checkout_date'])
                             - pd.to_datetime(features['checkin_date']))
    features['stay_days'] = features['stay_days'].apply(lambda x: x.days)
    features['days_till_vacation'] = (pd.to_datetime(features['checkin_date'])
                                      - pd.to_datetime(features['booking_datetime']))
    features['days_till_vacation'] = features['days_till_vacation'].apply(lambda x: x.days)
    features['is_checkin_on_weekend'] = features['checkin_date_dayofweek'].apply(lambda x: x > 4)

    # for title in []:
    #     del features[title]


def _add_categories(features, full_data, titles):
    for title in titles:
        features = pd.concat((features, pd.get_dummies(full_data[title], drop_first=True)),
                         axis=1)
    return features


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename)  #.dropna().drop_duplicates()
    good_categories = ["booking_datetime", "hotel_star_rating", "checkin_date",
                       "checkout_date", "hotel_live_date", "cancellation_datetime",
                       "guest_is_not_the_customer"]
    features = full_data[good_categories]
    _add_new_cols(features)
    features = _add_categories(features, full_data,
                               [#'accommadation_type_name', 'customer_nationality', 'hotel_country_code',
                                'charge_option'])

    # features["cancellation_datetime"].replace(np.nan, "", inplace=True)
    _to_date_number(features, ["booking_datetime", "checkin_date", "checkout_date", "hotel_live_date"])
    labels = features["cancellation_datetime"].between("2010-07-12", "2033-13-12").astype(int)
    del features["cancellation_datetime"]
    return features, labels


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pred = estimator.predict(X)
    pd.DataFrame(pred, columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(df, cancellation_labels)

    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set
    evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")
