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
	israel_df = data.loc[data['Country'] == 'Israel']
	israel_df['Year'] = israel_df['Year'].astype(str)
	israel_day_of_year = israel_df['DayOfYear']
	israel_temp = israel_df['Temp']
	fig = px.scatter(israel_df, x="DayOfYear", y="Temp", color="Year")
	fig.update_layout(
		xaxis_title='Day of Year',
		yaxis_title='Temperature',
		title='Temperature as a function of day of year, color coded by year'
	)
	fig.show()
	std_by_month = israel_df.groupby('Month').Temp.agg(np.std)
	fig = px.bar(std_by_month)
	fig.update_layout(
		yaxis_title='Std of temperature',
		title='Standard deviation of temperature in each month'
	)
	fig.show()
	
	# # Question 3 - Exploring differences between countries
	
	mean_temps = data.groupby(['Country', 'Month']).Temp.agg(np.mean).reset_index()
	temp_stds = data.groupby(['Country', 'Month']).Temp.agg(np.std).reset_index()
	fig = px.line(mean_temps, x='Month', y='Temp', color='Country', error_y=temp_stds['Temp'])
	fig.update_layout(
		yaxis_title='Mean Temperature',
		title='Mean Temperature across months in different countries'
	)
	fig.show()
	israel_df['Year'] = israel_df['Year'].astype(int)
	
	#
	# # Question 4 - Fitting model for different values of `k`
	trainX, trainY, testX, testY = split_train_test(israel_day_of_year, israel_temp, train_proportion=.75)
	pol_degrees = range(1, 11)
	losses = []
	for d in pol_degrees:
		polyfit = PolynomialFitting(d)
		polyfit.fit(trainX, trainY)
		cur_loss = polyfit.loss(testX, testY)
		losses.append(cur_loss)
		print(f"Test error for polynomial degree {d}: {cur_loss}")
	fig = px.bar(x=pol_degrees, y=losses)
	fig.update_layout(
		xaxis_title='Degree',
		yaxis_title='MSE',
		title='Mean Loss with Different Polynomial Degree of Fitting'
	)
	fig.show()
	#
	# # Question 5 - Evaluating fitted model on different countries
	chosen_degree = np.argmin(losses) + 1
	polyfit = PolynomialFitting(chosen_degree)
	polyfit.fit(israel_day_of_year, israel_temp)
	data_by_country = data.groupby('Country')
	
	def calc_loss(grp):
		x = np.array(grp.DayOfYear)
		y = np.array(grp.Temp)
		return polyfit.loss(x, y)

	country_names = []
	error_by_country = []
	for country_name, country_data in data_by_country:
		country_names.append(country_name)
		error_by_country.append(calc_loss(country_data))
		
	fig = px.bar(x=country_names, y=error_by_country)
	fig.update_layout(
		xaxis_title='Country',
		yaxis_title='LOSS',
		title='The LOSS values across countries with a model fitted for Israel'
	)
	fig.show()

