'''
Functions relating to creating the neural networks for river flow. These are trained on data from the CEH river gauge database using forcing from WRF model (precipitation and temperature)

'''

import numpy as np
import pickle
import datetime as dt
from scipy.stats.stats import pearsonr

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def windowSum(data, window):
	"""
	Helper function to prepare data summed over a given window for input into nn model

	Parameters
	----------
	data : array
		The data to be window summed
	window : integer
		Length (in array entries) over which to sum the data

	Returns
	-------
	output : array, length data
		The window summed data

	"""
	output = np.zeros([len(data),1])
	output[0:window,0] = np.sum(data[0:window])
	output[window:,0] = data[window:]	

	for i in range(1, window):
		output[window:,0] = output[window:,0] + data[0+i:-window+i]	
	
	return output

def lagData(data, lag):
	"""
	Helper function to prepare data lagged over a given number of entries for input into nn model

	Parameters
	----------
	data : array
		The data to be lagged
	window : integer
		Number of entries over which to lag the data

	Returns
	-------
	output : array, length data
		The lagged data

	"""

	output = np.zeros([len(data),1])
	output[0:lag,0] = data[0]
	output[lag:,0] = data[0:-lag]	

	return output

def baseline_model(input_len, node_width):
	"""
	Creates the basic dense layer neural network model ready for training. It has one layer of width node_width, and a second of width one

	Parameters
	----------
	input_len : integer
		The number of dimensions of the input
	
	node_width : integer
		The number of nodes within the dense layer	

	Returns
	-------
	model : Keras sequential model
		The dense layer nn model

	"""
	# create model
	model = Sequential()
	model.add(Dense(node_width, input_dim=input_len, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def modelErrorMetrics(preds, obs):
	"""
	Provides error metrics for model predictions, RMSE, Pearson correlation, and N-S efficiency	

	Parameters
	----------
	preds : array
		Array of model predictions
	obs : array
		Array of corresponding observations

	Returns
	-------
	stats : array
		Array with [RMSE, Correlation coefficient, N-S efficiency]

	"""
	rmse = np.sqrt(((preds - obs)**2).mean())
	print('RMSE - '+str(rmse))

	corr = pearsonr(preds, obs)
	print('Pearson correlation - '+str(corr[0]))

	nss = 1 - np.mean((preds - obs)**2)/ np.mean((obs - np.mean(obs))**2)
	print('Nash-Sutcliffe efficiency - '+str(nss))

	return np.asarray([rmse, corr[0], nss])

def runNNtrain(train_flux, train_data, no_epochs):
	"""
	Sets up and trains a neural net model training it to predict train_flux using train_data. An input scala

	Parameters
	----------
	train_flux : array
		1d array of flux data for the neural net to try and predict
	train_data : list
		List of independent data to use for predicting 
	no_epochs : integer
		The number of epochs to use in model training

	Returns
	-------
	nn_scaler : sklearn scaler	
		The scaling used on the independent data. When using the neural net to make predictions the independent data must 
		be run through this scaler first
	nn_model : the keras neural net model
		The nn model

	"""
	# scale the data ready for neural net fitting
	nn_scaler = StandardScaler()
	nn_scaler.fit(train_data)

	train_data_scale = nn_scaler.transform(train_data)

	# set up the neural net
	nn_model = baseline_model(train_data.shape[1], train_data.shape[1])
	# and train it
	nn_model.fit(train_data_scale,train_flux, nb_epoch=no_epochs, batch_size=5, verbose=0)

	# output the model and the error metrics
	return [nn_scaler, nn_model]

def nn_create_run_data(precipitation, temp,  precipitation_sums_lags, temp_sums_lags):
	"""
	Makes the independent variables (predicative/explanatory variables) for a neural network. It is given the
	precipitation and temperature time series and returns these along with them summed and lagged over the given intervals.

	Parameters
	----------
	precipitation : array
		A 1D array of the precipitation data series
	temp : array
		A 1D array of the temperature data series
	precipitation sums_lags : 2 tuple of integer lists
		The summed and lagged time series to produce, in the form:
			[[[precipitation sums], [precipitation lags]], [[temperature sums], [temperature lags]]
        i.e. [[[7,14], [1,2,3]], [[7,28], [1,2,3]]] would give precipitation summed for the last week and two weeks, and temperature
		summed for the last week and four weeks, and both lagged for 1,2, and 3 days

	Returns
	-------
	nn_run_data : list
		A list of all the independent data for the neural network

	"""
	nn_run_data = np.asarray([precipitation, temp]).T

	for this_sum in precipitation_sums_lags[0]:
		nn_run_data = np.append(nn_run_data, windowSum(nn_run_data[:,0], this_sum), axis=1)

	for this_lag in precipitation_sums_lags[1]:
		nn_run_data = np.append(nn_run_data, lagData(nn_run_data[:,0], this_lag), axis=1)

	for this_sum in temp_sums_lags[0]:
		nn_run_data = np.append(nn_run_data, windowSum(nn_run_data[:,1], this_sum), axis=1)

	for this_lag in temp_sums_lags[1]:
		nn_run_data = np.append(nn_run_data, lagData(nn_run_data[:,1], this_lag), axis=1)

	return nn_run_data



def create_generic_nn(river_dict, generic_model_files=None, train_dates=None, pt_sums_lags=None):
	"""
	For a dictionary of river objects this creates a neural net for predicting flows for a generic river. As well as the rain and 
	temperature profile for each river the size of the catchment is also passed to the nn training.


	Parameters
	----------
	river_dict : dictionary of river objects
		The rivers on which to train the generic model

	generic_model_files : optional, 2 tuple of strings
		The names to save the generic neural network, the first is the h5 file for the nn and the second for the
		training data. Default is ['generic_nn.h5', 'generic_nn_train']

	train_dates : optional, 2 tuple of datetime objects
		The dates to use for training [start_date, end_date]. The default is [dt.datetime(2005,1,1), dt.datetime(2007,12,30)]

	pt_sums_lags : optional, 2 tuple of 2 tuples of integer arrays
		The sums and lags of precipitation and temperature data to use in training the model, in the format
		[[[precipitation sums], [precipitation lags]], [[temperature sums], [temperature lags]]
		Default is [[[7,14,21,30,60], [1,2,3,4,5,6]], [[7,14,28], [1,2,3,4,5,6]]]

	Returns
	-------
	model_file : str
		Name of the file the generic model is saved to

	nn_dict : dictionary 
		Dictionary containing the neural net training data, and the scaler for pre-scaling input data	

	"""

	if generic_model_files is None:
		generic_model_files = ['generic_nn.h5', 'generic_nn_train']

	if pt_sums_lags is None:
		precipitation_sums_lags = [[7,14,21,30,60], [1,2,3,4,5,6]]
		temp_sums_lags = [[7,14,28], [1,2,3,4,5,6]]
	else:
		precipitation_sums_lags = pt_sums_lags[0] 
		temp_sums_lags = pt_sums_lags[1]

	if train_dates is None:
		train_dates = [dt.datetime(2005,1,1), dt.datetime(2007,12,30)]

	# get the combined data from all the rivers in river_dict

	all_flux = np.empty(1)
	all_flux[:] = np.nan

	all_train_data = np.empty([1,23])
	all_train_data[:] = np.nan

	# collate the training data from every river, appending the catchment size
	for this_river in river_dict.values():
		print('Adding data from ' + this_river.river_name)
		try:
			[this_flux_data, this_train_data] = this_river.prepNeuralNetData(train_dates, True)[0:2]

			this_train_data = np.append(this_train_data, np.ones([len(this_train_data),1])*this_river.catchment_area, axis=1)

			all_flux = np.append(all_flux, this_flux_data)
			all_train_data = np.append(all_train_data, this_train_data, axis=0)

		except TypeError:
			print(this_river.river_name + ': No flux data to add')	


	all_flux = all_flux[1:]
	all_train_data = all_train_data[1:,:]

	[generic_scaler, generic_model] = runNNtrain(all_flux, all_train_data, 500)

	# save the model seperately as it don't like being pickled
	generic_model.save(generic_model_files[0])

	# and save out the rest of the relevant data
	nn_dict = {'all_flux':all_flux, 'all_train_data':all_train_data, 'nn_scaler':generic_scaler}

	with open(generic_model_files[1], 'wb') as output_file:
		pickle.dump(nn_dict, output_file, pickle.HIGHEST_PROTOCOL)
	output_file.close()

	return [generic_model_files[0], nn_dict]

