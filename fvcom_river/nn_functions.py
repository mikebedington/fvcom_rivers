'''
Functions relating to creating the neural networks for river flow. These are trained on data from the CEH river gauge database using forcing from WRF model (precipitation and temperature)

The key functions are:

runNNtrain - trains a neural network
baseline_model - the setup of the neural net
create_generic_nn - creates a neural net trained on data from multiple rivers (in river_dict), creating a generic model which can be applied for ungauged rivers

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
	output = np.zeros([len(data),1])
	output[0:window,0] = np.sum(data[0:window])
	output[window:,0] = data[window:]	

	for i in range(1, window):
		output[window:,0] = output[window:,0] + data[0+i:-window+i]	
	
	return output

def lagData(data, lag):
	output = np.zeros([len(data),1])
	output[0:lag,0] = data[0]
	output[lag:,0] = data[0:-lag]	

	return output

def baseline_model(input_len, node_width):
    # create model
    model = Sequential()
    model.add(Dense(node_width, input_dim=input_len, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def modelErrorMetrics(preds, obs):
    rmse = np.sqrt(((preds - obs)**2).mean())
    print('RMSE - '+str(rmse))

    corr = pearsonr(preds, obs)
    print('Pearson correlation - '+str(corr[0]))

    nss = 1 - np.mean((preds - obs)**2)/ np.mean((obs - np.mean(obs))**2)
    print('Nash-Sutcliffe efficiency - '+str(nss))

    return np.asarray([rmse, corr[0], nss])

def runNNtrain(train_flux, train_data, no_epochs):

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
	if generic_model_files is None:
		generic_model_files = ['generic_nn.h5', 'generic_nn_train']

	if pt_sums_lags is None:
		precipitation_sums_lags = [[7,14,21,30,60], [1,2,3,4,5,6]]
		temp_sums_lags = [[7,14,28], [1,2,3,4,5,6]]

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

