import numpy as np
import datetime as dt
from calendar import monthrange
import pickle
import csv
import copy
from sklearn import datasets, linear_model
from math import radians, cos, sin, asin, sqrt
import gc

import PyFVCOM as pf

import fvcom_river.read_wrf_variable as rv


def make_WRF_squares_list(wrf_directory):
	'''
	Get the corners of each of the WRF grid squares


	Returns
	-------

	corners - a 2x4xn of the lat/lon of each of the four corners of the WRF grid squares

	'''


	winfo = rv.winfo = rv.get_WRF_info(wrf_directory)
	lat_grid = winfo['XLAT']
	lon_grid = winfo['XLON']
	differences = np.asarray([lat_grid[1:-1,0:-2] - lat_grid[1:-1,1:-1],lat_grid[1:-1,1:-1] - lat_grid[1:-1,2:], lon_grid[0:-2,1:-1] - lon_grid[1:-1, 1:-1] , lon_grid[1:-1, 1:-1] - lon_grid[2:,1:-1]])

	lat_grid = lat_grid[1:-1,1:-1]
	lon_grid = lon_grid[1:-1,1:-1]

	corners = np.asarray([[lat_grid - np.squeeze(differences[0,:,:]) , lon_grid - np.squeeze(differences[2,:,:])],
 						[lat_grid - np.squeeze(differences[0,:,:]) , lon_grid + np.squeeze(differences[3,:,:])],
						[lat_grid + np.squeeze(differences[1,:,:]) , lon_grid + np.squeeze(differences[3,:,:])],
						[lat_grid + np.squeeze(differences[1,:,:]) , lon_grid - np.squeeze(differences[2,:,:])]])

	return corners

def read_csv_unheaded(filename, cols):

	output = []

	for i in range(0,cols):
		this_list = []

		with open(filename, 'rt') as this_file:
			this_file_data = csv.reader(this_file)

			for row in this_file_data:
				this_list.append(row[i])
		
		output.append(this_list)
	return output

def read_csv_dict(filename):
	output = {}

	with open(filename, 'rt') as this_file:
		this_file_data = csv.DictReader(this_file)

		for row in this_file_data:
			test_row = row

		this_keys = test_row.keys()

		for tk in this_keys:
			this_entry = []

			with open(filename, 'rt') as this_file:
				this_file_data = csv.DictReader(this_file)

				for row in this_file_data:
					this_entry.append(row[tk])

			output[tk] = this_entry

	return output

def make_date_list(start_date, end_date, step):
	# returns a list of datetimes going from start_date to end_date with an interval of step where step is in days (fractional for hours).	
	date_list = []
	for n in np.arange(0,(end_date - start_date).days + 1, step):
		date_list.append(start_date + dt.timedelta(int(n)))

	return date_list	

def add_year_both_series(river_list, years, wrf_directory):
	# get wrf data one year at a time
	for this_year in years:
		for this_month in range(1,13):
			print(str(this_year) + ' month '+ str(this_month))
			this_year_month_data = rv.read_WRF_year_month(this_year, this_month, wrf_directory)
	
			for this_river in river_list:
				this_river_year_series_rain = np.sum(np.sum(this_year_month_data['RAINNC']*this_river.wrf_catchment_factors, axis=2), axis=1)
				this_river.addToSeries('catchment_precipitation', this_river_year_series_rain, this_year_month_data['times'])
				this_river_year_series_rain = None
			
				this_river_year_series_temp = np.zeros(len(this_year_month_data['times']))		
	
				for i in range(0, len(this_year_month_data['times'])):
					this_river_year_series_temp[i] = np.average(this_year_month_data['T2'][i,:,:], weights=this_river.wrf_catchment_factors)

				this_river.addToSeries('catchment_temp', this_river_year_series_temp, this_year_month_data['times'])
			
				this_river_year_series_temp = None

def ll_dist(lon1, lat1, lon2, lat2):
	"""
	Calculate the great circle distance between two points 
	on the earth (specified in decimal degrees)

	Parameters
	----------
	lon1,lat1, lon2, lat2 - the lat and lon of the two points to calc the distance between

	Returns
	-------
	dist - distance between point one and two

	"""
	# convert decimal degrees to radians 
	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

	# haversine formula 
	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
	c = 2 * asin(sqrt(a)) 
	r = 6371 # Radius of earth in kilometers. Use 3956 for miles
	return c * r

def missing_val_interpolate(series_date_list, series, **kwargs):
	
	if 'complete_dates' in kwargs:
		complete_dates = kwargs['complete_dates']
	else:
		complete_dates = series_date_list
	
	series_complete = np.empty(len(complete_dates))
	series_complete[:] = np.nan

	# interpolate to nan valued points
	for this_ind, this_date in enumerate(complete_dates):
		if [x for x in series_date_list if x == this_date]:
			series_complete[this_ind] = [series[y] for y,x in enumerate(series_date_list) if x == this_date][0]

	nans = np.isnan(series_complete)
	x = lambda z: z.nonzero()[0]
	series_complete[nans] = np.interp(x(nans), x(~nans), series_complete[~nans])
	
	return series_complete

def make_generic_temp_model(river_dict, temp_model_file=None, temp_series_lags=None):
	if temp_model_file is None:
		temp_model_file = 'generic_temp_model.pk1'

	if temp_series_lags is None:
		temp_series_lags=[0,1,2]

	all_fit_data = np.empty([1,len(temp_series_lags) + 3])
	all_fit_data[:] = np.nan

	for this_river in river_dict.values():
		if hasattr(this_river, 'temp_model_fitdata'):
			print('Adding data to temp model from ' + this_river.river_name)
			obs_temps = np.asarray(copy.deepcopy(this_river.temp_gauge_data[1]), 'float')
			comp_dates = np.asarray(copy.deepcopy(this_river.temp_gauge_data[0]))
			obs_heights = np.asarray(copy.deepcopy(this_river.temp_gauge_data[4]))

			dependents = np.zeros([len(comp_dates), len(temp_series_lags) + 3])

			for this_ind, this_dt in enumerate(comp_dates):
				comp_dates[this_ind] = this_dt + dt.timedelta(hours=12)

			for this_col, this_lag in enumerate(temp_series_lags):
				dependents[:,this_col+1] = this_river.getWRFTempTimeLagSeries(this_lag, comp_dates)

			# Add in the heights of the relevant	
			dependents[:,-2] = obs_heights
			dependents[:,-1] = this_river.catchment_area
			dependents[:,0] = obs_temps
			dependents = dependents[~np.isnan(dependents).any(axis=1)]
	
			all_fit_data = np.append(all_fit_data, this_river.temp_model_fitdata , axis=0)
	
	all_fit_data = all_fit_data[1:,:]

	temp_model_general = linear_model.LinearRegression()
	temp_model_general.fit(all_fit_data[:,1:], all_fit_data[:,0])
	
	temp_model_dict = {'temp_model':temp_model_general, 'temp_model_lags':temp_series_lags, 'temp_model_fitdata':all_fit_data}
	
	with open(temp_model_file, 'wb') as output_file:
		pickle.dump(temp_model_dict, output_file, pickle.HIGHEST_PROTOCOL)
	output_file.close()
	
	return temp_model_dict

def apply_near_temp_model(river_dict, dist_threshold=200, model_check=[0]):
	river_ll = []
	river_keys = []
	
	for this_key, this_river in river_dict.items():
		river_keys.append(this_key)
		river_ll.append([this_river.mouth_lon, this_river.mouth_lat])
	river_keys = np.asarray(river_keys)
	river_ll = np.asarray(river_ll)

	river_dists = []
	for this_river in river_ll:
		river_dists.append(pf.grid.haversine_distance(this_river, river_ll.T))
	river_dists = np.asarray(river_dists)
	river_include = river_dists < dist_threshold

	# To strip out existing generic models if desired
	for this_river in river_dict.values():
		if np.array_equal(this_river.temp_model.coef_, model_check):
			del this_river.temp_model

	for this_key, this_river in river_dict.items():
		if not hasattr(this_river, 'temp_model'):
			print(this_river.river_name + ': building local temp model')
			
			near_river_list = river_keys[np.squeeze(river_include[river_keys == this_key,:])]
			close_river_dict = {}
			for this_key in near_river_list:
				close_river_dict[this_key] = river_dict[this_key]
			
			make_generic_temp_model(close_river_dict, temp_model_file='temp.pk1')
			this_river.retrieveGenericTempModel(temp_model_file='temp.pk1')
			gc.collect()

	return river_dict

def apply_catchment_size_temp_model(river_dict, catchment_thresholds, model_check=[0]):
	river_ca = []
	river_keys = []

	for this_key, this_river in river_dict.items():
		river_keys.append(this_key)
		river_ca.append(this_river.catchment_area)
		river_ll.append([this_river.mouth_lon, this_river.mouth_lat])
	river_keys = np.asarray(river_keys)
	river_ca = np.asarray(river_ca)
	river_ll = np.asarray(river_ll)

	# To strip out existing generic models if desired
	exclude_riv = []
	for this_key, this_river in river_dict.items():
		if np.array_equal(this_river.temp_model.coef_, model_check):
			del this_river.temp_model
			exclude_riv.append(this_key)

	# Make catchment based models
	river_catchment_bands = np.zeros(len(river_ca))
	for this_ind, (this_c_low, this_c_high) in enumerate(zip(catchment_thresholds[:-1], catchment_thresholds[1:])):
		river_catchment_bands[np.logical_and(river_ca >= this_c_low, river_ca < this_c_high)] = this_ind + 1
	river_catchment_bands[river_ca >= catchment_thresholds[-1]] = np.max(river_catchment_bands) + 1
	river_catchment_bands_all = copy.deepcopy(river_catchment_bands)
	
	useable_bands = np.unique(river_catchment_bands)
	filenames = ['temp_model_ca_{}.pk1'.format(this_band) for this_band in useable_bands]
	river_catchment_bands[np.isin(river_keys, exclude_riv)] = -999
	
	for this_band, this_file in zip(useable_bands, filenames):
		print('Building band {} temp model'.format(this_band))
		this_band_rivers = river_keys[river_catchment_bands == this_band]
		this_band_river_dict = {}
		for this_river in this_band_rivers:
			this_band_river_dict[this_river] = river_dict[this_river]
		make_generic_temp_model(this_band_river_dict, temp_model_file=this_file)

	# Apply to rivers without an existing model
	for this_key, this_river in river_dict.items():
		if not hasattr(this_river, 'temp_model'):
			this_river_band = int(river_catchment_bands_all[river_keys == this_key][0])
			print('River {} has no temp model adding band {} model'.format(this_key, this_river_band))
			this_model_file = filenames[this_river_band] 
			this_river.retrieveGenericTempModel(temp_model_file=this_model_file)

	return river_dict

def apply_nearest_temp_model(river_dict, distance_catchment_weight=[1,1], model_check=[0]):
	river_ca = []
	river_ll = []
	river_keys = []

	for this_key, this_river in river_dict.items():
		river_keys.append(this_key)
		river_ca.append(this_river.catchment_area)
	river_keys = np.asarray(river_keys)
	river_ca = np.asarray(river_ca)

	# To strip out existing generic models if desired
	exclude_riv = []
	for this_key, this_river in river_dict.items():
		if np.array_equal(this_river.temp_model.coef_, model_check):
			del this_river.temp_model
			exclude_riv.append(this_key)

	# Make catchment based models
	river_catchment_bands = np.zeros(len(river_ca))
	for this_ind, (this_c_low, this_c_high) in enumerate(zip(catchment_thresholds[:-1], catchment_thresholds[1:])):
		river_catchment_bands[np.logical_and(river_ca >= this_c_low, river_ca < this_c_high)] = this_ind + 1
	river_catchment_bands[river_ca >= catchment_thresholds[-1]] = np.max(river_catchment_bands) + 1
	river_catchment_bands_all = copy.deepcopy(river_catchment_bands)

	useable_bands = np.unique(river_catchment_bands)
	filenames = ['temp_model_ca_{}.pk1'.format(this_band) for this_band in useable_bands]
	river_catchment_bands[np.isin(river_keys, exclude_riv)] = -999

	for this_band, this_file in zip(useable_bands, filenames):
		print('Building band {} temp model'.format(this_band))
		this_band_rivers = river_keys[river_catchment_bands == this_band]
		this_band_river_dict = {}
		for this_river in this_band_rivers:
			this_band_river_dict[this_river] = river_dict[this_river]
		make_generic_temp_model(this_band_river_dict, temp_model_file=this_file)

	# Apply to rivers without an existing model
	for this_key, this_river in river_dict.items():
		if not hasattr(this_river, 'temp_model'):
			this_river_band = int(river_catchment_bands_all[river_keys == this_key][0])
			print('River {} has no temp model adding band {} model'.format(this_key, this_river_band))
			this_model_file = filenames[this_river_band]
			this_river.retrieveGenericTempModel(temp_model_file=this_model_file)

	return river_dict

def get_pyfvcom_prep(river_obj_list, start_date, end_date, ersem=False, ersem_vars=None, noisy=False):
	"""
	Takes a list of river like objects* and returns the data required by the PyFVCOM preproccesing module (pf.preproc.Model.add_Rivers)

	*e.g. River, RiverMulti, RiverLTLS
	

	Parameters
	----------
	river_obj_list : iterator of river like objects
		The rivers 
	start_date, end_date : datetimes
		The period of river data to output
	ersem : optional, boolean
		Whether to output the ersem_dict or not
	ersem_vars : optional, list of str
		List of ersem variables to include. If not specified the default list is 
			['N4_n', 'N3_n', 'O2_o', 'N1_p', 'N5_s', 'O3_c', 'O3_TA', 'O3_bioalk', 'Z4_c', 'Z5_n', 'Z5_p', 'Z5_c', 'Z6_n', 'Z6_p', 'Z6_c']

	Returns
	-------
	positions : array size no_rivers x 2
		River positions (lon, lat)
	names : array size no_rivers
		River names following ROSA convention of river_lon_lat
	times : array size no_rivers
		Array of python datetime objects for records
	flux_array : array size no_rivers x no_times
		Array of river flux values
	temperature : array size no_rivers x no_times
		Array of river temperature values 
	salinity :  
		Array of river salinity values

	"""
	positions = []
	names = []

	flux_array = []
	temperature_array = []
	salinity_array = []

	times = np.asarray([start_date + dt.timedelta(days=int(i)) for i in np.arange(0, (end_date - start_date).days + 1)])

	for this_river in river_obj_list:
		if noisy:
			print('Adding flux, temp, and salinity for: {}'.format(this_river.river_name))

		positions.append([this_river.mouth_lon, this_river.mouth_lat])
		names.append('river_{:4f}_{:4f}'.format(this_river.mouth_lon, this_river.mouth_lat))

		flux_array.append(this_river.getGaugeFluxSeries(start_date, end_date)[1])
		temperature_array.append(this_river.getTempModelSeries(start_date, end_date)[1])
		salinity_array.append(this_river.getSalinitySeries(start_date, end_date)[1])

	temperature = np.asarray([np.squeeze(this_temp) for this_temp in temperature_array]).T
	flux_array = np.asarray([np.squeeze(this_flux) for this_flux in flux_array]).T
	salinity_array = np.asarray([np.squeeze(this_sal) for this_sal in salinity_array]).T

	if ersem:
		ersem_dict = {}
		if ersem_vars is None:
			ersem_vars = ['N4_n', 'N3_n', 'O2_o', 'N1_p', 'N5_s', 'O3_c', 'O3_TA', 'O3_bioalk',
											'Z4_c', 'Z5_n', 'Z5_p', 'Z5_c', 'Z6_n', 'Z6_p', 'Z6_c']
		for this_var in ersem_vars:
			if noisy:
				print('Adding ersem variable {}'.format(this_var))
			this_nutrient = []
			for this_river in river_obj_list:
				this_nutrient.append(this_river.getNutrientSeries(this_var, start_date, end_date)[1])
			ersem_dict[this_var] = np.asarray(this_nutrient).T

	else:
		ersem_dict = False

	return np.asarray(positions), np.asarray(names), np.asarray(times), flux_array, temperature, salinity_array, ersem_dict

