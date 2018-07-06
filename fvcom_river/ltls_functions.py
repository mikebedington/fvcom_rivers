import numpy as np
import datetime as dt
import pickle as pk
import netCDF4 as nc
import sklearn.linear_model as skl
import copy

import PyFVCOM as pf


###### Object to contain LTLS model output data ######
class _passive_data_store():
	def __init__(self):
		pass

class ltls_data_store():
	def __init__(self, ltls_indices):
		self.ltls_indices = ltls_indices
		self.ltls_vars_op = {'cal':'sum', 'ngas':'sum', 'dic':'sum',  'fsed':'sum', 'don':'sum', 'DOC':'sum', 'so4':'sum',
				'tdp':'sum', 'amm':'sum', 'nit':'sum', 'flow':'sum', 'ph':'average', 'O2':'average', 'Chl':'average', 'temp':'average'}
		self.times_monthly = _passive_data_store()
		self.month_data = _passive_data_store()

	def retrieve_daily_flow(self, flow_nc, ref_date=dt.datetime(1858,11,17,0,0)):
		time_int = flow_nc.variables['time'][:]
		self.time_dt = np.asarray([ref_date + dt.timedelta(days=int(this_time)) for this_time in time_int])
		flux_raw = []
		for this_index in self.ltls_indices:
			flux_raw.append(flow_nc.variables['river_flux'][:, this_index[0], this_index[1]])
		self.river_flux = np.sum(np.asarray(flux_raw), axis=0)

	def make_nutrient_conc(self, atom_weights=None, ox_weight=None):
		if atom_weights is None:
			atom_weights = {'dic':12 , 'tdp':30.9, 'amm':14, 'nit':14}

		for this_nutrient_var in atom_weights.keys():
			this_nutrient_raw = getattr(self, this_nutrient_var)
			this_grams_per_m_3 = ((this_nutrient_raw/self.flow) * (10**6)) / (60*60*24)
			this_mmol_per_gram = (this_grams_per_m_3 * 1000)/atom_weights[this_nutrient_var]
			setattr(self, this_nutrient_var + '_conc', np.asarray(this_mmol_per_gram))

		if ox_weight is None:
			ox_weight = 32
		
		setattr(self, 'O2_conc', getattr(self, 'O2')*ox_weight)

	def retrieve_monthly_nutrient(self, nutrient_nc):
		self.times_monthly.month = nutrient_nc.variables['month'][:]
		self.times_monthly.year = nutrient_nc.variables['year'][:]
		self.times_monthly.time_dt = np.asarray([dt.datetime(this_year, this_month, 15,0,0) for this_year, this_month in zip(self.times_monthly.year, self.times_monthly.month)])
		this_flux_weights = []
		for this_index in self.ltls_indices:
			this_flux_weights.append(nutrient_nc.variables['flow'][:,this_index[0], this_index[1]])
		this_flux_weights = np.sum(np.asarray(this_flux_weights), axis=1)

		for this_var in self.ltls_vars_op.keys():
			this_var_raw = []
			for this_index in self.ltls_indices:
				this_var_raw.append(nutrient_nc.variables[this_var][:,this_index[0],this_index[1]])

			if self.ltls_vars_op[this_var] == 'sum':
				this_var_proc = np.sum(np.asarray(this_var_raw), axis=0)
			else:
				this_var_proc = np.average(np.asarray(this_var_raw), axis=0, weights=this_flux_weights)

			setattr(self.month_data, this_var, this_var_proc)

	def make_nutrient_monthly_conc(self, atom_weights=None, ox_weight=None):
		if not hasattr(self.times_monthly, 'month'):
			print('No nutrient data retrieved yet')
			return

		days_in_months = np.asarray([31,28,31,30,31,30,31,31,30,31,30,31])
		if atom_weights is None:
			atom_weights = {'dic':12 , 'tdp':30.9, 'amm':14, 'nit':14}

		for this_nutrient_var in atom_weights.keys():
			this_nutrient_raw = getattr(self.month_data, this_nutrient_var)
			this_nutrient_concs = []
			for this_ind, this_month in enumerate(self.times_monthly.month):
				this_grams_per_m_3 = ((this_nutrient_raw[this_ind]/self.month_data.flow[this_ind]) * (10**6)) / (60*60*24*days_in_months[int(this_month) -1])
				this_mmol_per_gram = (this_grams_per_m_3 * 1000)/atom_weights[this_nutrient_var]
				this_nutrient_concs.append(this_mmol_per_gram)
			setattr(self.month_data, this_nutrient_var + '_conc', np.asarray(this_nutrient_concs))

		if ox_weight is None:
			ox_weight = 32

		setattr(self.month_data, 'O2_conc', getattr(self.month_data, 'O2')*ox_weight)

	def get_daily_flow(self, start_date, end_date):
		time_sel = np.logical_and(self.time_dt >= start_date, self.time_dt <= end_date)
		return self.time_dt[time_sel], self.river_flux[time_sel]

	def get_nutrient_daily(self, start_date, end_date, nutrient):
		choose_dates = np.logical_and(self.time_dt >= start_date, self.time_dt <= end_date)	
		return self.time_dt[choose_dates], getattr(self, nutrient)[choose_dates]

	def get_nutrient_daily_interp(self, start_date, end_date, nutrient):
		ref_date = dt.datetime(1900,1,1)
		time_month_sel = np.logical_and(self.times_monthly.time_dt >= start_date, self.times_monthly.time_dt <= end_date)
		time_month_ms = np.asarray([(this_time - ref_date).total_seconds() for this_time in self.times_monthly.time_dt[time_month_sel]])
		nutrient_raw = getattr(self.month_data, nutrient)

		daily_series_dt = [start_date]
		daily_series_ms = [(start_date - ref_date).total_seconds()]
		while daily_series_dt[-1] < end_date:
			daily_series_dt.append(daily_series_dt[-1] + dt.timedelta(days=1))
			daily_series_ms.append((daily_series_dt[-1] - ref_date).total_seconds())
		daily_series_dt = np.asarray(daily_series_dt)
		daily_series_ms = np.asarray(daily_series_ms)

		interped_nutrient = np.interp(daily_series_ms, time_month_ms, nutrient_raw[time_month_sel])
		return daily_series_dt, interped_nutrient

	@staticmethod
	def ltls_tonne_to_conc(ltls_data, var_name, ltls_flow):
		tot_flow = np.sum(ltls_flow, axis=1)
		tot_input = np.sum(ltls_data, axis=1)

		this_grams_per_m_3 = ((tot_input/tot_flow) * (10**6)) / (60*60*24)
		this_mmol_per_gram = (this_grams_per_m_3 * 1000)/atom_weights[var_name]

		return this_mmol_per_gram

###### River-like class to 
class ltls_comp_river():
	def __init__(self, river_obj, ltls_data_store, nemo_river = None):
		self.river_obj = river_obj
		self.ltls_data_store = ltls_data_store

		self.mouth_lon = self.river_obj.mouth_lon
		self.mouth_lat = self.river_obj.mouth_lat
		self.river_name = self.river_obj.river_name
	
		if nemo_river is not None:
			self.nemo_river_obj = nemo_river

	def train_flux_scaling_model(self, train_start, train_end):
		"""
		Regress the ltls daily flow against the gauge flow to model, this allows a prediction of

		"""
		gauge_data = np.asarray(self.river_obj.getGaugeFluxSeries(train_start, train_end)[1])
		ltls_data = np.asarray(self.ltls_data_store.get_daily_flow(train_start, train_end)[1])

		lf = skl.LinearRegression()
		lf.fit(gauge_data.reshape(-1, 1), ltls_data)

		self.ltls_flux_mod = [lf.intercept_, lf.coef_, lf.score(gauge_data.reshape(-1, 1), ltls_data)]
	
	def plotRescaleVsLTLS(self, start_date, end_date):
		"""
		Plot the rescaled river flux against the ltls data to assess the model produced by self.rescale_flux

		"""
		days_in_months = np.asarray([31,28,31,30,31,30,31,31,30,31,30,31])
		date_list, flux_pred_list = self.getGaugeFluxSeries(start_date, end_date)
		year_list = np.asarray([this_date.year for this_date in date_list])
		month_list = np.asarray([this_date.month for this_date in date_list])

		ltls_plot = []
		pred_plot = []

		for (this_month, this_year), this_ltls_flux in zip(
						zip(self.ltls_data_store.times_monthly.month, self.ltls_data_store.times_monthly.year), self.ltls_data_store.month_data.flow):
			choose_data = np.logical_and(month_list == this_month, year_list == this_year)
			this_month_pred_flux = np.sum(np.asarray(flux_pred_list)[choose_data])
			ltls_plot.append(this_ltls_flux * days_in_months[int(this_month) -1])
			pred_plot.append(this_month_pred_flux)

		plt.figure()
		plt.plot(ltls_plot, c='red')
		plt.plot(pred_plot, c='blue')
		plt.legend(['LTLS', 'Gauge prediction'])
		plt.savefig(self.river_obj.river_name + '_month_flux_comp.png', dpi=300)
		plt.close()

	def getGaugeFluxSeries(self, start_date, end_date):
		"""
		Get the predicted flux (underlying river object gauge flux (observed or modelled) then with ltls rescaling applied
		"""
		date_list, raw_flux = self.river_obj.getGaugeFluxSeries(start_date, end_date)		
		flux_list = self.ltls_flux_mod[0] + self.ltls_flux_mod[1]*raw_flux
		
		return date_list, flux_list

	def getTempModelSeries(self, start_date, end_date):
		"""
		Get the series of predicted river temperatures, this either from interpolating monthly ltls data (if self.use_ltls_temp is
		True, this is not passed in this function to allow it to mimic a river object) or from the temperature regression model of the 
		river object

		"""

		if hasattr(self,'use_ltls_temp') and self.use_ltls_temp:
			date_list, temp_list = self.getNutrientSeries('temp', start_date, end_date)
		else:	
			date_list, temp_list = self.river_obj.getTempModelSeries(start_date,end_date)

		return date_list, temp_list

	def getNutrientSeries(self, nutrient_name, start_date, end_date):
		"""
		Get a nutrient series. In order of priority if available it returns LTLS daily data, then monthly LTLS data interpolated to daily, then
		the associated nemo object, or if none of the above then zeros are returned.

		Returns
		-------
		date_list : list of datetime objects
			Dates of daily nutrient predictions
		nutrient_list : list
			Nutrient values, zeros if neither ltls or nemo data are available

		"""
		ersem_to_ltls_var = {'N4_n':'amm_conc', 'N3_n':'nit_conc', 'O2_o':'O2_conc', 'N1_p':'tdp_conc', 'temp':'temp'}
	
		if nutrient_name in ersem_to_ltls_var.keys():
			this_ltls = ersem_to_ltls_var[nutrient_name]
			if hasattr(self.ltls_data_store, this_ltls):
				date_list, nutrient_list = self.ltls_data_store.get_nutrient_daily(start_date, end_date, this_ltls)
			else:
				date_list, nutrient_list = self.ltls_data_store.get_nutrient_daily_interp(start_date, end_date, this_ltls)
		elif hasattr(self, 'nemo_river_obj') and hasattr(self.nemo_river_obj, nutrient_name):
			date_list, nutrient_list = self.nemo_river_obj.getNutrientSeries(nutrient_name, start_date, end_date)
		else:
			date_list = [start_date + dt.timedelta(days=int(i)) for i in np.arange(0, (end_date - start_date).days +1)] 
			nutrient_list = np.zeros(len(date_list))

		return date_list, nutrient_list

	def getSalinitySeries(self, start_date, end_date):
		date_list, salinity_list = self.river_obj.getSalinitySeries(start_date, end_date)
		return date_list, salinity_list

class nemoRiver():
	def __init__(self, river_index_no, river_nc_path, river_mouth_ll, vars_list=None, expand_dates=None):
		nemo_nc = nc.Dataset(river_nc_path, 'r')
		times_str = [b''.join(this_row) for this_row in nemo_nc.variables['Times'][:]]
		self.dates = [dt.datetime.strptime(this_row.decode('utf-8').rstrip(), '%Y/%m/%d %H:%M:%S') for this_row in times_str]
		self.river_index_no = river_index_no
		self.mouth_lon = river_mouth_ll[0]
		self.mouth_lat = river_mouth_ll[1]	
		self.river_name = 'river_{}_{}'.format(self.mouth_lon, self.mouth_lat)

		if vars_list is None:
			vars_list = ['river_flux', 'river_temp', 'river_salt', 'N1_p', 'N3_n', 'N4_n', 'N5_s', 'O2_o', 'O3_TA', 'O3_c', 'O3_bioalk',
							'Z4_c', 'Z5_n', 'Z5_p', 'Z5_c', 'Z6_n', 'Z6_p', 'Z6_c']
		self._add_all_vars(vars_list, nemo_nc)

		if expand_dates is not None:
			self._expandDateSeries(expand_dates[0], expand_dates[1])

	def _add_all_vars(self, vars_list, nemo_nc):
		for this_var in vars_list:
			try:
				setattr(self, this_var, nemo_nc.variables[this_var][:, self.river_index_no])
			except KeyError:
				print('No {} in nemo river'.format(this_var))
				vars_list.remove(this_var)
		self.vars_list = vars_list
	
	def _expandDateSeries(self, start_date, end_date):
		dates_yday = np.asarray([this_date.timetuple().tm_yday for this_date in self.dates])
		yday_unique, yday_map = np.unique(dates_yday, return_index=True)

		all_dates = [start_date + dt.timedelta(days = int(i)) for i in np.arange(0, (end_date - start_date).days + 1)]
		all_dates_yday = np.asarray([this_date.timetuple().tm_yday for this_date in all_dates])

		if 366 not in yday_unique:
			all_dates_yday[all_dates_yday == 366] = 365 # deal with leap years if there aren't any in the original data

		new_yday_unique, new_yday_map = np.unique(all_dates_yday, return_inverse=True)
			
		old_to_new_unique = []
		for this_day in new_yday_unique:
			old_to_new_unique.append(yday_map[yday_unique == this_day][0])

		for this_var in self.vars_list:
			var_raw = getattr(self, this_var)[yday_map]
			var_raw = var_raw[old_to_new_unique]
			var_expand = var_raw[new_yday_map]
			setattr(self, this_var, var_expand)

		self.dates = all_dates

	def getGaugeFluxSeries(self, start_date, end_date):
		date_list, flux_list = self.getNutrientSeries('river_flux', start_date, end_date)
		return date_list, flux_list

	def getTempModelSeries(self, start_date, end_date):
		date_list, temp_list = self.getNutrientSeries('river_temp', start_date, end_date)	
		return date_list, temp_list

	def getSalinitySeries(self, start_date, end_date):
		date_list, sal_list = self.getNutrientSeries('river_salt', start_date, end_date)   
		return date_list, sal_list

	def getNutrientSeries(self, nutrient_name, start_date, end_date):
		date_list = [start_date + dt.timedelta(days = int(i)) for i in np.arange(0, (end_date - start_date).days + 1)]
		dates_choose = np.isin(self.dates, date_list)
		nutrient_list = [this_entry for this_entry in getattr(self, nutrient_name)[dates_choose]]	
		return date_list, nutrient_list

class nemoMultiRiver(nemoRiver):
	def _add_all_vars(self, vars_list, nemo_nc):
		for this_var in vars_list:
			try:
				if this_var == 'river_flux':
					setattr(self, this_var, np.sum(nemo_nc.variables[this_var][:, self.river_index_no], axis=1))
				else:
					setattr(self, this_var, np.mean(nemo_nc.variables[this_var][:, self.river_index_no], axis=1))
			except KeyError:
				print('No {} in nemo river'.format(this_var))
				vars_list.remove(this_var)
		self.vars_list = vars_list
