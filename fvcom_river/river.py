# generally available modules
import numpy as np
import sys
import subprocess
import os
import shapefile
from shapely import geometry as gm
from shapely.ops import cascaded_union
import datetime as dt
import csv
import copy
import math
import sqlite3 as sq
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import pickle
import glob as gb
import netCDF4 as nc

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


# these are other modules I've written or borrowed
import fvcom_river.read_wrf_variable as rv
import fvcom_river.os_conversion as os_conversion
import fvcom_river.river_functions as rfun
import fvcom_river.nn_functions as nn_fun  # functions relating to neural network

"""
A class for river objects to provide forcing for FVCOM model. The rivers are defined by their CEH gauge number (if gauged), their name, lat/lon of mouth, and a .shp of their catchment area. The aim of the class is to collate and model flux data (river flow) and temperature data. CEH gauge flux and EA water temperature database respectively are used as observations for model training and output from a WRF model (variables 'T2' and 'RAINNC') as dependent variables. Multiple linear regression is used to model the river temperature and a neural network is used to model the river flux.

A child class RiverMulti provides functionality for when multiple CEH gauges contribute to a single river output.

An example of use is shown in the seperate script river_setup_example.py.


Naming conventions
------------------

Attributes are lowercase seperated by "_"
Methods follow java(ish) naming conventions

Methods starting 'retrieve' are getting data from outside sources
Methods starting 'get' are for retrieving data from the River object
Methods starting 'make' use the object data to create some model/other useful thing to add to the object



Data locations
--------------
CEH gauge data is in two seperate directories 'flux_data' which will contain the river fluxs (automoatically retrieved by the river object where they exist) and 'catchments' which will have seperate subdirectories for each river named after their gauge_id_no and containg a catchment polygon in OS coordinates gauge_id_no.shp.


EA temperature data. It is expected the data is in a database temperature.db with the observation data in a table full_data where the schema and table names are from the original msaccess EA database.

WRF data. The WRF data is expected in monthly nc files with the two variables of interest in. A shell script for generating these from the standard output is included as make_WRF_month_nc.sh.


"""


class River:
    def __init__(self, river_name, id_no, mouth_lon, mouth_lat, temp_model_obs_no_threshold = 50, nn_model_obs_no_threshold = 50, temp_model_obs_dist_threshold = 100):
        # Initialise the river with its name and associated CEH gauge number, ideally this should be the most downstream gauge available
        # The catchment_precipitation and catchment_temp are the lists of the summed precipitation in the catchment from the WRF model and mean 2m surface temperature from the WRF model respectively, catchment_list[0] will be the dates and catchment_list[1] the values
        self.river_name = river_name
        self.ceh_gauge_id_no = id_no
        self.ceh_gauged_river_names = []
        self.mouth_lon = mouth_lon
        self.mouth_lat = mouth_lat

        # thresholds for rejecting model building (insufficient data)
        self.nn_model_obs_threshold = nn_model_obs_no_threshold

        self.temp_obs_dist_threshold = temp_model_obs_dist_threshold # in km, the max distance away from the river mouth temperature observations can be, to exclude similiar named rivers

        self.catchment_precipitation = [[],[]]
        self.catchment_temp = [[],[]]

    def retrieveFlux(self, *args, ceh_data_path=None):
        # A method to get the daily river flux data from the associated CEH gauge. It uses the shell script get_data_mb.sh to retrieve the data from the internet and save it in my data directory.
        if args:
            gauge_id_no = args[0]
        else:
            gauge_id_no = self.ceh_gauge_id_no

        if not self.ceh_gauge_id_no:
            print(self.river_name + ": Can't retrieve flux, no associated gauge id(s)")
            return

        if ceh_data_path==None:
            if not self.ceh_data_path:
                print(self.river_name + ": Can't retrieve flux no data path specified")
                return
            else:
                ceh_data_path = self.ceh_data_path

        flux_data, gauge_lat, gauge_lon = self._fluxNlocParsing(gauge_id_no, ceh_data_path)

        self.gauge_lat = gauge_lat
        self.gauge_lon = gauge_lon

        if not np.any(flux_data[0]):
            print(self.river_name + ': Warning no gauge flux data retrieved')

        self.flux = flux_data

        return [flux_data, [gauge_lon, gauge_lat]]

    def _fluxNlocParsing(self, gauge_id_no, ceh_data_path):
        home_dir = os.getcwd() # Not sure why I can't get the addpath to work so use this clunkier method instead
        os.chdir(ceh_data_path + '/flux_data/')

        # shell script picks up it's target from this file
        f = open("stations.csv", "w")
        f.write(str(gauge_id_no)+", "+self.river_name+"\n")
        f.close()

        # run shell script to retrieve flux data
        subprocess.call(['./get_flux_data.sh'])

        # read in output
        flux_data = rfun.read_csv_unheaded(str(gauge_id_no)+'_data.csv', 2)

        gauge_loc_file = (str(gauge_id_no)+'_loc.txt')

        with open(gauge_loc_file, 'rt') as open_file:
            gauge_loc_os_text=open_file.read().replace('\n', '')
        open_file.close()

        # And whisk us back to our working directory where the os_conversion stuff is
        os.chdir(home_dir)

        gauge_loc_os_text = gauge_loc_os_text.split(',')[1]

        gauge_loc_os = os_conversion.OS_text_convert(gauge_loc_os_text)
        [gauge_lat, gauge_lon] = os_conversion.OStoLL(gauge_loc_os[0], gauge_loc_os[1])

        # convert the dates to python datetimes to be more useful
        date_list = []
        flux_list = []
        for i in range(0, len(flux_data[1])):
            if flux_data[1][i]:
                flux_list.append(float(flux_data[1][i]))
                date_list.append(dt.datetime.strptime(flux_data[0][i], '%Y-%m-%d'))

        flux_data = [date_list, flux_list]

        return flux_data, gauge_lat, gauge_lon

    def getGaugeFluxSeries(self, start_date, end_date):
        if not isinstance(start_date, dt.datetime) and isinstance(end_date, dt.datetime):
            print('Requires datetime object inputs for start and end date')
            return

        if not self.ceh_gauge_id_no:
            print(self.river_name + ': No associated gauge id')

            if hasattr(self, 'nn_model_file'):
                print(self.river_name + ': Returning neural net series instead')
                modelled_data = self.getNeuralNetFluxSeries(start_date, end_date)
                return modelled_data

            else:
                return

        if not hasattr(self, 'flux'):
            print('No flux data -- going to get it')
            self.retrieveFlux()

        date_list = rfun.make_date_list(start_date, end_date,1)
        flux_dates = list(set(date_list).intersection(self.flux[0]))
        flux_dates.sort()

        flux_series = [[self.flux[1][x] for x,y in enumerate(self.flux[0]) if y == this_date][0] for this_date in flux_dates]

        if len(flux_series) < 2:
            if hasattr(self, 'nn_model_file'):
                print(self.river_name + ': No available gauge flux data, returning nueral net modelled version')
                modelled_data = self.getNeuralNetFluxSeries(start_date, end_date)
                return modelled_data

            else:
                print(self.river_name + ': No available gauge flux data')
                return

        if  len(date_list) - len(flux_series) > 0:
            if hasattr(self, 'nn_model_file'):
                flux_series_complete = self.getNeuralNetFluxSeries(start_date, end_date)

                for this_ind, this_date in enumerate(flux_series_complete[0]):
                    if [flux_series[x] for x,y in enumerate(flux_dates) if y == this_date]:
                        flux_series_complete[1][this_ind] = [flux_series[x] for x,y in enumerate(flux_dates) if y == this_date][0]
                flux_series_complete = flux_series_complete[1]
                print(self.river_name + str(len(flux_series_complete) - len(flux_series))+' missing flux values -- neural net used')

            else:
                flux_series_complete = rfun.missing_val_interpolate(flux_dates, flux_series, complete_dates=date_list)
                print(self.river_name + str(len(flux_series_complete) - len(flux_series))+' missing flux values -- interpolation used')

        else:
            flux_series_complete = flux_series

        return [date_list, flux_series_complete]

    def retrieveGaugeCatchment(self,  *args, ceh_data_path=None):
        # A method to get the shape polygon of the catchment for the associated CEH gauge.
        if args:
            gauge_id_no = str(args[0])
        else:
            if not self.ceh_gauge_id_no:
                print(self.river_name + ': No gauge id specified, expecting catchment file named ' + self.river_name + '.shp')
                gauge_id_no = self.river_name
            else:
                gauge_id_no = str(self.ceh_gauge_id_no)

        if ceh_data_path==None:
            if not self.ceh_data_path:
                print(self.river_name + ": Can't retrieve flux no data path specified")
                return
            ceh_data_path = self.ceh_data_path

        home_dir = os.getcwd()
        os.chdir(ceh_data_path + '/catchments/' + gauge_id_no+'/') # This assumes catchment already retrieved I can't get the curl to work for automatic download, need to ask Pierre sometime
        this_catchment_shp = shapefile.Reader(gauge_id_no+".shp")

        # We want to turn the .shp file into a list of coordinates for the corners of the polygon, first we retrieve the list of coordinates
        if len(this_catchment_shp.shapeRecords()) == 1:
             points_list_os = this_catchment_shp.shapeRecords()[0].shape.points
        else:
            print('Error - More than one catchment area defined')
            return

        # The points are in british national grid (ordanance survey) references so convert them to lat lon
        points_list_convert = []
        for this_point_os in points_list_os:
            points_list_convert.append(os_conversion.OStoLL(this_point_os[0], this_point_os[1]))

        # Add the polygon to the river object
        catchment_poly_points = points_list_convert

        # and a couple of other useful bits of info about the catchment: the bounding box and the total area of the catchment
        bbox = this_catchment_shp.bbox
        catchment_bbox = np.asarray([os_conversion.OStoLL(bbox[0], bbox[1]), os_conversion.OStoLL(bbox[2], bbox[3])])

        catchment_area = (gm.Polygon(catchment_poly_points)).area

        os.chdir(home_dir)

        self.catchment_poly_points = catchment_poly_points
        self.catchment_bbox = catchment_bbox
        self.catchment_area = catchment_area

        return [catchment_poly_points, catchment_bbox, catchment_area]

    def makeWRFCatchmentFactors(self, *args, wrf_data_path=None):
        # A method to make weighting factors for the WRF grid points with respect to how much of the river catchment falls within their demesne. The weighting is (area of WRF 'square' intersecting with catchment)/(area of WRF 'square')

        # Make a list of the wrf 'squares' surrounding each calculation point. Possibly should be replaced by squares based on the XLON_V, XLON_U,..etc as currently calculated manually using the offsets between the XLON, and XLAT grid which is probably not strictly correct (though should be pretty close)
        if args:
            c_bbox = args[0]
            c_poly_points = args[1]
        else:
            c_bbox = self.catchment_bbox
            c_poly_points = self.catchment_poly_points

        if wrf_data_path==None:
            if not self.wrf_data_path:
                print(self.river_name + ": Can't retrieve catchment no data path specified")
                return
            wrf_data_path = self.wrf_data_path

        wrf_squares = rfun.make_WRF_squares_list(wrf_data_path)

        wrf_catchment_factors = np.zeros(np.squeeze(wrf_squares[0,0,:,:].shape))

        # First to reduce computation grab only the WRF 'squares' where at least one corner is within the bounding box
        corners_in_catchment = (wrf_squares[:,0,:,:] > c_bbox[0,0]) & (wrf_squares[:,0,:,:] < c_bbox[1,0]) & (wrf_squares[:,1,:,:] > c_bbox[0,1]) & (wrf_squares[:,1,:,:] < c_bbox[1,1])

        partial_inside = np.any(corners_in_catchment, axis=0)

        # Turn the catchment into a geometry poly so we can do intersections
        catchment_poly = gm.Polygon(c_poly_points)

        # Loop over the WRF 'squares' which have one corner falling in the bounding box and add their factor to the matrix
        for these_indices in np.asarray(np.where(partial_inside)).T:
            # Get the four corners
            this_square_poly = gm.Polygon(wrf_squares[:,:,these_indices[0], these_indices[1]])

            # proportion in catchment area
            wrf_catchment_factors[these_indices[0], these_indices[1]] = this_square_poly.intersection(catchment_poly).area / this_square_poly.area

        # We missed out the first and last row and column of the WRF lat lon when defining the squares so reinsert them
        wrf_catchment_factors_adj = np.zeros([wrf_catchment_factors.shape[0]+2, wrf_catchment_factors.shape[1]+2])
        wrf_catchment_factors_adj[1:-1, 1:-1] = wrf_catchment_factors

        self.wrf_catchment_factors = wrf_catchment_factors_adj
        # Add to the river object
        return wrf_catchment_factors_adj

    def addToSeries(self, att_series, add_series, dates, override=False):
        # A general method for adding data to one of the objects series (i.e. cathment_precipitation, catchment_temp)
        this_series = getattr(self, att_series, [])

        # We're doing list append so don't want to double up dates, this should probably be replaced by functionality which replaces already existing dates
        not_existing_dates = [x for x,y in enumerate(dates) if y not in this_series[0]]
        add_series_new = np.asarray(add_series)[not_existing_dates]
        dates_new = np.asarray(dates)[not_existing_dates]

        # Add the new data
        for i in range(0, len(dates_new)):
            this_series[0].append(dates_new[i])
            this_series[1].append(add_series_new[i])

        if override and not len(not_existing_dates) == len(dates):
            existing_bool = np.ones(len(dates), dtype=bool)
            existing_bool[not_existing_dates] = False

            override_series = np.asarray(add_series)[existing_bool]
            override_dates = np.asarray(dates)[existing_bool]

            for i, this_date in enumerate(override_dates):
                this_series_ind = this_series[0].index(this_date)
                this_series[1][this_series_ind] = override_series[i]

        # resort by date
        sort_ind = np.asarray(this_series[0]).argsort()
        this_series = [list(np.asarray(this_series[0])[sort_ind]), list(np.asarray(this_series[1])[sort_ind])]

        # add back to self
        setattr(self, att_series, this_series)

    def retrieveWRFSeries(self,years, wrf_data_path=None):
        # Get mean temperature from the WRF model across the catchment area for the years in the list 'years'
        print('If retrieving data for multiple gauges it is more efficient to call the functions in gauge_functions.py outside of the class method')

        if wrf_data_path==None:
            if not self.wrf_data_path:
                print(self.river_name + ": Can't retrieve catchment no data path specified")
                return
            wrf_data_path = self.wrf_data_path

        rfun.add_year_both_series([self],years, wrf_data_path)

    def getWRFDailyPrecipitationSeries(self, start_date, end_date, *offset):
        # The WRF precipitation is at 3hr timesteps but to model flux we want a daily series for comparison. This returns the series between start_date and end_date with the series offset by a lag 'offset' if required.

        # If no offset given don't lag
        if not offset:
            offset = [0]

        date_list = rfun.make_date_list(start_date, end_date,1)

        # Loop over the daily list and sum the precipitation
        selected_series = [[],[]]

        for this_date in date_list:
            selected_series[0].append(this_date)
            selected_series[1].append(np.nansum([self.catchment_precipitation[1][x] for x,y  in enumerate(self.catchment_precipitation[0]) if (y - dt.timedelta(offset[0])).date() == this_date.date()]))

        selected_series_complete = rfun.missing_val_interpolate(selected_series[0], selected_series[1])

        return [selected_series[0], selected_series_complete]

    def getWRFDailyTempSeries(self, start_date, end_date, *offset):
        if not offset:
            offset = [0]

        date_list = rfun.make_date_list(start_date, end_date,1)

        selected_series = [[],[]]

        for this_date in date_list:
            selected_series[0].append(this_date)
            selected_series[1].append(np.mean([self.catchment_temp[1][x] for x,y  in enumerate(self.catchment_temp[0]) if (y - dt.timedelta(offset[0])).date() == this_date.date()]))

        selected_series_complete = rfun.missing_val_interpolate(selected_series[0], selected_series[1])

        return [selected_series[0], selected_series_complete]

    def getSeriesDates(self, start_date, end_date, att_series):
        this_series = getattr(self, att_series, [])

        date_list = rfun.make_date_list(start_date, end_date,1)

        selected_series = [[],[]]

        for this_date in date_list:
            selected_series[0].append(this_date)
            if [this_series[1][x] for x,y  in enumerate(this_series[0]) if y.date() == this_date.date()]:
                selected_series[1].append([this_series[1][x] for x,y  in enumerate(this_series[0]) if y.date() == this_date.date()][0])
            else:
                selected_series[1].append(np.nan)

        return selected_series

    def makeSeasonalFlux(self):
        # models a seasonal flux curve

        # if the flux isn't already here give up
        if not self.flux:
            print(self.river_name + ': No flux series retrieved')
            return

        # array of day numbers for a year, goes to 366 to cope with leapy ears
        yds = range(1,367)

        # get the mean over the whole flux series by year day, on two lines for readibility
        series_yday = [x.timetuple().tm_yday for x in self.flux[0]]
        daily_mean = [np.mean([x for y,x in enumerate(self.flux[1]) if series_yday[y] == this_yd]) for this_yd in yds]

        # fit a polynomial to the data
        poly_order = 5
        fit_poly = np.polyfit(yds, daily_mean, poly_order)

        # add to the object as both the polynomial and as the deas seasoned flux data
        self.seasonal_flux_poly = np.poly1d(fit_poly)

        self.flux_deseason = [self.flux[0], self.flux[1] - self.seasonal_flux_poly(series_yday)]

        return [np.poly1d(fit_poly), [self.flux[0], self.flux[1] - self.seasonal_flux_poly(series_yday)]]

    ### Temperature regression model methods
    def retrieveTempObsData(self, EA_data_path=None):
        # A method to get the observation data from the environment agency water temperature archive. Appropriate stations are identified by a) matching the river name b) checking that the stations lie within the set threshold (self.temp_obs_dist_threshold) of the river mouth. This is possibly not entirely foolproof but hopefully similiarly named rivers close together behave in the same way!

        if EA_data_path==None:
            if not self.EA_data_path:
                print(self.river_name + ": Can't retrieve temp observations no data path specified")
                return
            EA_data_path = self.EA_data_path

        home_dir = os.getcwd()
        os.chdir(EA_data_path)

        # set up the connection to the database of temperature measurements. The database is the Environment agency water temperature archive for the south-west
        conn = sq.connect('temperature_db.db')
        c = conn.cursor()
        db_data = [[],[],[],[],[],[]]
        # query for all the temperature records from stations with the relevant river name in the title (includes some upstream but not often).
        for query_river_name in self.ceh_gauged_river_names:

            query_str = 'select a.sampleDate,a.detResult,a.siteID_text, b.siteX,b.siteY,b.siteZ from full_data as a inner join (select siteID, siteX, siteY, siteZ from siteInfo where siteName like "%R ' + query_river_name + ' %") as b on a.siteID = b.siteID where date(sampleDate) > date("2000-01-01 00:00:00") order by sampleDate;'

            c.execute(query_str)
            all_rows = c.fetchall()

            for i in range(0, len(all_rows)):
                db_data[0].append(dt.datetime.strptime(all_rows[i][0], '%Y-%m-%d %H:%M:%S'))
                db_data[1].append(all_rows[i][1])
                db_data[4].append(all_rows[i][5])
                [lat, lon] = os_conversion.OStoLL(int(all_rows[i][3]), int(all_rows[i][4]))
                db_data[2].append(lon)
                db_data[3].append(lat)

        os.chdir(home_dir)

        # due to the limited number of river names remove any records from further than a threshold distance away as being from another river, this is a far from perfect condition but better than nowt
        dist_thresh = self.temp_obs_dist_threshold # in km

        site_dists = np.zeros(len(db_data[0]))
        for i in range(0, len(site_dists)):
            site_dists[i] = rfun.ll_dist(float(self.mouth_lon), float(self.mouth_lat), db_data[2][i], db_data[3][i])

        db_data[5] = site_dists
        self.temp_gauge_data_raw = copy.deepcopy(db_data)

        allowed_sites = site_dists <= dist_thresh

        for this_ind,this_col in enumerate(db_data):
            db_data[this_ind] = np.asarray(this_col)[allowed_sites]

        self.temp_gauge_data = db_data

        return db_data

    def getWRFTempTimeLagSeries(self, lag_days, chosen_dates, interp=True):
        chosen_dates = np.asarray([this_date - dt.timedelta(days=lag_days) for this_date in chosen_dates])
        chosen_unique, chosen_unique_inv = np.unique(chosen_dates, return_inverse=True)

        lag_series_temp = np.empty([len(chosen_unique)])
        lag_series_temp[:] = np.nan

        dates_we_have = np.isin(chosen_unique, self.catchment_temp[0])
        catchment_dates = np.isin(self.catchment_temp[0], chosen_unique)
        lag_series_temp[dates_we_have] = np.asarray(self.catchment_temp[1])[catchment_dates]
 

        if len(chosen_unique) != np.sum(dates_we_have) and interp:
            missing_dates_ind = np.invert(dates_we_have)
            missing_dates = chosen_unique[missing_dates_ind]            

            ref_date = dt.datetime(1950,1,1)
            temp_date_sec = np.asarray([(this_date - ref_date).total_seconds() for this_date in self.catchment_temp[0]])
            missing_dates_sec = np.asarray([(this_date - ref_date).total_seconds() for this_date in missing_dates])
 
            lag_series_temp[missing_dates_ind] = np.interp(missing_dates_sec, temp_date_sec, self.catchment_temp[1])


        lag_series = lag_series_temp[chosen_unique_inv]

        return lag_series

    def makeTempRegrModel(self, temp_series_lags=None):
        # Makes a multiple linear regression model for the temperature based on the EA temperature archive data for the river. It regresses against the temperature series of mean catchment temperature from the WRF model (self.catchment_temp see above) at the lags given by the list 'temp_series_lags' and also the vertical height of the EA measurement station

        if temp_series_lags is None:
            temp_series_lags=[0,1,2]

        # take proper copies otherwise when we shift the time we move the underlying data
        obs_temps = np.asarray(copy.deepcopy(self.temp_gauge_data[1]), 'float')
        comp_dates = np.asarray(copy.deepcopy(self.temp_gauge_data[0]))
        obs_heights = np.asarray(copy.deepcopy(self.temp_gauge_data[4]))

        # trim to only observations within the period we have WRF data
        remove_dates = np.logical_or(comp_dates < np.min(self.catchment_temp[0]),
                            comp_dates > np.max(self.catchment_temp[0]))

        obs_temps = obs_temps[~remove_dates]
        comp_dates = comp_dates[~remove_dates]
        obs_heights = obs_heights[~remove_dates]

        if len(obs_temps) < self.temp_obs_dist_threshold:
            print(self.river_name + ': Insufficient observations for training temp model')
            return

        dependents = np.zeros([len(comp_dates), len(temp_series_lags) + 3])

        for this_ind, this_dt in enumerate(comp_dates):
           comp_dates[this_ind] = this_dt + dt.timedelta(hours=12)

        for this_col, this_lag in enumerate(temp_series_lags):
            dependents[:,this_col+1] = self.getWRFTempTimeLagSeries(this_lag, comp_dates)

        # Add in the heights of the relevant
        dependents[:,-2] = obs_heights
        dependents[:,-1] = self.catchment_area
        dependents[:,0] = obs_temps
        dependents = dependents[~np.isnan(dependents).any(axis=1)]

        temp_model = linear_model.LinearRegression()
        temp_model.fit(dependents[:,1:], dependents[:,0])

        # Add the data to the river object. We include the data the model was trained on so we can get back scores etc.
        self.temp_model_fitdata = dependents
        self.temp_model = temp_model
        self.temp_model_lags = temp_series_lags

        return [dependents, temp_model, temp_series_lags]

    def retrieveGenericTempModel(self,temp_model_file=None):
        # Add a temp model from file e.g. for rivers with no observation data
        if temp_model_file is None:
            temp_model_file = 'generic_temp_model.pk1'

        with open(temp_model_file, 'rb') as open_file:
            temp_model_dict = pickle.load(open_file)

        self.temp_model = temp_model_dict['temp_model']
        self.temp_model_lags = temp_model_dict['temp_model_lags']
        self.temp_model_fitdata = temp_model_dict['temp_model_fitdata']

        return [temp_model_dict['temp_model_fitdata'], temp_model_dict['temp_model'], temp_model_dict['temp_model_fitdata']]

    def getTempModelSeries(self, start_date, end_date, siteZ=0, step=1, **kwargs):
        # Get a modelled series of temperature data between the dates given
        if not hasattr(self, 'temp_model'):
            print('No temperature model -- creating default model')
            self.makeTempRegrModel()

        if (not 'override' in kwargs or not kwargs['override']) and math.floor(step) == step and start_date.hour == 0:
            print('Temp model is trained at midday so changing to midday values')
            start_date = start_date + dt.timedelta(hours=12)
            end_date = end_date + dt.timedelta(hours=12)

        date_list = rfun.make_date_list(start_date, end_date,step)

        dependents = np.zeros([len(date_list), len(self.temp_model_lags)+2])

        for this_col, this_lag in enumerate(self.temp_model_lags):
            dependents[:, this_col] = self.getWRFTempTimeLagSeries(this_lag, date_list)

        dependents[:,-2] = siteZ
        dependents[:,-1] = self.catchment_area

        date_list_regr = np.asarray(date_list)[~np.isnan(dependents).any(axis=1)]
        dependents = dependents[~np.isnan(dependents).any(axis=1)]

        temp_series = self.temp_model.predict(dependents)

        temp_series_complete = rfun.missing_val_interpolate(date_list_regr, temp_series, complete_dates=date_list)

        if  len(temp_series_complete) - len(temp_series) > 0:
            print(self.river_name + ': ' + str(len(temp_series_complete) - len(temp_series))+' missing temp model values -- interpolation used')

        return [date_list, temp_series_complete]

    ### Neural Network flux prediction methods
    def prepNeuralNetData(self, prep_dates, return_flux):
        # Get and arrange the data in the format required for training the neural network
        if not hasattr(self, 'precipitation_sums_lags'):
            print('Oops NN not setup yet')
            return

        # Extend the data so that it gets the lags/sums correct for the first days
        max_lag_sum = max(np.max(np.max(self.temp_sums_lags)), np.max(np.max(self.precipitation_sums_lags)))
        prep_dates_ext = [prep_dates[0] - dt.timedelta(days=int(max_lag_sum)), prep_dates[1]]

        precipitation_data = np.asarray(self.getWRFDailyPrecipitationSeries(prep_dates_ext[0], prep_dates_ext[1], 0.5)[1])
        temp_data = np.asarray(self.getWRFDailyTempSeries(prep_dates_ext[0], prep_dates_ext[1])[1])

        nn_input_data = nn_fun.nn_create_run_data(precipitation_data, temp_data, self.precipitation_sums_lags, self.temp_sums_lags)

        date_list = np.asarray(rfun.make_date_list(prep_dates_ext[0], prep_dates_ext[1],1))

        if return_flux:
            if not hasattr(self, 'flux'):
                print(self.river_name+': No flux series, attempting to retrieve')
                if not self.retrieveFlux() or not self.retrievFlux()[0]:
                    print(self.river_name + ': Insufficient training data for NN model')
                    return

            flux_data = np.asarray(self.getSeriesDates(prep_dates_ext[0], prep_dates_ext[1], 'flux')[1])

        else:
            flux_data = np.zeros(len(date_list))

        # remove nan rows
        all_data = np.concatenate((flux_data[:,None], nn_input_data), axis=1)
        date_list = date_list[~np.isnan(all_data).any(axis=1)]
        all_data = all_data[~np.isnan(all_data).any(axis=1)]

        # add catchment for generic model
        if self.nn_domain == 'generic':
            all_data = np.append(all_data, np.ones([len(all_data), 1])*self.catchment_area, axis=1)

        return [all_data[:,0], all_data[:,1:], date_list]

    def makeNeuralNetFluxModel(self, no_epochs=200, train_dates=None):
        if not hasattr(self, 'precipitation_sums_lags'):
            print(self.river_name+': No sum/lag settings for NN. Setting defaults')
            self.precipitation_sums_lags = [[7,14,21,30,60], [1,2,3,4,5,6]]
            self.temp_sums_lags = [[7,14,28], [1,2,3,4,5,6]]

        self.nn_domain = 'local'

        if train_dates is None:
            train_dates = [dt.datetime(2005,1,1), dt.datetime(2007,12,30)]

        try:
            [train_flux, train_data] = self.prepNeuralNetData(train_dates, True)[0:2]
        except TypeError:
            print('No flux data available')
            return

        if len(train_flux) < self.nn_model_obs_threshold:
            print(self.river_name + ': Insufficient training data for NN model')
            return

        # train the model
        nn_output = nn_fun.runNNtrain(train_flux, train_data, no_epochs)

        # save the model seperately as it don't like being pickled
        self.nn_model_file = self.river_name + '_nn_model.h5'
        nn_output[1].save(self.nn_model_file)

        self.nn_scaler = nn_output[0]

        return [self.nn_model_file, self.nn_scaler]

    def retrieveGenericNeuralNetFluxModel(self, generic_model_files=None):
        if generic_model_files is None:
            generic_model_files = ['generic_nn.h5', 'generic_nn_train']

        self.nn_model_file = generic_model_files[0]

        with open(generic_model_files[1], 'rb') as open_file:
            nn_scaler = pickle.load(open_file)
        open_file.close()

        self.precipitation_sums_lags = [[7,14,21,30,60], [1,2,3,4,5,6]]
        self.temp_sums_lags = [[7,14,28], [1,2,3,4,5,6]]

        self.nn_scaler = nn_scaler['nn_scaler']
        self.nn_domain = 'generic'

        return [self.nn_model_file, self.nn_scaler]

    def getNeuralNetFluxSeries(self, start_date, end_date):
        # Get a modelled flux series
        # prep data
        [nn_data, flux_dates] = self.prepNeuralNetData([start_date, end_date], False)[1:]
        # scale data
        nn_data_scale = self.nn_scaler.transform(nn_data)
        # run through model
        nn_model = load_model(self.nn_model_file)
        flux_preds = nn_model.predict(nn_data_scale)

        date_list = rfun.make_date_list(start_date, end_date,1)
        flux_series_complete = rfun.missing_val_interpolate(flux_dates, flux_preds, complete_dates=date_list)

        if  len(flux_series_complete) - len(flux_preds) > 0:
            print(self.river_name + ': ' + str(len(flux_series_complete) - len(flux_preds))+' missing NN flux model values -- interpolation used')

        return [date_list, flux_series_complete]

    def testNNoos(self, oos_dates=None):
        if oos_dates is None:
            oos_dates = [dt.datetime(2011,1,1), dt.datetime(2011,12,30)]

        [oos_flux, oos_data] = self.prepNeuralNetData(oos_dates, True)[0:2]

        if not oos_flux.size:
            print(self.river_name + ': No out of sample flux data')
            return [np.nan, np.nan, np.nan]

        # scale the data
        oos_data_scale = self.nn_scaler.transform(oos_data)

        # predict on out of sample
        nn_model = load_model(self.nn_model_file)
        oos_preds = nn_model.predict(oos_data_scale)

        # plot it up
        plt.figure(self.river_name)
        plt.plot(oos_flux)
        plt.plot(oos_preds)

        plt.savefig(self.river_name+'_nnfit2.png')
        plt.show()

        # and generate error metrics
        oos_preds = np.reshape(oos_preds, [len(oos_preds)])
        error_metrics = nn_fun.modelErrorMetrics(oos_preds, oos_flux)

        return error_metrics

    def getSalinitySeries(self, start_date, end_date):
        date_list = [start_date + dt.timedelta(int(i)) for i in np.arange(0, (end_date - start_date).days + 1)]
        salinity_list = list(np.ones(len(date_list)) * self.salinity)
        return date_list, salinity_list


class RiverMulti(River):
    def retrieveFlux(self, ceh_data_path=None):
        flux_dict = {}
        all_gauge_lon = []
        all_gauge_lat = []

        if ceh_data_path==None:
            if not self.ceh_data_path:
                print(self.river_name + ": Can't retrieve flux no data path specified")
                return
            ceh_data_path = self.ceh_data_path

        for this_gauge in self.ceh_gauge_id_no:
            # run method from super for each id
            [flux_dict[this_gauge], this_gauge_ll]  = River.retrieveFlux(self, this_gauge, ceh_data_path=ceh_data_path)
            all_gauge_lon.append(this_gauge_ll[0])
            all_gauge_lat.append(this_gauge_ll[1])

        # find the dates for which all gauge data exists
        common_dates = list(flux_dict.values())[0][0]

        for this_keys, this_gauge_flux in flux_dict.items():
            common_dates = list(set(common_dates).intersection(set(this_gauge_flux[0])))

        common_dates.sort()

        # now sum all the gauge data for those dates
        total_flux = []
        for this_date in common_dates:
            this_flux = 0
            for this_key, this_gauge_flux in flux_dict.items():
                if [this_gauge_flux[1][x] for x,y in enumerate(this_gauge_flux[0]) if y == this_date]:
                    this_flux = this_flux + [this_gauge_flux[1][x] for x,y in enumerate(this_gauge_flux[0]) if y == this_date][0]
            total_flux.append(this_flux)

        flux = [common_dates, total_flux]
        self.flux = flux
        self.gauge_lon = all_gauge_lon
        self.gauge_lat = all_gauge_lat

        return [flux, [all_gauge_lon, all_gauge_lat]]

    def retrieveGaugeCatchment(self, ceh_data_path=None):
        # Overwritten method to get multiple shape files

        catchment_poly_points = []
        catchment_bbox = []
        catchment_area = 0

        if ceh_data_path==None:
            if not self.ceh_data_path:
                print(self.river_name + ": Can't retrieve flux no data path specified")
                return
            ceh_data_path = self.ceh_data_path

        for this_gauge in self.ceh_gauge_id_no:
            [this_cpp, this_bbox, this_ca] = River.retrieveGaugeCatchment(self, this_gauge, ceh_data_path=ceh_data_path)
            catchment_poly_points.append(this_cpp)
            catchment_bbox.append(this_bbox)
            catchment_area = catchment_area + this_ca

        self.catchment_poly_points = catchment_poly_points
        self.catchment_bbox = catchment_bbox
        self.catchment_area = catchment_area

        return [catchment_poly_points, catchment_bbox, catchment_area]

    def makeWRFCatchmentFactors(self):
        # Overwritten method to deal with multiple catchment areas
        all_cf = []

        for i in range(0, len(self.catchment_poly_points)):
            all_cf.append(River.makeWRFCatchmentFactors(self, self.catchment_bbox[i], self.catchment_poly_points[i]))

        wrf_catchment_factors = np.zeros(all_cf[0].shape)

        for this_cf in all_cf:
            wrf_catchment_factors = wrf_catchment_factors + this_cf

        self.wrf_catchment_factors = wrf_catchment_factors

        return wrf_catchment_factors


class RiverMultiRoi(RiverMulti):
    def _fluxNlocParsing(self, gauge_id_no, ceh_data_path):
        quality_ok = ['31', 'C', 'F', 'B', 'P', '*']
        gauge_flux_file = '{}flux_data/{}_flux.csv'.format(ceh_data_path, gauge_id_no)
        flux_data_raw = np.asarray(rfun.read_csv_unheaded(gauge_flux_file, 4))

        time_dt = np.asarray([dt.datetime.strptime('{} {}'.format(this_data[0], this_data[1]), '%Y/%m/%d %H:%M:%S') for this_data in flux_data_raw[0:2,:].T])
        flux = flux_data_raw[2,:]
        qual_flag = flux_data_raw[3,:]
        qual_pass = np.isin(qual_flag, quality_ok)
        qual_pass = np.logical_and(qual_pass, np.invert(flux == ''))
        flux_data = [time_dt[qual_pass], np.asarray(flux[qual_pass], dtype=float)]

        gauge_loc_file = '{}flux_data/{}_loc.txt'.format(ceh_data_path, gauge_id_no)
        loc_data_raw = np.loadtxt(gauge_loc_file, delimiter=',', dtype=str)
        gauge_lat = float(loc_data_raw[3])
        gauge_lon = float(loc_data_raw[4])

        return flux_data, gauge_lat, gauge_lon

    def retrieveGaugeCatchment(self, rosa_data_path, rose_gauge_no):
        # Overwritten method to get the rosa scale shape file
        catchment_shp_filestr = gb.glob('{}/{}/*.shp'.format(rosa_data_path, rose_gauge_no))

        this_catchment_shp = shapefile.Reader(catchment_shp_filestr[0][0:-4])

        # We want to turn the .shp file into a list of coordinates for the corners of the polygon, first we retrieve the list of coordinates
        if len(this_catchment_shp.shapeRecords()) == 1:
             catchment_poly_points = this_catchment_shp.shapeRecords()[0].shape.points
        else:
            print('Error - More than one catchment area defined')
            return

        # some are a bit complicated causing self intersections so smooth the boundaries a bit
        raw_poly = gm.Polygon(catchment_poly_points)
        smooth_poly = raw_poly.buffer(0.001)
        smooth_poly = smooth_poly.buffer(-0.001)
        catchment_poly_points = np.asarray([smooth_poly.exterior.xy[0], smooth_poly.exterior.xy[1]]).T

        # and a couple of other useful bits of info about the catchment: the bounding box and the total area of the catchment
        catchment_bbox = smooth_poly.bounds
        catchment_area = smooth_poly.area

        # need to reverse the lat/lon in poly points and bbox
        catchment_bbox = np.asarray(catchment_bbox).reshape([2,2])
        catchment_bbox = catchment_bbox[:,[1,0]]
        catchment_poly_points = np.asarray(catchment_poly_points)[:,[1,0]]

        # add tp river object
        self.catchment_poly_points = catchment_poly_points
        self.catchment_bbox = catchment_bbox
        self.catchment_area = catchment_area

        return [catchment_poly_points, catchment_bbox, catchment_area]

    def makeWRFCatchmentFactors(self):
        wrf_catchment_factors = River.makeWRFCatchmentFactors(self, self.catchment_bbox, self.catchment_poly_points)
        return wrf_catchment_factors

    def retrieveTempObsData(self, EA_data_path=None):
        # For ROI rivers we have temperature time series data of temperature in files
        # return list of arrays: [date, temp, lon, lat, height, distance from mouth]

        if EA_data_path==None:
            if not self.EA_data_path:
                print(self.river_name + ": Can't retrieve temp observations no data path specified")
                return
            EA_data_path = self.EA_data_path
        
        gauge_temps = []
        for this_gauge_id in self.ceh_gauge_id_no:
            try:
                temperature_filestr = gb.glob('{}/{}temp.csv'.format(EA_data_path, rose_gauge_no)) 
                filedata = np.loadtxt(temperature_filestr[0],delimiter=',',dtype=str)

                dates_dt = []
                temps = []

                for this_row in filedata:
                    dates_dt.append(dt.datetime.strptime('{} {}'.format(this_row[0], this_row[1]),'%d/%m/%Y %H:%M:%S'))
                    temps.append(float(this_row[2]))
                gauge_temps.append([np.asarray(date_dt), np.asarray(temps)])

            except:
                pass

        # Do something about having multiple measurements from different gauges
        if len(gauge_temps) == 0:
            print('No temperature observations available')
        else:
            all_dates_list = np.unique([item for sublist in gauge_temps for item in sublist[0]])
            mean_temps = []
            for this_date in all_dates_list:
                pass 
            

        # Throw away non-midday temperatures



        # Zero distance from gauge


        # Gauges are relatively close to the mouths and we don't know heights so put a small number in

        site_dists = np.zeros(len(db_data[0]))
        for i in range(0, len(site_dists)):
            site_dists[i] = rfun.ll_dist(float(self.mouth_lon), float(self.mouth_lat), db_data[2][i], db_data[3][i])

        db_data[5] = site_dists
        self.temp_gauge_data_raw = copy.deepcopy(db_data)

        for this_ind,this_col in enumerate(db_data):
            db_data[this_ind] = np.asarray(this_col)[allowed_sites]

        self.temp_gauge_data = db_data

        return db_data


class RiverEMODNET(River):
    def _fluxNlocParsing(self, gauge_id_no, ceh_data_path):

        if gauge_id_no[0:2] in ['EX', 'IF']:
            country_code = 'GL'
        else:
            country_code = 'NO'

        if hasattr(self, 'qc_pass_codes'):
            qc_pass_codes = self.qc_pass_codes
        else:
            qc_pass_codes = [1,5]

        this_filestr = '{}flux_data/{}/{}_TS_RF_{}.nc'.format(ceh_data_path, gauge_id_no, country_code, gauge_id_no)

        try:
            this_gauge_nc = nc.Dataset(this_filestr, 'r')
        except OSError:
            print("Can't find {}".format(this_filestr))
            return

        this_flow = np.squeeze(this_gauge_nc['RVFL'][:])
        this_flow_qc = np.squeeze(this_gauge_nc['RVFL_QC'][:])
        this_flow_time = np.squeeze(this_gauge_nc['TIME'][:])

        this_flow_qc_bool = np.isin(this_flow_qc, [1,5])
        this_flow = this_flow[this_flow_qc_bool]
        this_flow_time = this_flow_time[this_flow_qc_bool]

        this_flow_int = np.arange(np.ceil(np.min(this_flow_time)), np.floor(np.max(this_flow_time))+1)
        this_flow_interped = np.interp(this_flow_int, this_flow_time, np.squeeze(this_flow))

        this_flow_dt = np.asarray([self.ref_date + dt.timedelta(days=float(this_time)) for this_time in this_flow_int])

        flux_data = [this_flow_dt, this_flow_interped]

        gauge_lat = this_gauge_nc.variables['LATITUDE'][:][0]
        gauge_lon = this_gauge_nc.variables['LONGITUDE'][:][0]

        return flux_data, gauge_lat, gauge_lon

    def retrieveGaugeCatchment(self, gauge_name=None):
        if gauge_name == None:
            gauge_name = self.ceh_gauged_river_names

        # Overwritten method to get the rosa scale shape file
        catchment_shp_filestr = gb.glob('{}catchments/{}/*.shp'.format(self.ceh_data_path, gauge_name))

        this_catchment_shp = shapefile.Reader(catchment_shp_filestr[0])

        # We want to turn the .shp file into a list of coordinates for the corners of the polygon, first we retrieve the list of coordinates
        all_polys = []
        for this_rec in this_catchment_shp.shapeRecords():
            all_polys.append(this_rec.shape.points)

        raw_polys = []
        for this_ll in all_polys:
            raw_polys.append(gm.Polygon(this_ll))

        catchment_poly_points = cascaded_union(raw_polys).exterior.xy
        catchment_poly_points = np.asarray([np.asarray(catchment_poly_points[0]), np.asarray(catchment_poly_points[1])]).T
    
        # some are a bit complicated causing self intersections so smooth the boundaries a bit
        raw_poly = gm.Polygon(catchment_poly_points)
        smooth_poly = raw_poly.buffer(0.001)
        smooth_poly = smooth_poly.buffer(-0.001)
        catchment_poly_points = np.asarray([smooth_poly.exterior.xy[0], smooth_poly.exterior.xy[1]]).T

        # and a couple of other useful bits of info about the catchment: the bounding box and the total area of the catchment
        catchment_bbox = smooth_poly.bounds
        catchment_area = smooth_poly.area

        # need to reverse the lat/lon in poly points and bbox
        catchment_bbox = np.asarray(catchment_bbox).reshape([2,2])
        catchment_bbox = catchment_bbox[:,[1,0]]
        catchment_poly_points = np.asarray(catchment_poly_points)[:,[1,0]]

        # add tp river object
        self.catchment_poly_points = catchment_poly_points
        self.catchment_bbox = catchment_bbox
        self.catchment_area = catchment_area

        return [catchment_poly_points, catchment_bbox, catchment_area]

class RiverMultiEMODNET(RiverEMODNET):
    def retrieveFlux(self, ceh_data_path=None):
        flux_dict = {}
        all_gauge_lon = []
        all_gauge_lat = []

        if ceh_data_path==None:
            if not self.ceh_data_path:
                print(self.river_name + ": Can't retrieve flux no data path specified")
                return
            ceh_data_path = self.ceh_data_path

        for this_gauge in self.ceh_gauge_id_no:
            # run method from super for each id
            [flux_dict[this_gauge], this_gauge_ll]  = River.retrieveFlux(self, this_gauge, ceh_data_path=ceh_data_path)
            all_gauge_lon.append(this_gauge_ll[0])
            all_gauge_lat.append(this_gauge_ll[1])

        # find the dates for which all gauge data exists
        common_dates = list(flux_dict.values())[0][0]

        for this_keys, this_gauge_flux in flux_dict.items():
            common_dates = list(set(common_dates).intersection(set(this_gauge_flux[0])))

        common_dates.sort()

        # now sum all the gauge data for those dates
        total_flux = []
        for this_date in common_dates:
            this_flux = 0
            for this_key, this_gauge_flux in flux_dict.items():
                if [this_gauge_flux[1][x] for x,y in enumerate(this_gauge_flux[0]) if y == this_date]:
                    this_flux = this_flux + [this_gauge_flux[1][x] for x,y in enumerate(this_gauge_flux[0]) if y == this_date][0]
            total_flux.append(this_flux)

        flux = [common_dates, total_flux]
        self.flux = flux
        self.gauge_lon = all_gauge_lon
        self.gauge_lat = all_gauge_lat

        return [flux, [all_gauge_lon, all_gauge_lat]]

    def retrieveGaugeCatchment(self, ceh_data_path=None):
        # Overwritten method to get multiple shape files

        catchment_poly_points = []
        catchment_bbox = []
        catchment_area = 0

        if ceh_data_path==None:
            if not self.ceh_data_path:
                print(self.river_name + ": Can't retrieve flux no data path specified")
                return
        else:
            self.ceh_data_path = ceh_data_path

        for this_gauge in self.ceh_gauged_river_names:
            [this_cpp, this_bbox, this_ca] = RiverEMODNET.retrieveGaugeCatchment(self, this_gauge)
            catchment_poly_points.append(this_cpp)
            catchment_bbox.append(this_bbox)
            catchment_area = catchment_area + this_ca

        self.catchment_poly_points = catchment_poly_points
        self.catchment_bbox = catchment_bbox
        self.catchment_area = catchment_area

        return [catchment_poly_points, catchment_bbox, catchment_area]

    def makeWRFCatchmentFactors(self):
        # Overwritten method to deal with multiple catchment areas
        all_cf = []

        for i in range(0, len(self.catchment_poly_points)):
            all_cf.append(River.makeWRFCatchmentFactors(self, self.catchment_bbox[i], self.catchment_poly_points[i]))

        wrf_catchment_factors = np.zeros(all_cf[0].shape)

        for this_cf in all_cf:
            wrf_catchment_factors = wrf_catchment_factors + this_cf

        self.wrf_catchment_factors = wrf_catchment_factors

        return wrf_catchment_factors


class RiverSimple(River):
    def __init__(self, river_name, mouth_lon, mouth_lat):
        self.river_name = river_name
        self.mouth_lon = mouth_lon
        self.mouth_lat = mouth_lat

    def addFlux(self, date_list, flux_list):
        self.flux = [date_list, flux_list]

    def getGaugeFluxSeries(self, start_date, end_date):
        series_data = self.getSeriesDates(start_date, end_date, 'flux')
        return series_data

    def getTempModelSeries(self, start_date, end_date):
        date_list = [start_date + dt.timedelta(int(i)) for i in np.arange(0, (end_date - start_date).days + 1)]
        date_yday = np.asarray([this_date.timetuple().tm_yday for this_date in date_list])
        temp_series_complete = self.temperature_function(date_yday) 
        return [date_list, temp_series_complete]

