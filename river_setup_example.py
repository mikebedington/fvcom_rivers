# Load standard libraries
import pickle
import sys

# Load river class and associated functions
import river as r
import river_functions as rfun
import FVCOM_write_functions as ncfun
import nn_functions as nnfun

'''
Example of file to model rivers and output a netCDF in the form used by FVCOM.

It takes the rivers in 'stations.csv' and creates a river_dict based on these

'''

#save_file = 'loc_v0_riv_kings.pk1'
#nc_file = 'loc_v0_riv_nolim'
save_file = 'test.pk1'
nc_file = 'test_test'

mesh_file = 'loc_v0_grd.dat'

ceh_data_path = '/users/modellers/mbe/Data/CEH_gauges/' 
wrf_data_path = '/users/modellers/mbe/Data/WRF/' 
EA_data_path = '/users/modellers/mbe/EA_water_temp_archive/sqlite_db/'

# Load the river names
river_list_names = rfun.read_csv_unheaded('stations.csv', 4)

all_r_names = river_list_names[0]

for this_ind, this_name in enumerate(all_r_names):
	all_r_names[this_ind] = this_name.strip()

if bool(len(set(all_r_names)) - len(all_r_names)):
	print('Error - Duplicate river name')
	sys.exit(0)

# Initialise a list of the rivers
print('Initialising rivers')
river_dict = {}
for i in range(0, len(river_list_names[0])):
	try:
		float(river_list_names[1][i].strip())
		river_dict[river_list_names[0][i].strip()] = r.River(river_list_names[0][i].strip(), river_list_names[1][i].strip(), river_list_names[2][i].strip(), river_list_names[3][i].strip())

	except ValueError:
		id_list = river_list_names[1][i]
		id_list = id_list.replace('[','')
		id_list = id_list.replace(']','')
		id_list = id_list.split(';')

		try: 
			for this_ind, this_gauge_id in enumerate(id_list):
				id_list[this_ind] = this_gauge_id.strip()
				float(this_gauge_id.strip())
	
			river_dict[river_list_names[0][i].strip()]  = r.RiverMulti(river_list_names[0][i].strip(), id_list, river_list_names[2][i].strip(), river_list_names[3][i].strip())

		except ValueError:
			print(river_list_names[0][i].strip() + ': No gauge id, assuming ungauged')
			river_dict[river_list_names[0][i].strip()] = r.River(river_list_names[0][i].strip(), False, river_list_names[2][i].strip(), river_list_names[3][i].strip())


ceh_data_path = '/users/modellers/mbe/Data/CEH_gauges'

# add a variety of useful things
print('Getting flux and catchments')
for this_river in river_dict.values():
	this_river.ceh_data_path = ceh_data_path
	this_river.wrf_data_path = wrf_data_path
	this_river.EA_data_path = EA_data_path
	this_river.retrieveFlux();
	this_river.retrieveGaugeCatchment();
	this_river.makeWRFCatchmentFactors();


# Get the WRF data for the catchments
#There is a memory leak in the implementation of the netcdf library so this function which is more elegant and flexible was a pain to use as you have to reload for each year 
#rfun.add_year_rain_temp_series('locate_rivers.pk1', [2005])
# Instead use this version which uses netcdfs of only the variables of interest concatenated over a whole year each
print('Getting WRF data')
rfun.add_year_both_series(river_dict.values(), [2005,2006,2007,2011], wrf_data_path)

temp_db_path = '/data/euryale4/backup/mbe/Data/EA_water_temp_archive/sqlite_db/'

print('Making temp and flux models')
for this_key, this_river in river_dict.items():
	print(this_river.river_name)
	this_river.retrieveTempObsData(temp_db_path);
	this_river.makeTempRegrModel();
	this_river.makeNeuralNetFluxModel();

# make the generic nn model and add to any rivers for which there was insufficient training data
print('Making generic nn model')
nnfun.create_generic_nn(river_dict);

for this_river in river_dict.values():
	if not hasattr(this_river, 'nn_model_file'):
		print(this_river.river_name + ': adding generic nueral net model')
		this_river.retrieveGenericNeuralNetFluxModel();

# make the generic temp regr model and add to any rivers for which there was insufficent training data
print('Making generic temp model')
rfun.make_generic_temp_model(river_dict);

for this_river in river_dict.values():
	if not hasattr(this_river, 'temp_model'):
		print(this_river.river_name + ': adding generic temp model')
		this_river.retrieveGenericTempModel();


#add flow ratios (from Uncles et al) and estuary lengths etc (manually measured)
print('Adding manual data')
manual_river_data = rfun.read_csv_dict('manual_river_data.csv')
manual_river_list = manual_river_data['River_name']


for this_key, this_river in river_dict.items():
	if [x for x,y in enumerate(manual_river_list) if y == this_river.river_name]:
		this_river.flow_ratio = float([manual_river_data['flow_ratio'][x] for x,y in enumerate(manual_river_list) if y == this_river.river_name][0])
		this_river.total_estuary_length = float([manual_river_data['total_estuary_length'][x] for x,y in enumerate(manual_river_list) if y == this_river.river_name][0])
		this_river.length_locate_mesh_to_mhw = float([manual_river_data['length_mesh_to_mhw'][x] for x,y in enumerate(manual_river_list) if y == this_river.river_name][0])
	else:
		this_river.flow_ratio=1
		this_river.total_estuary_length = 1
		this_river.length_locate_mesh_to_mhw = 1


with open(save_file, 'wb') as output_file:
       pickle.dump(river_dict, output_file, pickle.HIGHEST_PROTOCOL)
output_file.close()

print('Writing netcdf file')
river_dict = ncfun.assign_river_nodes(river_dict, mesh_file) 
ncfun.write_FVCOM_netcdf(river_dict, nc_file, min_temp=0, max_temp=30)



