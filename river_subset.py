import netCDF4 as nc
import numpy as np
import pickle as pk
import copy 
import datetime as dt

import FVCOM_write_functions as fwf


rivers_to_keep = ['Fowey', 'Seaton', 'Tiddy', 'Lynher', 'Tamar', 'Tavy', 'Plym', 'Yealm', 'Erme', 'Avon']

river_dict_file = 'tamar_v2_2015.pk1' 
mesh_file = 'tamar_v2_grd.dat'

output_file = 'tamar_v2_2015_riv'

with open(river_dict_file, 'rb') as input_file:
	river_dict = pk.load(input_file)
input_file.close()

dict_keys = copy.deepcopy(list(river_dict.keys()))
for this_key in dict_keys:
	if this_key not in rivers_to_keep:
		del river_dict[this_key]

river_dict = fwf.assign_river_nodes(river_dict, mesh_file, max_discharge=150)
fwf.write_FVCOM_netcdf(river_dict, output_file, min_temp=1, max_temp=30, start_date=dt.datetime(2015,1,1), end_date = dt.datetime(2015,12,31))

ediment_types = ['mud_1', 'mud_2', 'sand_1', 'sand_2']
flow_value = 0.005

#############################
river_nc = nc.Dataset(output_file + '.nc', 'a')

time_dim = river_nc.dimensions['time'].size
rivers_dim = river_nc.dimensions['rivers'].size

flow_val_exp = np.ones([time_dim, rivers_dim])*flow_value

for this_sediment in sediment_types:
    this_sed_var = river_nc.createVariable(this_sediment, 'f4', ('time','rivers'))
    this_sed_var.long_name = 'Mud mud glorious mud: ' + this_sediment
    this_sed_var.units = 'kgm^-3'
    this_sed_var[:] = flow_val_exp

river_nc.close()


