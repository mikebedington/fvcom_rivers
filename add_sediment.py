import netCDF4 as nc
import numpy as np

river_nc_filestr = 'tamar_v2_2005_2006_riv.nc'
sediment_types = ['mud_1', 'mud_2', 'sand_1', 'sand_2']

def sediment_function(Q, cnst):
	spm = cnst * ((Q/10) ** 1.2)
	spm[Q < 10] = 5
	return spm
 

#############################
river_nc = nc.Dataset(river_nc_filestr, 'r+')

time_dim = river_nc.dimensions['time'].size
river_flux = river_nc.variables['river_flux'][:]

river_name_list_raw = river_nc.variables['name_list_real'][:]
river_name_list = []

for this_row in river_name_list_raw:
	river_name_list.append(str(b''.join(this_row), 'utf-8'))

spm_raw = []
for this_river, this_q in zip(river_name_list, river_flux.T):
	if 'Tamar' in this_river:
		cnst = 4.6
	else:
		cnst = 3.2
	spm_raw.append(sediment_function(this_q, cnst))

spm_val = np.vstack(spm_raw).T/1000

for this_sediment in sediment_types:
	#this_sed_var = river_nc.createVariable(this_sediment, 'f4', ('time','rivers'))
	#this_sed_var.long_name = 'Mud mud glorious mud: ' + this_sediment
	#this_sed_var.units = 'kgm^-3'
	if this_sediment == 'mud_1':
		river_nc.variables[this_sediment][:] = spm_val
	else:
		river_nc.variables[this_sediment][:] = np.zeros(np.shape(river_flux))

river_nc.close()


