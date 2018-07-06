A set of functions for modelling the river input into a FVCOM 

river_dict is then a dictionary of the rivers used in the domain





Neural network flow models
--------------------------

There is an option to 


Th neural network training functions are in nn_functions.py. The python keras package is used for modelling. The nn design used is a Sequential model with two layers (Dense layers, the standard fully connected nn layers), one layer of width    and one with a single output node. The loss function used is mean squared error and the Adam optimiser.

The training data is a series of lagged series and window summed   . If I could get reccurent nns working properly in Python I probably should use them instead.




WRF data
--------

WRF data used are the variables T2, RAINNC, and TIMES. It is setup to take th    hey are taken from monthly netcdf files. An example shell script for using the ncecat program to do this is included.

The WRF grid data comes from a file 



CEH river gauge data
--------------------
The UK rivers are modelled on the 





Environment Agency river temperature archive
--------------------------------------------







Manually provided data
----------------------



flow_ratio
total_estuary_length
length_locate_mesh_to_mhw


The flow_ratio and the relationship between length down estuary and salinity are based on Uncles et al. (2015)



