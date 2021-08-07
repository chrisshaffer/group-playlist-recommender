import hdf5_getters
import numpy as np
import pandas as pd

hdf5path = '/home/chris/DSI/couples-playlist-recommender/data/TRAXLZU12903D05F94.h5'

h5 = hdf5_getters.open_h5_file_read(hdf5path)

# get all getters
getters = filter(lambda x: x[:4] == 'get_', hdf5_getters.__dict__.keys())
# getters.remove("get_num_songs") # special case

dct = {}
for getter in getters:
	res = hdf5_getters.__getattribute__(getter)(h5)
	# if res.__class__.__name__ == 'ndarray':
	# 	print(f'{getter[4:]}: shape ={res.shape}')
	# else:
	dct[getter[4:]] = res

df = pd.DataFrame(dct)

h5.close()

