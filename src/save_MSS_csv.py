import hdf5_getters
import numpy as np
import pandas as pd
import os

mydicts = []
MSS_path = "../data/MillionSongSubset"
paths = []
for root, _, files in os.walk(MSS_path):
    for f in files:
        path = os.path.relpath(os.path.join(root, f), ".")
#     if h5:
#         h5.close()
        paths.append(path)
    
for path in paths:
    h5 = hdf5_getters.open_h5_file_read(path)

    # get all getters
    getters = filter(lambda x: x[:4] == 'get_', hdf5_getters.__dict__.keys())
    # getters.remove("get_num_songs") # special case

    dct = {}
    for getter in getters:
        res = hdf5_getters.__getattribute__(getter)(h5)
        dct[getter[4:]] = res
    mydicts.append(dct)
    h5.close()
    
df = pd.DataFrame(mydicts)
df.info()
df.to_csv(MSS_path + '.csv')