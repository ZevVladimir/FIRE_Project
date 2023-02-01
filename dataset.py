import numpy as np
import matplotlib.pyplot as plt
import h5py

filename = "galaxies_catalog_tng100_099.hdf5"
array = np.empty(0)

with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
    #print(type(f[a_group_key])) 

    
    data = np.empty(f['catgrp_Group_M_Crit200'].shape, f['catgrp_Group_M_Crit200'].dtype)
    data = np.array(list(f['catgrp_Group_M_Crit200']))
    print("Data finished")
    print(data)
    primary = np.empty(f['catgrp_is_primary'].shape, f['catgrp_is_primary'].dtype)
    primary = np.array(list(f['catgrp_is_primary']))
    print("Primary Finished")
    print(primary)

    for i in range(500000):
        if(i % 50000 == 0):
            print(i)
        if primary[i] == True:
            array = np.append(array, data[i])
    print("finished array")
    print(array)
    normalized_array = array/np.linalg.norm(array)
    
    print(normalized_array)

    graph = plt.hist(normalized_array, bins='auto')
    plt.show()

    # preferred methods to get dataset values:
    #ds_obj = f['catgrp_is_primary']      # returns as a h5py dataset object
    #ds_arr = f['catgrp_is_primary'][()]  # returns as a numpy array
    