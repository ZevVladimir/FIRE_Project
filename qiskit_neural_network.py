from locale import normalize
import numpy as np
import matplotlib.pyplot as plt
import h5py

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy import interpolate
from scipy import stats
from scipy import special


from qiskit import BasicAer
from qiskit.circuit.library import ZFeatureMap
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel 
from qiskit_machine_learning.algorithms import PegasosQSVC

from sklearn.model_selection import train_test_split
from qiskit.utils import algorithm_globals
from sklearn.svm import SVC



file = '/home/zeevvladimir/Personal_Project/TNG300_RF_data-20221026T024254Z-001/TNG300_RF_data/'
Group_M_Mean200_dm = np.load(file+'Group_M_Mean200_dm.npy')*1e10
GroupPos_dm = np.load(file+'GroupPos_dm.npy')/1000
GroupConc_dm = np.load(file+'GroupConc_dm.npy')
GroupEnv_dm = np.load(file+'GroupEnv_dm.npy')
GroupEnvAnn_dm = np.load(file+'GroupAnnEnv_R5_dm.npy')
GroupEnvTH_dm = np.load(file+'GroupEnv_dm_TopHat_1e11mass.npy')[:,15] #already masked for Mhalo>1e11
GroupSpin_dm = np.load(file+'GroupSpin_dm.npy')
GroupNsubs_dm = np.load(file+'GroupNsubs_dm.npy')
GroupVmaxRad_dm = np.load(file+'GroupVmaxRad_dm.npy')
Group_SubID_dm = np.load(file+'GroupFirstSub_dm.npy') #suhalo ID's
Group_Shear_dm = np.load(file+'GroupShear_qR_dm_1e11Mass.npy') #already,masked for Mhalo>1e11
print(Group_Shear_dm.shape)
SubVdisp_dm = np.load(file+'SubhaloVelDisp_dm.npy')
SubVmax_dm = np.load(file+'SubhaloVmax_dm.npy')
SubGrNr_dm = np.load(file+'SubhaloGrNr_dm.npy') #Index into the Group table of the FOF host/parent of Subhalo
SubhaloPos_dm = np.load(file+'SubhaloPos_dm.npy')/1000
count_dm = np.load(file+'GroupCountMass_dm.npy')
cent_count_dm = np.load(file+'GroupCountCentMass_dm.npy')
sat_count_dm = count_dm-cent_count_dm
GroupEnvTH_1_3 = np.load(file+'GroupEnv_dm_TopHat_1e11mass.npy')[:,7] #env at 1.3 Mpc
GroupEnvTH_2_5 = np.load(file+'GroupEnv_dm_TopHat_1e11mass.npy')[:,11] #env at 2.6 Mpc

#mask off the lowest mass halos
mass_mask = Group_M_Mean200_dm>1e11

#Interpolate the shear at the radius of the halo using group shear file
rEnv=np.logspace(np.log10(0.4),np.log10(10),20) #scales at which shear was calculated
rad=(np.load(file+'Group_R_Mean200_dm.npy')/1e3)[mass_mask] #halo radius
shear=np.zeros(len(rad))
for i in range(len(rad)):
    ShearFit=interpolate.InterpolatedUnivariateSpline(rEnv,Group_Shear_dm[i])
    shear[i]=ShearFit(1*rad[i])
    mask = shear<0; shear[mask]=1e-3 #To ameliorate unrealistic values
shear[mask].shape
rEnv

shear_1Mpc = Group_Shear_dm[:,7] # we also want shear value at approx at 1.3Mpc

# create group velocity dispersion and Vmax based on thos of most massive gal
parents_Vdisp = SubVdisp_dm[Group_SubID_dm]
parents_Vmax = SubVmax_dm[Group_SubID_dm]

#create testing cube
boxLen=137
maskBox=GroupPos_dm[mass_mask][:,0]<boxLen
maskBox*=GroupPos_dm[mass_mask][:,1]<boxLen
maskBox*=GroupPos_dm[mass_mask][:,2]<boxLen


#organize data for training/testing
#features
mass_train = Group_M_Mean200_dm[mass_mask][~maskBox]
mass_test = Group_M_Mean200_dm[mass_mask][maskBox]
env_train = GroupEnv_dm[mass_mask][~maskBox]
env_test = GroupEnv_dm[mass_mask][maskBox]
envann_train = GroupEnvAnn_dm[mass_mask][~maskBox]
envann_test = GroupEnvAnn_dm[mass_mask][maskBox]
envth_train = GroupEnvTH_dm[~maskBox]
envth_test = GroupEnvTH_dm[maskBox]
conc_train = GroupConc_dm[mass_mask][~maskBox]
conc_test = GroupConc_dm[mass_mask][maskBox]
spin_train = GroupSpin_dm[mass_mask][~maskBox]
spin_test = GroupSpin_dm[mass_mask][maskBox]
ngals_train = GroupNsubs_dm[mass_mask][~maskBox]
ngals_test = GroupNsubs_dm[mass_mask][maskBox]
vdisp_train = parents_Vdisp[mass_mask][~maskBox]
vdisp_test = parents_Vdisp[mass_mask][maskBox]
vmax_train = parents_Vmax[mass_mask][~maskBox]
vmax_test = parents_Vmax[mass_mask][maskBox]
vmax_rad_train = GroupVmaxRad_dm[mass_mask][~maskBox]
vmax_rad_test = GroupVmaxRad_dm[mass_mask][maskBox]
shear_train = shear[~maskBox]
shear_test = shear[maskBox]
shear_1Mpc_train = shear_1Mpc[~maskBox]
shear_1Mpc_test = shear_1Mpc[maskBox]
envth_1Mpc_train = GroupEnvTH_1_3[~maskBox]
envth_1Mpc_test = GroupEnvTH_1_3[maskBox]
envth_2Mpc_train = GroupEnvTH_2_5[~maskBox]
envth_2Mpc_test = GroupEnvTH_2_5[maskBox]
#labels
#number of galaxy counts
counts_train = count_dm[mass_mask][~maskBox]
counts_test = count_dm[mass_mask][maskBox]
#number of satellite counts
sat_counts_train = sat_count_dm[mass_mask][~maskBox]
sat_counts_test = sat_count_dm[mass_mask][maskBox]
#number of central counts
cent_counts_train = cent_count_dm[mass_mask][~maskBox]
cent_counts_test = cent_count_dm[mass_mask][maskBox]

## make arrays holding all the parameters
n_params=12
train_params = np.zeros((mass_train.shape[0],n_params), dtype = np.float64)
train_params[:,0] = mass_train
train_params[:,1] = envann_train
train_params[:,2] = envth_train
train_params[:,3] = envth_1Mpc_train
train_params[:,4] = envth_2Mpc_train
train_params[:,5] = env_train #GS
train_params[:,6] = conc_train
train_params[:,7] = shear_train
train_params[:,8] = shear_1Mpc_train
train_params[:,9] = spin_train
train_params[:,10] = vmax_train
train_params[:,11] = vdisp_train

test_params = np.zeros((mass_test.shape[0],n_params), dtype = np.float64)
test_params[:,0] = mass_test
test_params[:,1] = envann_test
test_params[:,2] = envth_test
test_params[:,3] = envth_1Mpc_test
test_params[:,4] = envth_2Mpc_test
test_params[:,5] = env_test
test_params[:,6] = conc_test
test_params[:,7] = shear_test
test_params[:,8] = shear_1Mpc_test
test_params[:,9] = spin_test
test_params[:,10] = vmax_test
test_params[:,11] = vdisp_test

## choose your paramter for training and testing
param_indeces = [0, 2]
X_test = test_params[:,param_indeces]
X_train= train_params[:,param_indeces]

y_train = counts_train
y_test = counts_test

#Classical ML
# svc = SVC()
# _ = svc.fit(X_train, y_train)

# train_score_c4 = svc.score(X_train, y_train)
# test_score_c4 = svc.score(X_test, y_test)

# print(f"Classical SVC on the training dataset: {train_score_c4:.2f}")
# print(f"Classical SVC on the test dataset:     {test_score_c4:.2f}")
 
X_test = X_test[:10,]
X_train = X_train[:10,]
y_test = y_test[:10,]
y_train = y_train[:10,]

print("X_test")
print(X_test)
print("X_train")
print(X_train)
print("y_test")
print(y_test)
print("y_train")
print(y_train)

X_test = np.resize(X_test, (len(X_test),2))
X_train = np.resize(X_train, (len(X_train),2))
y_test = np.resize(y_test, (len(y_test),2))
y_train = np.resize(y_train, (len(y_train),2))

max_x_test = np.amax(X_test)
min_x_test = np.amin(X_test)
max_x_train = np.amax(X_train)
min_x_train = np.amin(X_train)
max_y_test = np.amax(y_test)
min_y_test = np.amin(y_test)
max_y_train = np.amax(y_train)
min_y_train = np.amin(y_train)

#scale all values so they are between -1 and 1
for i in range(len(X_test)):    
    X_test[i] = -1 + 2 * ((X_test[i] - min_x_test)/(max_x_test - min_x_test))
    
for i in range(len(X_train)):
    X_train[i] = -1 + 2 * ((X_train[i] - min_x_train)/(max_x_train - min_x_train))

# for i in range(len(y_test)):
#     y_test[i] = -1 + 2 * ((y_test[i] - min_y_test)/(max_y_test - min_y_test))

# for i in range(len(y_train)):
#     y_train[i] = -1 + 2 * ((y_train[i] - min_y_train)/(max_y_train - min_y_train))
    
# number of qubits is equal to the number of features
# num_qubits = 2

# # number of steps performed during the training procedure
# tau = 100

# # regularization parameter
# C = 1000

# algorithm_globals.random_seed = 12345

# feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=1)

# qkernel = QuantumKernel(feature_map=feature_map)


# pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel, C=C, num_steps=tau)

# training
# pegasos_qsvc.fit(X_train, y_train)

# # testing
# pegasos_score = pegasos_qsvc.score(X_test, y_test)
# print(f"PegasosQSVC classification test score: {pegasos_score}")

# grid_step = 0.2
# margin = 0.2
# grid_x, grid_y = np.meshgrid(
#     np.arange(-margin, np.pi + margin, grid_step), np.arange(-margin, np.pi + margin, grid_step)
# )

# meshgrid_features = np.column_stack((grid_x.ravel(), grid_y.ravel()))
# meshgrid_colors = pegasos_qsvc.predict(meshgrid_features)

# plt.figure(figsize=(5, 5))
# meshgrid_colors = meshgrid_colors.reshape(grid_x.shape)
# plt.pcolormesh(grid_x, grid_y, meshgrid_colors, cmap="RdBu", shading="auto")

# plt.scatter(
#     X_train[:, 0][y_train == 0],
#     X_train[:, 1][y_train == 0],
#     marker="s",
#     facecolors="w",
#     edgecolors="r",
#     label="A train",
# )
# plt.scatter(
#     X_train[:, 0][y_train == 1],
#     X_train[:, 1][y_train == 1],
#     marker="o",
#     facecolors="w",
#     edgecolors="b",
#     label="B train",
# )

# plt.scatter(
#     X_test[:, 0][y_test == 0],
#     X_test[:, 1][y_test == 0],
#     marker="s",
#     facecolors="r",
#     edgecolors="r",
#     label="A test",
# )
# plt.scatter(
#     X_test[:, 0][y_test == 1],
#     X_test[:, 1][y_test == 1],
#     marker="o",
#     facecolors="b",
#     edgecolors="b",
#     label="B test",
# )

# plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
# plt.title("Pegasos Classification")
# plt.show()
