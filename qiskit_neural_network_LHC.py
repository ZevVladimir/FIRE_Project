import numpy as np
import matplotlib.pyplot as plt

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.visualization import *

#These imports were used within the IBM environment and don't work here but aren't necessary
#from qiskit.tools.jupyter import *
#from ibm_quantum_widgets import *
#from qiskit.providers.aer import QasmSimulator

import numpy as np
from scipy import stats
#from scipy.optimize import curve_fit
from scipy import stats
from scipy import special
from scipy import interpolate
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit.algorithms.optimizers import COBYLA
from qiskit_aer import AerSimulator
from qiskit.utils import QuantumInstance
from qiskit.utils import algorithm_globals
from matplotlib import pyplot as plt
#from IPython.display import clear_output
import time
from qiskit_machine_learning.algorithms import VQC
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import os

#path to data
file_path = "../Data/Higgs_data.h5"



def READ_LHC_DATA(file_path, event_count, param_indices):
    lhc_data = h5py.File(file_path,'r')
    features = np.array(lhc_data["features"][:event_count, param_indices])
    targets = np.array(lhc_data["targets"][:event_count])  # signal events =1, background events =0
    return features, targets

#currently changeable parameters
seed = 1234
np.random.default_rng(seed=seed)
event_count = 1000
param_indices = np.arange(4) #28 features NOTE: check order to figure out what indicies are what parameters
#order is likely: lepton pT, lepton eta, lepton phi, missing energy magnitude, 
#missing energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt,
#jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, 
# jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag
max_iterations = 10

features, targets = READ_LHC_DATA(file_path, event_count, param_indices)
features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size = 0.50, random_state=seed)

X_train= features_train
X_test = features_test
print("Train shape: " + str(X_train.shape))
print(X_train)
print("Test shape: " + str(X_test.shape))

Y_train = targets_train.flatten()
Y_test = targets_test.flatten()

num_features = len(param_indices) 

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
#feature_map.decompose().draw(output="mpl", fold=20)

ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
#ansatz.decompose().draw(output="mpl", fold=20)

optimizer = COBYLA(maxiter=max_iterations)

feature_map_type = str(type(feature_map))
ansatz_type = str(type(ansatz))
optimizer_type = str(type(ansatz))


#TODO replace with sampler
quantum_instance = QuantumInstance(
    AerSimulator(),
    shots=1024,
    seed_simulator=seed,
    seed_transpiler=seed,
)

#plt.rcParams["figure.figsize"] = (12, 6)

def callback(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)
    i = len(objective_func_vals)
    training_times.append(time.time())
    elapsed = time.time() - training_times[i-1]
    print(f"Iteration {i} Training time: {round(elapsed,2)} seconds")
    #moved plotting to end so its not called every iteration and added iteration step counter
    

#choose 500 random indices to use to train the model
#random_indices_train = np.random.randint(0,X_train.shape[0], (500))
#X_train = X_train[random_indices_train]
#Y_train = Y_train[random_indices_train]

#create the VQC model 
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    loss='cross_entropy_sigmoid',
    optimizer=optimizer,
    quantum_instance=quantum_instance,
    callback=callback,
)

# clear objective value history
objective_func_vals = []
training_times = []

#train the model
start = time.time()
training_times.append(start)
print("Starting")
vqc.fit(X_train, Y_train)
#wanted to see if taking output of vqc would be any different (it looks like it ends up having slightly different scores in the end)
elapsed = time.time() - start
print(f"Training time: {round(elapsed)} seconds")

plt.figure(1)
plt.title("Objective function value against iteration")
plt.xlabel("Iteration")
plt.ylabel("Objective function value (Loss)")
plt.plot(range(len(objective_func_vals)), objective_func_vals)
plt.show()
plt.close()

#random_indices_test = np.random.randint(0,X_train.shape[0], (500))
#X_test = X_test[random_indices_test]
#Y_test = Y_test[random_indices_test]

train_score_q4 = vqc.score(X_train, Y_train)
test_score_q4 = vqc.score(X_test, Y_test)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")

#only 1 or 0 to indicate signal or background respectively
ypred = vqc.predict(X_test)

labels = ['Background', 'Signal']
colors = ['black', 'lime']
plt.figure(2) 
plt.subplot(211) 
#masking test values where prediction is indicated signal/background
signal = np.bool_(ypred)

n, bins, patches = plt.hist(X_test[signal][:,0],60, histtype='step', color=colors[0], label=labels[0]) # 30 patches (bins) for 20k events/rows, plot first feature, 0, which is lepton pT
n, bins, patches = plt.hist(X_test[~signal][:,0],60, histtype='step', color=colors[1], label=labels[1]) # same for signal events
plt.title('Discriminator')
plt.legend()
plt.subplot(212) 
n, bins, patches = plt.hist(X_test[signal][:,0],30, histtype='step', color=colors[0], label=labels[0], range=(0.97, 1.0), log=True)
n, bins, patches = plt.hist(X_test[~signal][:,0],30, histtype='step', color=colors[1], label=labels[1], range=(0.97, 1.0), log=True)
plt.legend()
plt.show()
plt.close()

fpr, tpr, _ = roc_curve(Y_test, ypred)

auc_roc = auc(fpr, tpr)

plt.figure(3,figsize=(6,6))
plt.plot(tpr, 1.0-fpr, lw=3, alpha=0.8,
        label="Shallow (AUC={:.3f})".format(auc_roc))
plt.xlabel("Signal efficiency")
plt.ylabel("Background rejection")
plt.legend(loc=3)
plt.xlim((0.0, 1.0))
plt.ylim((0.0, 1.0))
plt.show()
plt.close()
