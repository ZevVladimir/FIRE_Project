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
from scipy.optimize import curve_fit
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
from IPython.display import clear_output
import time
from qiskit_machine_learning.algorithms import VQC
import h5py
from sklearn.model_selection import train_test_split

#path to data
file_path = "../Data/Higgs_data.h5"

seed = 1234
np.random.default_rng(seed=seed)

def READ_LHC_DATA(file_path, event_count, param_indices):
    lhc_data = h5py.File(file_path,'r')
    features = np.array(lhc_data["features"][:event_count, param_indices])
    targets = np.array(lhc_data["targets"][:event_count])  # signal events =1, background events =0
    return features, targets

event_count = 500

param_indices = np.arange(21,28) #28 features NOTE: check order to figure out what indicies are what parameters

features, targets = READ_LHC_DATA(file_path, event_count, param_indices)
features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size = 0.50, random_state=seed)


X_train= features_train
X_test = features_test
print("Train shape: " + str(X_train.shape))
print(X_train)
print("Test shape: " + str(X_test.shape))

Y_train = targets_train
Y_test = targets_test

num_features = len(param_indices) 

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
#feature_map.decompose().draw(output="mpl", fold=20)

ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
#ansatz.decompose().draw(output="mpl", fold=20)

optimizer = COBYLA(maxiter=50)

#TODO replace with sampler
quantum_instance = QuantumInstance(
    AerSimulator(),
    shots=1024,
    seed_simulator=algorithm_globals.random_seed,
    seed_transpiler=algorithm_globals.random_seed,
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
TRAINED_MODEL = vqc.fit(X_train, Y_train)
#wanted to see if taking output of vqc would be any different (it looks like it ends up having slightly different scores in the end)
elapsed = time.time() - start
print(f"Training time: {round(elapsed)} seconds")

plt.title("Objective function value against iteration")
plt.xlabel("Iteration")
plt.ylabel("Objective function value")
plt.plot(range(len(objective_func_vals)), objective_func_vals)
plt.show()

#random_indices_test = np.random.randint(0,X_train.shape[0], (500))
#X_test = X_test[random_indices_test]
#Y_test = Y_test[random_indices_test]

train_score_q4 = vqc.score(X_train, Y_train)
test_score_q4 = vqc.score(X_test, Y_test)
TRAIN_SCORE = TRAINED_MODEL.score(X_train, Y_train)
TEST_SCORE = TRAINED_MODEL.score(X_test, Y_test)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")

print(f"[OUTPUT OF VQC] Quantum VQC on the training dataset: {TRAIN_SCORE:.2f}")
print(f"[OUTPUT OF VQC] Quantum VQC on the test dataset:     {TEST_SCORE:.2f}")

ypred = vqc.predict(X_test)

#graph the predictions of VQC
#x = np.linspace(np.min(Y_test),np.max(Y_test), 100)
#fig, ax = plt.subplots(1,1, figsize=(8,6))
#ax.scatter(Y_test,ypred, label = '$\mathrm{VQC}$')
#ax.plot(x, x, linestyle='-', c='r')
#ax.set_xlabel(r'$\rm{TNG300}\ N_{\rm gals} $', fontsize = 20)
#ax.set_ylabel(r'$\rm{PREDICTED}\ N_{\rm gals} $', fontsize = 20)
#ax.set_title(r'$\rm{Prediction\ results\ |\ Subbox\ TNG300}$')
#plt.legend()
#plt.show()