#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer, QuantumCircuit
from qiskit.opflow import Z, I, StateFn
from qiskit.utils import QuantumInstance
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B

from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.exceptions import QiskitMachineLearningError

from typing import Union


quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024)


# ## Regression

num_samples = 20
eps = 0.2
lb, ub = -np.pi, np.pi
X_ = np.linspace(lb, ub, num=50).reshape(50, 1)
f = lambda x: np.sin(x)

X = (ub - lb)*np.random.rand(num_samples, 1) + lb
y = f(X[:,0]) + eps*(2*np.random.rand(num_samples)-1)

plt.plot(X_, f(X_), 'r--')
plt.plot(X, y, 'bo')
plt.show()


# ### Regression with an `OpflowQNN`
# 
# Here we restrict to regression with an `OpflowQNN` that returns values in $[-1, +1]$. More complex and also multi-dimensional models could be constructed, also based on `CircuitQNN` but that exceeds the scope of this tutorial.

# In[19]:


# construct simple feature map
param_x = Parameter('x')
feature_map = QuantumCircuit(1, name='fm')
feature_map.ry(param_x, 0)

# construct simple ansatz
param_y = Parameter('y')
ansatz = QuantumCircuit(1, name='vf')
ansatz.ry(param_y, 0)

# construct QNN
regression_opflow_qnn = TwoLayerQNN(1, feature_map, ansatz, quantum_instance=quantum_instance)


# In[20]:


# construct the regressor from the neural network
regressor = NeuralNetworkRegressor(neural_network=regression_opflow_qnn, 
                                   loss='l2', 
                                   optimizer=L_BFGS_B())


# In[21]:


# fit to data
regressor.fit(X, y)

# score the result
regressor.score(X, y)


# In[22]:


# plot target function
plt.plot(X_, f(X_), 'r--')

# plot data
plt.plot(X, y, 'bo')

# plot fitted line
y_ = regressor.predict(X_)
plt.plot(X_, y_, 'g-')
plt.show()


# ### Regression with the Variational Quantum Regressor (`VQR`)
# 
# Similar to the `VQC` for classification, the `VQR` is a special variant of the `NeuralNetworkRegressor` with a `OpflowQNN`. By default it considers the `L2Loss` function to minimize the mean squared error between predictions and targets.

# In[23]:


vqr = VQR(feature_map=feature_map, 
          ansatz=ansatz, 
          optimizer=L_BFGS_B(), 
          quantum_instance=quantum_instance)


# In[24]:


# fit regressor
vqr.fit(X, y)

# score result
vqr.score(X, y)


# In[25]:


# plot target function
plt.plot(X_, f(X_), 'r--')

# plot data
plt.plot(X, y, 'bo')

# plot fitted line
y_ = vqr.predict(X_)
plt.plot(X_, y_, 'g-')
plt.show()


# In[26]:


import qiskit.tools.jupyter
get_ipython().run_line_magic('qiskit_version_table', '')
get_ipython().run_line_magic('qiskit_copyright', '')




