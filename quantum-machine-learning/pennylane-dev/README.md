# PennyLane

Pennylane is developed by the Canadian company Xanadu, and is a Python library for differentiable programming of quantum computers. The value proposition is to "train a quantum computer the same way as a neural network." Xanadu's other product is Strawberry Fields, which is a cross-platform Python library for simulating and executing programs on quantum photonic hardware.


# Installation

```
pip install pennylane --upgrade
```

Additional quantum frameworks and plugins from other vendors

```
pip install pennylane-qiskit pennylane-cirq pennylane-forest
```

Qiskit by IBM, Cirl by Google and Forest by Rigetti


## ML Library integration

```
pip install autograd "tensorflow>=1.13.2" jax jaxlib
conda install pytorch torchvision torchaudio -c pytorch
```
