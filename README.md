# Quantum Machine Learning (QML)

Welcome to my Quantum Computing Repo 😺. Each of the individual quantum machine learning sub-folders have individual README's that include summaries and installation instructions for the respective libraries and frameworks.

This repo uses both Python and R and [supabase](https://app.supabase.io/) for data management.

# QC/QML Libraries

[Qiskit](https://qiskit.org/)
[PennyLane](https://pennylane.ai/)
[TensorFlow Quantum](https://www.tensorflow.org/quantum/concepts)
[D-Wave](https://www.dwavesys.com/)

# Repository Structure

```
├── adiabatic-theorem-with-mathematica
├── categorical-qm
│   └── toric-code
└── quantum-machine-learning
    ├── dwave-experiments
    │   ├── dwave-mis-illustration
    │   └── quantum-enabled-drug-discovery
    ├── pennylane-experiments
    │   └── helper_functions
    ├── qiskit-experiments
    └── tensorflow-quantum-experiments
```

To update readme directory structure:

```
> tree -v -L 3 --charset utf-8 -d

```

MongoDB Atlas
```
mongodb+srv://mercicle:<password>@categorical-qm-cluster.om5oy.mongodb.net/myFirstDatabase?retryWrites=true&w=majority
```

Whitelist shinyapps.io IPs:

```
54.204.34.9
54.204.36.75
54.204.37.78
34.203.76.245
3.217.214.132
34.197.152.155
```

# Articles

https://www.twilio.com/blog/environment-variables-python


# Python venv

```
python3 -m venv toric_vm
source toric_vm/bin/activate
pip3 install -r requirements.txt
deactivate
```
