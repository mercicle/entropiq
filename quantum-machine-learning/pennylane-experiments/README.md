# PennyLane

Pennylane is developed by the Canadian company Xanadu, and is a Python library for differentiable programming of quantum computers. The value proposition is to "train a quantum computer the same way as a neural network." Xanadu's other product is Strawberry Fields, which is a cross-platform Python library for simulating and executing programs on quantum photonic hardware.


# Installation

```
pip3 install pennylane --upgrade
```

Additional quantum frameworks and plugins from other vendors

```
pip3 install pennylane-qiskit pennylane-cirq pennylane-forest
```

Qiskit by IBM, Cirl by Google and Forest by Rigetti


## ML Library integration

```
pip3 install autograd "tensorflow>=1.13.2" jax jaxlib
conda install pytorch torchvision torchaudio -c pytorch
```


# Examples

[Gated Graph Recurrent Neural Networks](https://arxiv.org/abs/2002.01038)

> Graph processes exhibit a temporal structure determined by the sequence index and and a spatial structure determined by the graph support. To learn from graph processes, an information processing architecture must then be able to exploit both underlying structures. We introduce Graph Recurrent Neural Networks (GRNNs) as a general learning framework that achieves this goal by leveraging the notion of a recurrent hidden state together with graph signal processing (GSP). In the GRNN, the number of learnable parameters is independent of the length of the sequence and of the size of the graph, guaranteeing scalability. We prove that GRNNs are permutation equivariant and that they are stable to perturbations of the underlying graph support. To address the problem of vanishing gradients, we also put forward gated GRNNs with three different gating mechanisms: time, node and edge gates. In numerical experiments involving both synthetic and real datasets, time-gated GRNNs are shown to improve upon GRNNs in problems with long term dependencies, while node and edge gates help encode long range dependencies present in the graph. The numerical results also show that GRNNs outperform GNNs and RNNs, highlighting the importance of taking both the temporal and graph structures of a graph process into account.

[Quantum Graph Neural Networks Tutorial](https://pennylane.ai/qml/demos/tutorial_qgrnn.html)

[Quantum Graph Neural Networks Paper](https://arxiv.org/abs/1909.12264)

> We introduce Quantum Graph Neural Networks (QGNN), a new class of quantum neural network ansatze which are tailored to represent quantum processes which have a  > graph structure, and are particularly suitable to be executed on distributed quantum systems over a quantum network. Along with this general class of ansatze, we > introduce further specialized architectures, namely, Quantum Graph Recurrent Neural Networks (QGRNN) and Quantum Graph Convolutional Neural Networks (QGCNN). We > provide four example applications of QGNNs: learning Hamiltonian dynamics of quantum systems, learning how to create multipartite entanglement in a quantum network, unsupervised learning for spectral clustering, and supervised learning for graph isomorphism classification.

[Quantum transfer learning tutorial](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html)

[Transfer learning in hybrid classical-quantum neural networks](https://arxiv.org/abs/1912.08278)

> We extend the concept of transfer learning, widely applied in modern machine learning algorithms, to the emerging context of hybrid neural networks composed of > classical and quantum elements. We propose different implementations of hybrid transfer learning, but we focus mainly on the paradigm in which a pre-trained > classical network is modified and augmented by a final variational quantum circuit. This approach is particularly attractive in the current era of intermediate-> scale quantum technology since it allows to optimally pre-process high dimensional data (e.g., images) with any state-of-the-art classical network and to embed a > select set of highly informative features into a quantum processor. We present several proof-of-concept examples of the convenient application of quantum transfer > learning for image recognition and quantum state classification. We use the cross-platform software library PennyLane to experimentally test a high-resolution > image classifier with two different quantum computers, respectively provided by IBM and Rigetti.
