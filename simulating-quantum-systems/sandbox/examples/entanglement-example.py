
# pip3 install matplotlib numpy scipy
import matplotlib.pyplot as plt
import numpy as np
# scipy
from scipy.linalg import eigvals

# starting from http://verga.cpt.univ-mrs.fr/pages/CH-entangle.html

# We use a python implementation to create both, product and entangled states, and to compute the von Neumann entropy of a bipartite system in Hilbert space.

# von Neumann entropy of a bipartite system
def vn_ent(rho):
    """ computes von Neumann entropy, from the eigenvalues of rho"""
    pn = np.real(eigvals( np.einsum( 'ijkj->ik', rho ) ))
    n = pn > 0
    if len(n) > 0:
        S = -np.sum( pn[n] * np.log2(pn[n]) )
    else:
        S = 0.0 # p log p = 0 if p = 0
    return S

# Next, we create a product state |++⟩ = 1/2(|00⟩+|01⟩+|10⟩+|11⟩)
# ρ0 = |++⟩⟨++|
# where |+⟩ = ( |0⟩+|1⟩ )*1/sqrt(2)

# qubit
b0 = np.array([1,0])
b1 = np.array([0,1])
# balanced state 00 + 01 + 10 + 11
b00 = np.kron(b0, b0)
b01 = np.kron(b0, b1)
b10 = np.kron(b1, b0)
b11 = np.kron(b1, b1)
bb = ( b00 + b01 + b10 + b11 )/2

# density matrix
rho0 = np.kron(bb.reshape(4,1), bb.reshape(1,4))

# We partition the hilbert space into two equal parts, corresponding to qubit 1 and qubit 2.
# The density matrix can be viewed as a four rank tensor ρij,kl = ⟨i| ⊗ ⟨j|ρ|k⟩ ⊗|l⟩.

# bipartition and partial trace over 2
rho0_12 = rho0.reshape((2,2,2,2))
# contraction of axis 1, 3 (n2) gives rho_1
rho0_1 = np.einsum('ijkj -> ik', rho0_12)

#Using the CZ (controled Z operator, CZ=diag(1,1,1,−1), we define a new state |ψ⟩=CZ|++⟩:
# |ψ⟩=1/2* (|00⟩+|01⟩+|10⟩−|11⟩)
# and the first 1 qubit density matrix: ρ1=Tr2|ψ⟩⟨ψ is the partial trace over qubit 2.

# apply cphase gate
cphase = np.diag([1,1,1,-1])
bm = np.dot(cphase, bb)
rho = np.kron(bm.reshape(4,1), bm.reshape(1,4))
rho_12 = rho.reshape((2,2,2,2))
rho_1 = np.einsum('ijkj -> ik', rho_12)

# The von Neumann entropy of a general state ρ is S = −Tr[ ρ*log(ρ) ]
vn_ent(rho0_12) # output 0
vn_ent(rho_12) # output 1

##############################
### Random density matrix  ###
##############################

# create a random state of n qubits
n = 12
# normal distributed complex amplitudes
a = np.random.normal(0, 1, [2, 2**n])
a = a[0] + 1j*a[1]
psi = a/np.linalg.norm(a) # random state
rho = np.kron( psi.reshape([len(psi),1]),\
              np.conjugate(psi.reshape([1, len(psi)])) )

# split the systems into two equal parts
n2 = 2**(n//2)
vn_ent( rho.reshape([n2,n2,n2,n2]) ) # output 5.2742
# select first spin
vn_ent( rho.reshape([2,2**(n-1),2,2**(n-1)]) ) # output 1


# These results show that one spin in a generic random state is maximally entangled with the other spins, and that one half of the system is highly entangled with the other half; in our case with 12 spins the von Neumann entropy is SA≈5.3, near the maximum value of 6.
# In 1993 Don Page conjectured a formula,

# Don Page entropy:
da = 2**(n//2)
db = 2**(n//2)
np.log2(da) - (da**2 - 1)/(2*da*db*np.log(2))  # output 5.2788


##############################
### Exact diagnoalization  ###
##############################

#https://www.tensors.net/exact-diagonalization

# The goal of this code to is to find low-energy eigenstates of a Hamiltonian H, that is composed as a sum of local couplings `h`. In order to do this efficiently, we first write a function that computes H|Ψ> for an input quantum state |Ψ>, by applying each of the `h` couplings singularly. This function can then be passed to a standard sparse eigensolver such as `eigs`.
# applying the local couplings `h` to the state is most easily accomplished by using a tensordot function. This is built-in with numpy, and we provide custom versions for MATLAB and Julia.
# the index ordering conventions for the state and the Hamiltonian are presented below.


# doApplyHam.py
# ---------------------------------------------------------------------
# Routine used in the implementation of exact diagonalization.
#
# by Glen Evenbly (c) for www.tensors.net, (v1.2) - last modified 6/2019

import numpy as np


def doApplyHam(psiIn: np.ndarray,
               hloc: np.ndarray,
               N: int,
               usePBC: bool):
  """
  Applies local Hamiltonian, given as sum of nearest neighbor terms, to
  an input quantum state.
  Args:
    psiIn: vector of length d**N describing the quantum state.
    hloc: array of ndim=4 describing the nearest neighbor coupling.
    N: the number of lattice sites.
    usePBC: sets whether to include periodic boundary term.
  Returns:
    np.ndarray: state psi after application of the Hamiltonian.
  """
  d = hloc.shape[0]
  psiOut = np.zeros(psiIn.size)
  for k in range(N - 1):
    # apply local Hamiltonian terms to sites [k,k+1]
    psiOut += np.tensordot(hloc.reshape(d**2, d**2),
                           psiIn.reshape(d**k, d**2, d**(N - 2 - k)),
                           axes=[[1], [1]]).transpose(1, 0, 2).reshape(d**N)

  if usePBC:
    # apply periodic term
    psiOut += np.tensordot(hloc.reshape(d, d, d, d),
                           psiIn.reshape(d, d**(N - 2), d),
                           axes=[[2, 3], [2, 0]]
                           ).transpose(1, 2, 0).reshape(d**N)

  return psiOut


"""
mainExactDiag.py
---------------------------------------------------------------------
Script file for initializing exact diagonalization using the 'eigsh' routine
for a 1D quantum system.

by Glen Evenbly (c) for www.tensors.net, (v1.2) - last modified 06/2020
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from timeit import default_timer as timer

from doApplyHam import doApplyHam

# Simulation parameters
model = 'XX'  # select 'XX' model of 'ising' model
Nsites = 18  # number of lattice sites
usePBC = True  # use periodic or open boundaries
numval = 1  # number of eigenstates to compute

# Define Hamiltonian (quantum XX model)
d = 2  # local dimension
sX = np.array([[0, 1.0], [1.0, 0]])
sY = np.array([[0, -1.0j], [1.0j, 0]])
sZ = np.array([[1.0, 0], [0, -1.0]])
sI = np.array([[1.0, 0], [0, 1.0]])
if model == 'XX':
  hloc = (np.real(np.kron(sX, sX) + np.kron(sY, sY))).reshape(2, 2, 2, 2)
  EnExact = -4 / np.sin(np.pi / Nsites)  # Note: only for PBC
elif model == 'ising':
  hloc = (-np.kron(sX, sX) + 0.5 * np.kron(sZ, sI) + 0.5 * np.kron(sI, sZ)
          ).reshape(2, 2, 2, 2)
  EnExact = -2 / np.sin(np.pi / (2 * Nsites))  # Note: only for PBC


# cast the Hamiltonian 'H' as a linear operator
def doApplyHamClosed(psiIn):
  return doApplyHam(psiIn, hloc, Nsites, usePBC)


H = LinearOperator((2**Nsites, 2**Nsites), matvec=doApplyHamClosed)

# do the exact diag
start_time = timer()
Energy, psi = eigsh(H, k=numval, which='SA')
diag_time = timer() - start_time

# check with exact energy
EnErr = Energy[0] - EnExact  # should equal to zero

print('NumSites: %d, Time: %1.2f, Energy: %e, EnErr: %e' % (Nsites, diag_time, Energy[0], EnErr))

# Other diagonalization examples
# https://github.com/topics/exact-diagonalization
# https://www.netket.org/tutorials/netket3.html and https://github.com/netket/netket
# Ising Model:
## https://github.com/netket/netket/blob/master/Examples/Ising2d/ising2d.py
## https://github.com/netket/netket/blob/23c69feca6ecb82fd9dd1e06510e1eb91bbc34c7/netket/operator/_hamiltonian.py#L106

# tutorials
# https://jendrzejewski.synqs.org/post/2021-plotting-in-python/
# https://personal.math.ubc.ca/~pwalls/math-python/linear-algebra/eigenvalues-eigenvectors/
