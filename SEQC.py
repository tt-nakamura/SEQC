# solving Schroedinger Equation by Quantum Computer
# referece: M. A. Nielsen and I.L. Chuang
#   "Quantum Computation and Quantum Information" \S4.7

import numpy as np
from qiskit import QuantumCircuit
from QFT import QFT

def initialize(psi):
    """ set initial condition in quantum circuit
    psi = initial wave function (array of complex numbers)
    psi[j] = coefficient of basis state |j> (j=0...N-1)
    psi need not be normalized to 1 = \sum |psi[j]|^2
    N = 2^n = len(psi) must be power of two
    where n = number of qubit
    create qc = QuantumCircuit object and
    compose quantum gates in qc to initilize it by psi
    and return qc
    """
    N = len(psi)
    if N&(N-1): raise RuntimeError('N must be power of 2')
    n = N.bit_length() - 1
    qc = QuantumCircuit(n)
    q = qc.qubits
    y,p = np.abs(psi)**2, np.angle(psi)
    s = [np.reshape(y,(-1,2))]
    t = [np.diff(p[::1<<(n-1)])]
    for i in range(n-1):
        k = 1<<(n-1-i)
        s.append(np.sum(s[-1],1).reshape(-1,2))
        t.append(p[k>>1::k] - p[::k])

    s.reverse()
    qc.h(range(n))
    qc.global_phase = p[0]
    for i,s in enumerate(s):
        l = n-1-i
        for j,s in enumerate(np.sqrt(s)):
            a = 2*np.arctan2(s[1],s[0]) - np.pi/2
            if i:
                m = j^(j-1)
                qc.x([l+k+1 for k in range(i) if m&(1<<k)])
                qc.mcry(a, q[l+1:], q[l])
                qc.mcp(t[i][j], q[l+1:], q[l])
            else:
                qc.ry(a,l)
                qc.p(t[i][j], l)

    return qc

def potential_energy(qc, V, dt):
    """ multiply phase factor exp(-iV(x_j)dt) to each
    basis state |j> (j=0...N-1) in configuation space
    qc = QuantumCircuit object (created by qiskit)
    V = numpy array of real numbers V(x_j)
    dt = small time interval
    len(V) must be 2**(number of qubits)
    """
    q,n = qc.qubits, qc.num_qubits
    for i,t in enumerate(-V*dt):# exponentially slow
        m = i^(i-1)
        qc.x([k for k in range(n) if m&(1<<k)])
        qc.mcp(t, q[1:], q[0])


def kinetic_energy(qc, dt, L=1):
    """ multipty phase factor xp(-i*dt/2*p_k**2) to
    each basis state |k> (k=0...N-1) in momentum space
    where p_k = 2*pi*k/L for k=0...N/2-1 and
          p_k = 2*pi*(k-N)/L for k=N/2...N-1
    dt = small time interval
    L = length of periodicity in configuration space
    """
    n,a = qc.num_qubits, -dt*2*(np.pi/L)**2
    for i in range(n):
        qc.p(a*(1<<(i<<1)), i)
        if i+1 == n: a = -a # negative frequency part
        for j in range(i):
            qc.cp(a*(1<<(i+j+1)), i, j)

def evolve(qc, dt, V=None, repeat=1, L=1):
    """ quantum simulation
    apply exp(-i*H*dt) to wave funciton in qc
    where H = -(1/2)(d/dx)^2 + V is Hamiltonian
    using Trotter formula
    qc = QuantumCircuit object (created by qiskit)
    dt = time step of simulation
    V = potential energy (numpy array)
    len(V) must be 2^n where n = number of qubits
    if V is None, V=0 (free particle) is assumed
    repeat = number of repetition to advance dt
    L = length of periodicity in configuration space
    """
    for _ in range(repeat):
        if V is not None: potential_energy(qc, V, dt)
        QFT(qc)
        kinetic_energy(qc, dt, L)
        QFT(qc, inv=True)
