import numpy as np
import matplotlib.pyplot as plt
from qiskit import Aer,QuantumCircuit
from SEQC import initialize,evolve

def get_statevec(qc, simulator):
    job = simulator.run(qc)
    result = job.result()
    return result.data()['statevector']

n = 7 # number of qubits
L = 1 # length of periodicity
dt = 0.0001 # time step of simulation
M = 8 # number of time steps per plotting
M1 = 4 # number of plottings
x0 = 0.25 # center of initial wave packet in [0,L]
s = 0.05 # width of initial wave packet
k = 128 # initial wave momentum
V0 = 8192 # height (or depth) of potential box
x1 = 0.6 # left of potential box in [0,L]
x2 = 0.8 # right of potential box in [0,L]

N = 1<<n # number of intervals in [0,L]
x = np.arange(N)*L/N # coordinates in [0,L]
y = np.exp(-((x-x0)/s)**2/2 + 1j*k*x) # initial wave func
V = V0*((x>=x1) & (x<=x2)) # potential energy (box shape)

plt.figure(figsize=(5, 3.75))
simulator = Aer.get_backend('statevector_simulator')
qc = initialize(y)
psi = get_statevec(qc, simulator)
plt.plot(x, np.abs(psi)**2, label='t=0')
for i in range(M1):
    evolve(qc, dt, V, M, L)
    psi = get_statevec(qc, simulator)
    plt.plot(x, np.abs(psi)**2, label='t=%g'%((i+1)*M*dt))
    qc = QuantumCircuit(n)
    qc.set_statevector(psi)
    
plt.legend()
plt.xlabel(r'$x$')
plt.ylabel(r'$|\psi(x)|^2$')
plt.tight_layout()
plt.savefig('fig1.eps')
