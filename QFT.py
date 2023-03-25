from math import pi

def QFT(qc, inv=False):
    """ Quantum Fourier Transform
    qc = Quantum circuit object (created by qiskit)
    compose quantum gates in qc to perform QFT
    if inv is True, inverse Fourier Transform is performed
    reference: M. A. Nielsen and I. L. Chunag
      "Quantum Computation and Quantum Information" \S5.1
    """
    n,a = qc.num_qubits, pi/2
    if inv: a = -a
    for i in range(n-1,-1,-1):
        qc.h(i)
        for j in range(i):
            qc.cp(a/(1<<j), i-j-1, i)
    # bit reversal
    for i in range(n>>1): qc.swap(i, n-1-i)
