"""Randomized-QSVT-style ground-state preparation (Wang et al. 2025, arXiv:2510.06851).

The RQSVT ground-state algorithm filters a guess state onto the ground state with a
degree-d polynomial P(U) of the controlled time-evolution U = e^{i s H}:

    P(U)|phi_0>  ~  <v_0|phi_0> |v_0>          (degree d = O(lambda / Delta))

We realise the same filter polynomial via a Linear Combination of Unitaries
(PREPARE-SELECT-UNPREPARE) over time-evolutions:

    P(U) = sum_n c_n U^n ,   c_n = exp(-(n sigma)^2 / 2) e^{-i n eta_0}   (Gaussian filter)

This is exact and block-encoding-free (it is a block encoding of P(U) built from
controlled time-evolutions, never of H). It uses ceil(log2(2d+1)) ancilla qubits
rather than the single ancilla of the paper's GQSP construction -- a deliberate
trade: robust, high-degree GQSP phase-factor / complementary-polynomial computation
(Berntson-Sunderhauf 2024) is a specialised numerical task, whereas the LCU
coefficients here are exact Gaussian Fourier coefficients. The randomisation
(qDRIFT), the structured-Trotter evolution, and the Richardson extrapolation are
identical to the paper.

The controlled evolution U^n is realised three ways:
    'trotter' : structured Trotter step (Joshi et al.)   -- "structured Trotter + RQSVT"
    'qdrift'  : qDRIFT channel, importance-sampling H's terms (the paper's
                randomisation; depth independent of the term count)
    'exact'   : U^n = expm(i s n H) as a dense gate (small lattices; validation)

Richardson extrapolation over the Trotter/qDRIFT step count suppresses the
simulation error, as in the paper.

The guess state is the strong-coupling vacuum |0...0> (all links S^z = +1/2), which
is gauge invariant (G_j|0> = 0). The structured-Trotter evolution commutes with
every Gauss generator, so the filtered state stays in the physical sector and the
filter selects the gauge-invariant ground state even though U carries no penalty.
"""

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate, PauliEvolutionGate, StatePreparation
from qiskit.quantum_info import SparsePauliOp

from QuantumFields.gauge_field import QuantumLinkModel
from circuit import StructuredTrotter


def richardson_extrapolate(hs, ys):
    """Lagrange extrapolation of ys(hs) to h = 0."""
    total = 0.0
    for i in range(len(hs)):
        Li = 1.0
        for j in range(len(hs)):
            if j != i:
                Li *= (0.0 - hs[j]) / (hs[i] - hs[j])
        total += ys[i] * Li
    return total


class RQSVTGroundState:
    """LCU realisation of the RQSVT ground-state filter for the pure-gauge U(1) QLM."""

    def __init__(self, lattice, g=6.0, J=2.0, eta_max=2.6, resolve=2.5, max_degree=40):
        self.lattice = lattice
        self.n = lattice.n_qubits
        self.g, self.J = g, J

        model = QuantumLinkModel(lattice, g=g, J=J)
        self.H_phys = (model.build_electric_term()
                       + model.build_magnetic_term()).simplify()
        self._H_sparse = self.H_phys.to_matrix(sparse=True)

        # Reference spectrum (the paper's assumed knowledge of E0 and the gap Delta).
        H_pen = QuantumLinkModel(lattice, g=g, J=J,
                                 gauss_penalty=200.0).build_hamiltonian()
        evals, evecs = eigsh(H_pen.to_matrix(sparse=True), k=2, which='SA')
        self.E0, self.E1 = float(evals[0]), float(evals[1])
        self.gs_exact = evecs[:, 0]
        e_hi = float(eigsh(self._H_sparse, k=1, which='LA', return_eigenvectors=False)[0])
        e_lo = float(eigsh(self._H_sparse, k=1, which='SA', return_eigenvectors=False)[0])

        # Map the spectrum into eigenphases in (-pi, pi) (no aliasing): U = e^{i scale H}.
        self.scale = eta_max / max(abs(e_lo), abs(e_hi), 1e-9)
        self.eta0 = self.scale * self.E0
        eta_gap = self.scale * abs(self.E1 - self.E0)

        # Gaussian filter: width sigma < eta_gap resolves the gap; truncate at ~3 sigma.
        self.sigma = eta_gap / resolve
        self.degree = int(np.clip(np.ceil(3.0 / self.sigma), 2, max_degree))
        ns = np.arange(-self.degree, self.degree + 1)
        self.coeff_mag = np.exp(-(ns * self.sigma) ** 2 / 2.0)          # |c_n|
        self.n_anc = int(np.ceil(np.log2(2 * self.degree + 1)))

    # ---- controlled evolution U^power = e^{i scale*power*H} ----

    def _evolution_gate(self, power, backend, n_trotter, qdrift_steps, rng):
        qc = QuantumCircuit(self.n, name=f"U^{power}")
        if backend == 'exact':
            U = expm(1j * self.scale * power * self.H_phys.to_matrix())
            qc.append(UnitaryGate(U, label=f"U^{power}"), range(self.n))
        elif backend == 'trotter':
            dt = -self.scale / n_trotter                       # negative dt -> e^{+iH|dt|}
            step = StructuredTrotter(self.lattice, g=self.g, J=self.J, dt=dt).build_step()
            for _ in range(power * n_trotter):
                for inst in step.data:
                    if inst.operation.name != 'barrier':
                        qc.append(inst.operation, inst.qubits)
        elif backend == 'qdrift':
            terms = self._qdrift_terms()
            lam = sum(w for w, _ in terms)
            probs = [w / lam for w, _ in terms]
            steps = power * qdrift_steps
            tick = self.scale * power * lam / steps
            for _ in range(steps):
                _, (pauli, sign) = terms[rng.choice(len(terms), p=probs)]
                op = SparsePauliOp([pauli], coeffs=[1.0])
                qc.append(PauliEvolutionGate(op, time=-sign * tick), range(self.n))
        else:
            raise ValueError(f"unknown backend {backend!r}")
        return qc.to_gate(label=f"U^{power}")

    def _qdrift_terms(self):
        terms = []
        for pauli, coeff in zip(self.H_phys.paulis, self.H_phys.coeffs):
            w = abs(coeff.real)
            if w > 1e-12:
                terms.append((w, (pauli, float(np.sign(coeff.real)))))
        return terms

    # ---- LCU PREPARE / SELECT / UNPREPARE ----

    def _prepare_state(self):
        """Amplitude vector encoding sqrt(|c_{m-d}|/lambda) on |m>, m = 0..2d."""
        amps = np.zeros(2 ** self.n_anc, dtype=complex)
        amps[:2 * self.degree + 1] = np.sqrt(self.coeff_mag)
        return amps / np.linalg.norm(amps)

    def build_circuit(self, backend='trotter', n_trotter=2, qdrift_steps=200, seed=0):
        """LCU filter circuit: system qubits 0..n-1, ancilla register n..n+a-1."""
        rng = np.random.default_rng(seed)
        sys = list(range(self.n))
        anc = list(range(self.n, self.n + self.n_anc))
        qc = QuantumCircuit(self.n + self.n_anc)

        prep = StatePreparation(self._prepare_state())
        qc.append(prep, anc)
        # phase e^{-i m eta0} on |m> = product of single-qubit phases on the register
        for j, a in enumerate(anc):
            qc.p(-(2 ** j) * self.eta0, a)
        # SELECT: bit j controls U^{2^j}
        for j, a in enumerate(anc):
            cU = self._evolution_gate(2 ** j, backend, n_trotter,
                                      qdrift_steps, rng).control(1)
            qc.append(cU, [a] + sys)
        qc.append(prep.inverse(), anc)
        return qc

    # ---- run on Aer + read out ----

    def prepare(self, aer_backend, backend='trotter', n_trotter=2,
                qdrift_steps=200, seed=0):
        """Run on Aer, post-select ancilla=|0...0>, return prepared state + diagnostics."""
        import qiskit_aer  # noqa: F401  -- registers .save_statevector() on QuantumCircuit
        qc = self.build_circuit(backend, n_trotter, qdrift_steps, seed)
        qc.save_statevector()
        sv = np.asarray(aer_backend.run(transpile(qc, aer_backend)).result()
                        .get_statevector())

        psi = sv[:2 ** self.n]                              # ancilla=|0...0> block (ancilla = MSBs)
        success = float(np.vdot(psi, psi).real)
        if success < 1e-12:
            return {'energy': float('nan'), 'overlap': 0.0, 'success_prob': success}
        psi = psi / np.sqrt(success)
        return {'state': psi,
                'energy': float((np.conj(psi) @ (self._H_sparse @ psi)).real),
                'overlap': float(abs(np.vdot(self.gs_exact, psi)) ** 2),
                'success_prob': success}

    def estimate_energy(self, aer_backend, backend='trotter',
                        n_trotter_list=(1, 2, 3), qdrift_steps=200, seed=0):
        """Richardson-extrapolate <H> over the Trotter/qDRIFT step count."""
        hs, energies, runs = [], [], []
        for n in n_trotter_list:
            res = self.prepare(aer_backend, backend, n_trotter=n,
                               qdrift_steps=qdrift_steps * n, seed=seed)
            hs.append(1.0 / n)
            energies.append(res['energy'])
            runs.append((n, res))
        extra = richardson_extrapolate(hs, energies) if len(hs) > 1 else energies[0]
        return {'energy_richardson': extra, 'runs': runs, 'E0_exact': self.E0}

    def guess_energy(self):
        """<0...0| H |0...0> and its ground-state overlap (the unfiltered baseline)."""
        psi0 = np.zeros(2 ** self.n, dtype=complex)
        psi0[0] = 1.0
        return (float((np.conj(psi0) @ (self._H_sparse @ psi0)).real),
                float(abs(np.vdot(self.gs_exact, psi0)) ** 2))


if __name__ == "__main__":
    # Filter-coefficient sanity check (no qiskit needed).
    for sigma in (0.10, 0.05):
        ns = np.arange(-40, 41)
        c = np.exp(-(ns * sigma) ** 2 / 2)
        f = lambda d: abs(np.sum(c * np.exp(1j * ns * d)))
        print(f"sigma={sigma}: |f(0)|={f(0)/f(0):.3f}  |f(sigma)|={f(sigma)/f(0):.3f}  "
              f"|f(3sigma)|={f(3*sigma)/f(0):.3f}")
