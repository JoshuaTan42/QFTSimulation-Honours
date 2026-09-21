"""Thesis figures from results/summary.json (run main.py first)."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lattice import LatticeGrid
from rqsvt import RQSVTGroundState


def rqsvt_figure(summary, path="results/rqsvt_resources.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    rq = RQSVTGroundState(LatticeGrid(3, 3), J=summary['config']['J'], filter_kind="gaussian")
    w = np.linalg.eigvalsh(rq._H.toarray())
    E = np.linspace(-rq.lambda_H, rq.lambda_H, 1500)
    axes[0].plot(E, np.abs(rq.filter_response(E)), color="#1f4e79", label="Gaussian, D = %d" % rq.D)
    rs = RQSVTGroundState(LatticeGrid(3, 3), J=summary['config']['J'], filter_kind="step")
    axes[0].plot(E, np.abs(rs.filter_response(E)), color="#c0392b", label="step, D = %d" % rs.D)
    axes[0].vlines(w, 0, 0.08, color="k", lw=0.8, label="sector spectrum (21 levels)")
    axes[0].set(xlabel="energy E", ylabel="|F(E)|", title="(a) filters realised by the GQSP angles")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    res = summary['rqsvt']['gaussian']
    per_signal = res['resources']['gqsp_trotter']['two_qubit_gates']
    E0 = res['summary']['E0']
    n = [r['n'] for r in res['trotter']['runs']]
    axes[1].loglog([per_signal * k for k in n],
                   [abs(r['energy'] - E0) for r in res['trotter']['runs']],
                   'o-', color="#1f4e79", label="GQSP + structured Trotter (n = 1, 2, 4)")
    axes[1].loglog([per_signal * sum(n)], [abs(res['trotter']['error'])], '*', ms=13,
                   color="#1f4e79", label="  Richardson (all three circuits)")
    qd = res['qdrift']['runs']
    per_sample = res['resources']['gqsp_qdrift']['two_qubit_gates'] / qd[-1]['n']
    axes[1].loglog([per_sample * r['n'] for r in qd], [abs(r['energy'] - E0) for r in qd],
                   's-', color="#ca6f1e", label="GQSP + qDRIFT (N = 200, 400, 800)")
    axes[1].loglog([per_sample * sum(r['n'] for r in qd)], [abs(res['qdrift']['error'])], '*',
                   ms=13, color="#ca6f1e", label="  Richardson")
    be = res['resources']['block_encoded_qsvt']
    axes[1].axvline(be['two_qubit_gates'], color="k", ls="--",
                    label="block-encoded QSVT, 22 qubits (estimate)")
    axes[1].axvline(be['two_qubit_gates_unary_estimate'], color="0.5", ls=":",
                    label="  with unary iteration, 28 qubits")
    axes[1].set(xlabel="two-qubit gates", ylabel="|E - E0|",
                title="(b) 3x3 cylinder, 16 qubits: accuracy against circuit size")
    axes[1].legend(fontsize=7.5)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def trotter_figure(summary, path="results/trotter_resources.png"):
    rows = summary['trotter']['resources']
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    nq = [r['qubits'] for r in rows]
    ax.plot(nq, [r['structured']['two_qubit_depth'] for r in rows], 'o', color="#1f4e79",
            label="structured, compact plaquette gate")
    ax.plot(nq, [r['structured_pauli']['two_qubit_depth'] for r in rows], 's', color="#7b2d8e",
            label="structured, Pauli ladders")
    gen = [(r['qubits'], r['generic']['two_qubit_depth']) for r in rows if 'generic' in r]
    ax.plot(*zip(*gen), '^', color="#c0392b", label="generic first-order Trotter")
    ax.set(xlabel="link qubits $N_q$", ylabel="two-qubit depth per step",
           title="Two-qubit depth of one Trotter step")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    with open("results/summary.json") as fh:
        summary = json.load(fh)
    rqsvt_figure(summary)
    trotter_figure(summary)
    print("saved results/rqsvt_resources.png, results/trotter_resources.png")
