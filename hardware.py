"""Experiments for quantum hardware: the plate quench and a short ground-state filter.

Two experiments are sized for current devices (see the resource tables in the
thesis for why the full 3x3 filter is not):

  quench   the most-flippable product state on a W x H cylinder evolved by n
           structured Trotter steps (n = 0, 1, 2, ...), measured in the
           computational basis. Gives the return probability, flippability
           and electric field per plaquette or link against time.
  filter   the single-ancilla GQSP filter at low Fourier degree on a small
           lattice (2x2 by default), ancilla and links measured.

Measurement. Every observable in this model is either diagonal in the link
basis (S^z, flippability, Gauss's law) or a plaquette term U_p + U_p^dag. The
latter is measured with the same fan-out used by the plaquette gate:
    U_p + U_p^dag = F^dag (X_b (x) |011><011|_{r,t,l}) F,  F = CX(b,r) CX(b,t) CX(b,l),
so after F and a Hadamard on b, <U_p + U_p^dag> = E[(-1)^b 1{(r,t,l) = (0,1,1)}].
Plaquettes of one colour share no link and are measured together, so the
whole magnetic energy needs one extra circuit per colour (2, or 3 for odd W).

Error detection. A computational-basis shot that violates Gauss's law (or the
plate charges of the initial state) cannot come from the ideal circuit and is
discarded. Results are reported with and without this post-selection.

Backends
  ideal            noiseless sampling (Qiskit Aer)
  fake:<name>      Aer with the noise model and coupling map of a snapshot of
                   an IBM device, e.g. fake:kingston (Heron r2, 156 qubits)
  ibm:<name>       an IBM Quantum device through Qiskit Runtime (SamplerV2);
                   needs a saved account (QiskitRuntimeService.save_account)

    python hardware.py quench --backend fake:kingston --lattice 3x3 --steps 0 1 2 3 4 --dt 0.25
    python hardware.py filter --backend ideal --lattice 2x2 --degree 2
    python hardware.py quench --backend ibm:ibm_kingston --dry-run     # resources only
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from qiskit import QuantumCircuit, transpile

from circuit import StructuredTrotter, two_qubit_count, two_qubit_depth
from config import environment
from dynamics import SectorTrotter
from QuantumFields.gauge_field import default_plates
from lattice import LatticeGrid
from sector import GaugeSector

PIVOT_PATTERN = (0, 1, 1)        # (r, t, l) after the fan-out for a flippable plaquette


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def get_backend(spec: str):
    """Backend object for "ideal", "fake:<name>" or "ibm:<name>"."""
    if spec == "ideal":
        from qiskit_aer import AerSimulator
        return AerSimulator(method="statevector")
    kind, _, name = spec.partition(":")
    if kind == "fake":
        from qiskit_ibm_runtime import fake_provider
        cls = getattr(fake_provider, "Fake" + name.replace("ibm_", "").capitalize())
        return cls()
    if kind == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService
        return QiskitRuntimeService().backend(name)
    raise ValueError(f"unknown backend {spec!r} (use ideal, fake:<name> or ibm:<name>)")


def sample(circuits, backend, shots: int, spec: str) -> list:
    """Counts dictionaries (bit strings in Qiskit order) for transpiled circuits."""
    if spec == "ideal" or spec.startswith("fake:"):
        from qiskit_aer import AerSimulator
        sim = backend if spec == "ideal" else AerSimulator.from_backend(backend)
        result = sim.run(circuits, shots=shots, seed_simulator=2026).result()
        return [result.get_counts(k) for k in range(len(circuits))]
    from qiskit_ibm_runtime import SamplerV2
    job = SamplerV2(mode=backend).run(circuits, shots=shots)
    print(f"  submitted job {job.job_id()}; waiting for the result ...")
    return [pub.data.meas.get_counts() for pub in job.result()]


def isa_circuits(circuits, backend, spec: str, optimization_level: int = 3):
    """Transpile for the backend (ideal: to {cx, rz, sx, x} with all-to-all coupling)."""
    if spec == "ideal":
        return transpile(circuits, basis_gates=['cx', 'rz', 'sx', 'x'],
                         optimization_level=optimization_level, seed_transpiler=7)
    return transpile(circuits, backend=backend, optimization_level=optimization_level,
                     seed_transpiler=7)


def estimated_fidelity(isa, backend) -> float:
    """Product of (1 - error) over all gates and measurements, from the backend target."""
    target = getattr(backend, "target", None)
    if target is None:
        return float('nan')
    fid = 1.0
    for inst in isa.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        qargs = tuple(isa.find_bit(q).index for q in inst.qubits)
        try:
            props = target[name][qargs]
        except KeyError:
            continue
        if props is not None and props.error is not None:
            fid *= 1.0 - props.error
    return float(fid)


def resource_report(isa_list, backend, labels) -> list:
    rows = []
    for qc, label in zip(isa_list, labels):
        rows.append({'circuit': label, 'qubits_used': len({isa_q for inst in qc.data
                                                          for isa_q in inst.qubits}),
                     'two_qubit_gates': two_qubit_count(qc),
                     'two_qubit_depth': two_qubit_depth(qc),
                     'depth': qc.depth(),
                     'estimated_fidelity': estimated_fidelity(qc, backend)})
    return rows


# ---------------------------------------------------------------------------
# Measurement settings and estimators
# ---------------------------------------------------------------------------

def append_plaquette_basis(qc: QuantumCircuit, plaquettes):
    """Rotate disjoint plaquettes so that <U_p + U_p^dag> is read from Z outcomes."""
    for (b, r, t, l) in plaquettes:
        for x in (r, t, l):
            qc.cx(b, x)
        qc.h(b)


def bits_of(key: str, n: int) -> np.ndarray:
    """Qiskit count key -> array of bits indexed by qubit (key is little-endian)."""
    key = key.replace(" ", "")
    return np.array([int(key[-1 - q]) for q in range(n)], dtype=np.int8)


def counts_to_arrays(counts: dict, n: int):
    keys = list(counts)
    bits = np.array([bits_of(k, n) for k in keys]) if keys else np.zeros((0, n), np.int8)
    weights = np.array([counts[k] for k in keys], dtype=float)
    return bits, weights


def labels_of(bits: np.ndarray, qubits) -> np.ndarray:
    return sum(bits[:, q].astype(np.int64) << k for k, q in enumerate(qubits))


def mean_and_error(values, weights):
    total = weights.sum()
    if total == 0:
        return float('nan'), float('nan')
    mean = float(values @ weights / total)
    var = float(((values - mean) ** 2) @ weights / total)
    return mean, float(np.sqrt(var / total))


def z_basis_observables(bits, weights, lattice, allowed_labels, initial_bits=None) -> dict:
    """Diagonal observables from computational-basis shots, raw and Gauss-selected."""
    n = lattice.n_qubits
    links = bits[:, :n]
    labels = labels_of(links, range(n))
    physical = np.isin(labels, allowed_labels)
    out = {'shots': float(weights.sum()), 'physical_fraction': float(weights[physical].sum()
                                                                     / max(weights.sum(), 1))}
    for tag, sel in (("raw", np.ones(len(weights), bool)), ("gauss", physical)):
        w = weights[sel]
        lb = links[sel]
        res = {}
        sz = 0.5 - lb
        res['electric_field'] = [mean_and_error(sz[:, q], w)[0] for q in range(n)]
        flips = []
        for (b, r, t, l) in lattice.get_plaquettes():
            f = (((lb[:, b] == 1) & (lb[:, r] == 1) & (lb[:, t] == 0) & (lb[:, l] == 0)) |
                 ((lb[:, b] == 0) & (lb[:, r] == 0) & (lb[:, t] == 1) & (lb[:, l] == 1)))
            flips.append(f.astype(float))
        flips = np.array(flips).T if flips else np.zeros((len(w), 0))
        res['flippability'] = [mean_and_error(flips[:, p], w)[0] for p in range(flips.shape[1])]
        res['mean_flippability'] = mean_and_error(flips.mean(axis=1), w) if len(w) else (np.nan,) * 2
        if initial_bits is not None:
            hit = (labels[sel] == initial_bits).astype(float)
            res['return_probability'] = mean_and_error(hit, w)
        out[tag] = res
    return out


def plaquette_basis_observable(bits, weights, plaquettes, plaq_indices) -> dict:
    """<U_p + U_p^dag> for the plaquettes rotated by append_plaquette_basis."""
    out = {}
    for p, (b, r, t, l) in zip(plaq_indices, plaquettes):
        sel = (bits[:, r] == PIVOT_PATTERN[0]) & (bits[:, t] == PIVOT_PATTERN[1]) \
            & (bits[:, l] == PIVOT_PATTERN[2])
        values = np.where(sel, 1.0 - 2.0 * bits[:, b], 0.0)
        out[p] = mean_and_error(values, weights)
    return out


# ---------------------------------------------------------------------------
# Quench experiment
# ---------------------------------------------------------------------------

def quench_experiment(W=3, H=3, J=2.0, g=0.0, rk=0.0, field_pattern="uniform", plates=None,
                      dt=0.25, steps=(0, 1, 2, 3, 4), order=1, measure_energy=True,
                      backend_spec="ideal", shots=4000, dry_run=False, verbose=True) -> dict:
    lattice = LatticeGrid(W, H)
    plates = plates or default_plates(W)
    sector = GaugeSector(lattice, plates=plates)
    gs = sector.ground_state(g=g, J=J, rk=rk, field_pattern=field_pattern)
    idx = gs['support']
    Hs = gs['hamiltonian'][idx][:, idx]
    k0 = sector.most_flippable(idx, Hs)
    init = sector.label(int(idx[k0]))
    allowed = sector.basis[idx]
    trotter = StructuredTrotter(lattice, g=g, J=J, rk=rk, dt=dt, field_pattern=field_pattern,
                                order=order)
    colours = lattice.plaquette_sublayer_indices()
    plaqs = lattice.get_plaquettes()

    circuits, meta = [], []
    for n in steps:
        base = QuantumCircuit(lattice.n_qubits)
        for q in range(lattice.n_qubits):
            if (init >> q) & 1:
                base.x(q)
        base.compose(trotter.build_evolution(n), inplace=True)
        qz = base.copy()
        qz.measure_all()
        circuits.append(qz)
        meta.append({'steps': n, 'setting': 'z'})
        if measure_energy and J != 0.0:
            for c, members in enumerate(colours):
                qp = base.copy()
                append_plaquette_basis(qp, [plaqs[p] for p in members])
                qp.measure_all()
                circuits.append(qp)
                meta.append({'steps': n, 'setting': f'plaquette colour {c}', 'plaquettes': members})

    # exact reference: the same Trotter circuit emulated in the sector, and exact evolution
    emu = SectorTrotter.for_gauge(sector, g=g, J=J, rk=rk, field_pattern=field_pattern,
                                  dt=dt, order=order, support=idx)
    psi0 = np.zeros(len(idx), complex)
    psi0[k0] = 1.0
    reference = []
    for n in steps:
        psi = emu.step(psi0, n)
        full = np.zeros(sector.dim, complex)
        full[idx] = psi
        reference.append({'steps': n, 'time': n * dt,
                          'return_probability': float(abs(psi[k0]) ** 2),
                          'mean_flippability': float(sector.flippability(full).mean()),
                          'magnetic_energy': float(J * sector.plaquette_expectations(full).sum()),
                          'energy': float(np.vdot(psi, Hs @ psi).real)})

    backend = get_backend(backend_spec)
    t0 = time.time()
    isa = isa_circuits(circuits, backend, backend_spec)
    report = resource_report(isa, backend, [f"n={m['steps']} {m['setting']}" for m in meta])
    out = {'environment': environment(), 'experiment': 'quench', 'backend': backend_spec, 'lattice': f"{W}x{H}",
           'qubits': lattice.n_qubits, 'J': J, 'g': g, 'rk': rk, 'plates': plates,
           'dt': dt, 'order': order, 'shots': shots, 'initial_bits': init,
           'block_dim': len(idx), 'resources': report, 'reference': reference,
           'transpile_seconds': time.time() - t0}
    if verbose:
        for r in report:
            print(f"  {r['circuit']:<26s} 2q gates {r['two_qubit_gates']:5d}  2q depth "
                  f"{r['two_qubit_depth']:4d}  est. fidelity {r['estimated_fidelity']:.3f}")
    if dry_run:
        return out

    counts = sample(isa, backend, shots, backend_spec)
    results = {}
    for m, cnt in zip(meta, counts):
        bits, weights = counts_to_arrays(cnt, lattice.n_qubits)
        row = results.setdefault(m['steps'], {'steps': m['steps'], 'time': m['steps'] * dt,
                                              'magnetic': {}})
        if m['setting'] == 'z':
            row.update(z_basis_observables(bits, weights, lattice, allowed, init))
        else:
            members = m['plaquettes']
            row['magnetic'].update(plaquette_basis_observable(
                bits, weights, [plaqs[p] for p in members], members))
    for row in results.values():
        if row['magnetic']:
            vals = np.array([row['magnetic'][p][0] for p in sorted(row['magnetic'])])
            errs = np.array([row['magnetic'][p][1] for p in sorted(row['magnetic'])])
            row['magnetic_energy'] = (float(J * vals.sum()), float(abs(J) * np.sqrt((errs ** 2).sum())))
            row['magnetic'] = {str(k): v for k, v in row['magnetic'].items()}
    out['measured'] = [results[n] for n in steps]
    if verbose:
        print(f"  {'n':>3} {'t':>5} | {'P_return raw':>13} {'gauss':>8} {'ideal':>7} | "
              f"{'flip raw':>9} {'gauss':>7} {'ideal':>7} | {'E_B meas':>9} {'ideal':>8} | "
              f"{'phys frac':>9}")
        for row, ref in zip(out['measured'], reference):
            eb = row.get('magnetic_energy', (float('nan'),))[0]
            print(f"  {row['steps']:3d} {row['time']:5.2f} | {row['raw']['return_probability'][0]:13.3f} "
                  f"{row['gauss']['return_probability'][0]:8.3f} {ref['return_probability']:7.3f} | "
                  f"{row['raw']['mean_flippability'][0]:9.3f} {row['gauss']['mean_flippability'][0]:7.3f} "
                  f"{ref['mean_flippability']:7.3f} | {eb:+9.3f} {ref['magnetic_energy']:+8.3f} | "
                  f"{row['physical_fraction']:9.3f}")
        print("  (ideal = the same Trotter circuit without noise, emulated in the gauge sector)")
    return out


# ---------------------------------------------------------------------------
# Filter experiment
# ---------------------------------------------------------------------------

def filter_experiment(W=2, H=2, J=2.0, degree=2, n_steps=1, order=1, plates=None,
                      backend_spec="ideal", shots=8000, dry_run=False, verbose=True) -> dict:
    """Low-degree GQSP filter: success probability, energy, Gauss-selected distribution."""
    from rqsvt import RQSVTGroundState
    lattice = LatticeGrid(W, H)
    plates = plates or default_plates(W)
    rq = RQSVTGroundState(lattice, J=J, plates=plates, filter_kind="gaussian", degree=degree)
    emulated = rq.run("trotter", n_steps=n_steps, order=order)
    base = rq.build_circuit("trotter", n_steps=n_steps, order=order)
    n, anc = lattice.n_qubits, lattice.n_qubits
    colours = lattice.plaquette_sublayer_indices()
    plaqs = lattice.get_plaquettes()
    circuits, meta = [], []
    qz = base.copy()
    qz.measure_all()
    circuits.append(qz)
    meta.append({'setting': 'z'})
    for c, members in enumerate(colours):
        qp = base.copy()
        append_plaquette_basis(qp, [plaqs[p] for p in members])
        qp.measure_all()
        circuits.append(qp)
        meta.append({'setting': f'plaquette colour {c}', 'plaquettes': members})

    backend = get_backend(backend_spec)
    isa = isa_circuits(circuits, backend, backend_spec)
    report = resource_report(isa, backend, [m['setting'] for m in meta])
    out = {'environment': environment(), 'experiment': 'filter', 'backend': backend_spec, 'lattice': f"{W}x{H}",
           'qubits': n + 1, 'J': J, 'degree': rq.D, 'signal_operators': rq.n_signal,
           'n_steps': n_steps, 'order': order, 'shots': shots, 'E0': rq.E0,
           'emulated': {'success_prob': emulated['success_prob'],
                        'energy': emulated['energy'], 'overlap': emulated['overlap']},
           'initial_overlap': rq.initial_overlap, 'resources': report}
    if verbose:
        for r in report:
            print(f"  {r['circuit']:<22s} 2q gates {r['two_qubit_gates']:5d}  2q depth "
                  f"{r['two_qubit_depth']:4d}  est. fidelity {r['estimated_fidelity']:.3f}")
    if dry_run:
        return out

    counts = sample(isa, backend, shots, backend_spec)
    magnetic = {}
    for m, cnt in zip(meta, counts):
        bits, weights = counts_to_arrays(cnt, n + 1)
        keep = bits[:, anc] == 0                               # ancilla post-selection
        if m['setting'] == 'z':
            out['success_prob'] = float(weights[keep].sum() / weights.sum())
            out['z'] = z_basis_observables(bits[keep], weights[keep], lattice, rq.basis)
            labels = labels_of(bits[keep][:, :n], range(n))
            dist = np.zeros(rq.dim)
            pos = np.searchsorted(rq.basis, labels)
            ok = (pos < rq.dim) & (rq.basis[np.minimum(pos, rq.dim - 1)] == labels)
            np.add.at(dist, pos[ok], weights[keep][ok])
            dist /= max(dist.sum(), 1)
            target = np.abs(rq.gs_exact) ** 2
            out['classical_fidelity_gauss'] = float(np.sum(np.sqrt(dist * target)) ** 2)
        else:
            magnetic.update(plaquette_basis_observable(
                bits[keep], weights[keep], [plaqs[p] for p in m['plaquettes']], m['plaquettes']))
    vals = np.array([magnetic[p][0] for p in sorted(magnetic)])
    errs = np.array([magnetic[p][1] for p in sorted(magnetic)])
    out['energy'] = (float(J * vals.sum()), float(abs(J) * np.sqrt((errs ** 2).sum())))
    if verbose:
        print(f"  P(ancilla = 0): measured {out['success_prob']:.3f}, emulated "
              f"{emulated['success_prob']:.3f}")
        print(f"  energy: measured {out['energy'][0]:+.4f} +- {out['energy'][1]:.4f}, emulated "
              f"{emulated['energy']:+.4f}, exact E0 {rq.E0:+.4f} (initial overlap "
              f"{rq.initial_overlap:.3f})")
        print(f"  Gauss-selected shots: {out['z']['physical_fraction']:.3f}; classical fidelity "
              f"with |psi_0|^2 = {out['classical_fidelity_gauss']:.3f}")
    return out


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _save(out, name):
    os.makedirs("results", exist_ok=True)
    tag = out['backend'].replace(":", "_")
    path = f"results/hardware_{name}_{out['lattice']}_{tag}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist")
                  else float(o))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="experiment", required=True)
    for name in ("quench", "filter"):
        p = sub.add_parser(name)
        p.add_argument("--backend", default="ideal")
        p.add_argument("--lattice", default="3x3" if name == "quench" else "2x2")
        p.add_argument("--J", type=float, default=2.0)
        p.add_argument("--shots", type=int, default=4000)
        p.add_argument("--order", type=int, default=1)
        p.add_argument("--dry-run", action="store_true",
                       help="transpile and report resources without running")
    q = sub.choices["quench"]
    q.add_argument("--dt", type=float, default=0.25)
    q.add_argument("--steps", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    q.add_argument("--rk", type=float, default=0.0)
    q.add_argument("--no-energy", action="store_true", help="computational basis only")
    f = sub.choices["filter"]
    f.add_argument("--degree", type=int, default=2)
    f.add_argument("--n-steps", type=int, default=1)
    args = parser.parse_args(argv)
    W, H = (int(x) for x in args.lattice.lower().split("x"))
    print(f"{args.experiment} on {args.lattice}, backend {args.backend}")
    if args.experiment == "quench":
        out = quench_experiment(W, H, J=args.J, rk=args.rk, dt=args.dt, steps=tuple(args.steps),
                                order=args.order, measure_energy=not args.no_energy,
                                backend_spec=args.backend, shots=args.shots,
                                dry_run=args.dry_run)
    else:
        out = filter_experiment(W, H, J=args.J, degree=args.degree, n_steps=args.n_steps,
                                order=args.order, backend_spec=args.backend,
                                shots=args.shots, dry_run=args.dry_run)
    print(f"  -> {_save(out, args.experiment)}")
    return out


if __name__ == "__main__":
    main()
