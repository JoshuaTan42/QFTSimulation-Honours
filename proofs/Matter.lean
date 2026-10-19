/-
  Matter.lean: the matter-coupled model of Joshi et al. (QuantumFields/matter_field.py).
  Plain Lean 4, integer arithmetic only.
-/
namespace Casimir

/-- Qubits of a W x H open lattice: (W-1) H horizontal links, W (H-1) vertical
    links and one matter qubit per site. -/
def numOpen (W H : Nat) : Nat := (W - 1) * H + W * (H - 1) + W * H

/-- The lattices of Joshi et al.: 4 x 3 has 29 qubits and 5 x 4 has 51. -/
theorem joshi_qubits : numOpen 4 3 = 29 ∧ numOpen 5 4 = 51 := by decide

/-- The hopping term phi^dag_a S^+_l phi_b raises the charge rho_a and the
    divergence at a by one (l leaves a) and lowers both at b by one (l enters b),
    so G = div E - rho - Q is unchanged at both ends: hopping is gauge invariant. -/
theorem hopping_gauge_invariant (dEa rhoa dEb rhob Qa Qb : Int) :
    (dEa + 1) - (rhoa + 1) - Qa = dEa - rhoa - Qa ∧
    (dEb - 1) - (rhob - 1) - Qb = dEb - rhob - Qb := by
  exact ⟨by omega, by omega⟩

end Casimir
