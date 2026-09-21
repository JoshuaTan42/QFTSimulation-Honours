/-
  Lattice.lean: geometry of the W x H cylinder, mirroring lattice.py.
  Plain Lean 4 (no Mathlib). Check with:  lean Lattice.lean
-/

namespace Casimir

/-- horizontal link (i, j) -> i mod W + W j -/
def hLink (W i j : Nat) : Nat := i % W + W * j
/-- vertical link (i, j) -> W H + i mod W + W j -/
def vLink (W H i j : Nat) : Nat := W * H + i % W + W * j

/-- (bottom, right, top, left) link qubits of the plaquette anchored at (i, j). -/
def plaquette (W H i j : Nat) : List Nat :=
  [hLink W i j, vLink W H (i + 1) j, hLink W i (j + 1), vLink W H i j]

def numLinks (W H : Nat) : Nat := W * H + W * (H - 1)

/-- N_q = W (2H - 1) for every lattice with at least one row. -/
theorem numLinks_closed_form (W H : Nat) (h : 1 ≤ H) :
    numLinks W H = W * (2 * H - 1) := by
  unfold numLinks
  have e : 2 * H - 1 = H + (H - 1) := by omega
  rw [e, Nat.mul_add]

/-- The 3 x 3 cylinder has 15 link qubits ... -/
theorem links_3x3 : numLinks 3 3 = 15 := by decide
/-- ... and RQSVT adds exactly one ancilla: 16 qubits in total. -/
theorem qubits_3x3_with_ancilla : numLinks 3 3 + 1 = 16 := by decide
/-- Parameter table of the thesis: 2x2 -> 6, 4x4 -> 28, 3x6 -> 33. -/
theorem links_table : numLinks 2 2 = 6 ∧ numLinks 4 4 = 28 ∧ numLinks 3 6 = 33 := by decide

end Casimir
