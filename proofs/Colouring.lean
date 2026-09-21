/-
  Colouring.lean: which plaquettes may run in parallel.
  Two plaquettes conflict when they share a link qubit.
-/
import Lattice
namespace Casimir

def share (p q : List Nat) : Bool := p.any fun l => q.contains l

def anchors (W H : Nat) : List (Nat × Nat) :=
  (List.range (H - 1)).flatMap fun j => (List.range W).map fun i => (i, j)

/-- A colouring is proper when distinct plaquettes of equal colour share no link. -/
def proper (W H : Nat) (colour : Nat → Nat → Nat) : Bool :=
  (anchors W H).all fun a => (anchors W H).all fun b =>
    a == b || colour a.1 a.2 != colour b.1 b.2
      || !share (plaquette W H a.1 a.2) (plaquette W H b.1 b.2)

/-- Checkerboard of Joshi et al. (open boundaries). -/
def checkerboard (i j : Nat) : Nat := (i + j) % 2
/-- Three-colouring used for odd W: 0,1,0,1,...,2 around the ring, shifted by the row. -/
def ring3 (W i j : Nat) : Nat := ((if i % W = W - 1 then 2 else (i % W) % 2) + j) % 3

/-- The 3x3 cylinder of the thesis: the checkerboard is NOT a valid sublayer split ... -/
theorem checkerboard_fails_3x3 : proper 3 3 checkerboard = false := by decide +kernel
/-- ... the concrete witness: plaquettes (0,0) and (2,0) have equal parity and share a link. -/
theorem witness_3x3 :
    checkerboard 0 0 = checkerboard 2 0 ∧
    share (plaquette 3 3 0 0) (plaquette 3 3 2 0) = true := by decide
/-- ... while three sublayers suffice. -/
theorem ring3_proper_3x3 : proper 3 3 (ring3 3) = true := by decide +kernel

theorem checkerboard_fails_odd :
    proper 3 2 checkerboard = false ∧ proper 5 3 checkerboard = false
    ∧ proper 7 3 checkerboard = false := by decide +kernel
theorem checkerboard_proper_even :
    proper 2 3 checkerboard = true ∧ proper 4 4 checkerboard = true
    ∧ proper 6 4 checkerboard = true ∧ proper 8 3 checkerboard = true := by decide +kernel
theorem ring3_proper_odd :
    proper 3 6 (ring3 3) = true ∧ proper 5 4 (ring3 5) = true
    ∧ proper 7 3 (ring3 7) = true := by decide +kernel

/-- General statement behind ring3: on an odd ring of length W >= 3, neighbouring
    columns (periodic pair included) never receive the same colour ... -/
theorem ring_neighbours_differ (W i : Nat) (hW : 3 ≤ W) (hodd : W % 2 = 1) (hi : i < W) :
    (if i = W - 1 then 2 else i % 2) ≠
    (if (i + 1) % W = W - 1 then 2 else ((i + 1) % W) % 2) := by
  by_cases h1 : i = W - 1
  · have hz : (i + 1) % W = 0 := by
      have : i + 1 = W := by omega
      rw [this]; exact Nat.mod_self W
    have hne : ¬ ((i + 1) % W = W - 1) := by omega
    rw [if_pos h1, if_neg hne, hz]
    decide
  · have hlt : i + 1 < W := by omega
    have hmod : (i + 1) % W = i + 1 := Nat.mod_eq_of_lt hlt
    rw [if_neg h1, hmod]
    by_cases h2 : i + 1 = W - 1
    · rw [if_pos h2]; omega
    · rw [if_neg h2]; omega

/-- ... and vertically stacked plaquettes differ because the colour shifts by one mod 3. -/
theorem rows_differ (c j : Nat) : (c + j) % 3 ≠ (c + (j + 1)) % 3 := by omega

end Casimir
