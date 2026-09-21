/-
  GaugeInvariance.lean
  U_p = S+_b S+_r S-_t S-_l changes 2 S^z on (bottom, right, top, left) by
  (+2, +2, -2, -2). A diagonal operator sum_l c_l S^z_l commutes with U_p iff
  c_b + c_r - c_t - c_l = 0. All statements are integer arithmetic.
-/
import Lattice
namespace Casimir

/-- Commutator weight of a diagonal field with coefficients (cb, cr, ct, cl). -/
def commWeight (cb cr ct cl : Int) : Int := cb + cr - ct - cl

/-- Uniform field (Joshi et al.): commutes with every plaquette, so in the
    pure-gauge model it is a conserved quantity and generates no dynamics. -/
theorem uniform_field_commutes (g : Int) : commWeight g g g g = 0 := by
  unfold commWeight; omega

/-- Staggered field, s = -p on horizontal and +p on vertical links with
    p = (-1)^(i+j) the parity of the link's origin site. On the plaquette
    anchored at a site of parity p the four coefficients are
    (-p, -p, +p, +p), so the weight is -4p: never zero. -/
theorem staggered_field_weight (p : Int) : commWeight (-p) (-p) p p = -4 * p := by
  unfold commWeight; omega

/-- Signed incidence of link `l` at site (i, j): +1 leaving (right, up),
    -1 entering (left, down), 0 otherwise. -/
def incidence (W H i j l : Nat) : Int :=
  (if l = hLink W i j then 1 else 0)
  - (if l = hLink W (i + W - 1) j then 1 else 0)
  + (if j + 1 < H ∧ l = vLink W H i j then 1 else 0)
  - (if 0 < j ∧ l = vLink W H i (j - 1) then 1 else 0)

/-- Change of the Gauss generator at site (i, j) under U_p at anchor (pi, pj). -/
def gaussChange (W H i j pi pj : Nat) : Int :=
  match plaquette W H pi pj with
  | [b, r, t, l] => incidence W H i j b + incidence W H i j r
                    - incidence W H i j t - incidence W H i j l
  | _ => 0

/-- Every plaquette operator commutes with every Gauss generator, plate rows
    included, on the whole W x H cylinder. -/
def gaugeInvariant (W H : Nat) : Bool :=
  (List.range W).all fun i => (List.range H).all fun j =>
    (List.range W).all fun pi => (List.range (H - 1)).all fun pj =>
      gaussChange W H i j pi pj == 0

theorem gauge_invariant_3x3 : gaugeInvariant 3 3 = true := by decide +kernel
theorem gauge_invariant_2x2 : gaugeInvariant 2 2 = true := by decide +kernel
theorem gauge_invariant_4x4 : gaugeInvariant 4 4 = true := by decide +kernel
theorem gauge_invariant_3x6 : gaugeInvariant 3 6 = true := by decide +kernel
theorem gauge_invariant_6x5 : gaugeInvariant 6 5 = true := by decide +kernel

end Casimir
