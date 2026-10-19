/-
  Casimir.lean: identities used in the force extraction and in Richardson
  extrapolation. Integer arithmetic only.
-/
namespace Casimir

/-- The second difference removes the bulk a + b d exactly, so
    D2(d) = E0(d+1) - 2 E0(d) + E0(d-1) needs no bulk fit. -/
theorem second_difference_kills_bulk (a b d : Int) :
    (a + b * (d + 1)) - 2 * (a + b * d) + (a + b * (d - 1)) = 0 := by
  rw [Int.mul_add, Int.mul_sub]; omega

/-- Lattice central difference of 1/d^2 (cleared of denominators):
    (d-1)^2 - (d+1)^2 = -4d, i.e.
    [1/(d+1)^2 - 1/(d-1)^2] / 2 = -2d / (d^2-1)^2, which exceeds the continuum
    derivative -2/d^3 by the factor d^4/(d^2-1)^2 (16/9 at d = 2). -/
theorem central_difference_inverse_square (d : Int) :
    (d - 1) * (d - 1) - (d + 1) * (d + 1) = -4 * d := by
  have h1 : (d - 1) * (d - 1) = d * d - 2 * d + 1 := by
    rw [Int.sub_mul, Int.mul_sub, Int.mul_sub]; omega
  have h2 : (d + 1) * (d + 1) = d * d + 2 * d + 1 := by
    rw [Int.add_mul, Int.mul_add, Int.mul_add]; omega
  omega

theorem central_difference_bias_at_2 : (2 : Nat)^4 * 9 = 16 * (2^2 - 1)^2 := by decide

/-- Richardson weights for step counts n = 1, 2, 4 (h = 1, 1/2, 1/4) are
    (1/3, -2, 8/3). Scaled by 3, with h in units of 1/4:
    they sum to one and cancel the h and h^2 error terms. -/
theorem richardson_1_2_4 :
    (1 : Int) + (-6) + 8 = 3 ∧
    (1 : Int) * 4 + (-6) * 2 + 8 * 1 = 0 ∧
    (1 : Int) * 16 + (-6) * 4 + 8 * 1 = 0 := by decide

/-- Same for the qDRIFT schedule N = 200, 400, 800 (the ratios are identical). -/
theorem richardson_weights_scale_free (s : Int) :
    1 * (4 * s) + (-6) * (2 * s) + 8 * s = 0 := by omega

end Casimir
