/-
  PlaquettePauli.lean: the eight-string Pauli expansion of the plaquette term,
      16 (U + U^dag) = 2 * sum_k sign_k P_k,   U = S+ (x) S+ (x) S- (x) S-,
  with 2 S+- = X +- iY, checked on all 256 matrix entries over the Gaussian
  integers. Also (U + U^dag)^2 = U U^dag + U^dag U in its Pauli-Z form.
-/
namespace Casimir

structure GInt where
  re : Int
  im : Int
deriving DecidableEq, Repr

namespace GInt
def add (a b : GInt) : GInt := ⟨a.re + b.re, a.im + b.im⟩
def mul (a b : GInt) : GInt := ⟨a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re⟩
def smul (k : Int) (a : GInt) : GInt := ⟨k * a.re, k * a.im⟩
def zero : GInt := ⟨0, 0⟩
end GInt

/-- 2x2 matrices as functions of (row, column) bits. -/
abbrev M2 := Bool → Bool → GInt

def pI : M2 := fun r c => if r == c then ⟨1, 0⟩ else ⟨0, 0⟩
def pX : M2 := fun r c => if r != c then ⟨1, 0⟩ else ⟨0, 0⟩
def pY : M2 := fun r c => if r == c then ⟨0, 0⟩ else if r then ⟨0, 1⟩ else ⟨0, -1⟩
def pZ : M2 := fun r c => if r != c then ⟨0, 0⟩ else if r then ⟨-1, 0⟩ else ⟨1, 0⟩
/-- 2 S+ = X + iY and 2 S- = X - iY. -/
def sp2 : M2 := fun r c => GInt.add (pX r c) (GInt.mul ⟨0, 1⟩ (pY r c))
def sm2 : M2 := fun r c => GInt.add (pX r c) (GInt.mul ⟨0, -1⟩ (pY r c))

/-- Entry of A (x) B (x) C (x) D. -/
def kron4 (A B C D : M2) (r c : Bool × Bool × Bool × Bool) : GInt :=
  GInt.mul (GInt.mul (A r.1 c.1) (B r.2.1 c.2.1)) (GInt.mul (C r.2.2.1 c.2.2.1) (D r.2.2.2 c.2.2.2))

def bools : List Bool := [false, true]
def idx : List (Bool × Bool × Bool × Bool) :=
  bools.flatMap fun a => bools.flatMap fun b => bools.flatMap fun c => bools.map fun d => (a, b, c, d)

def lhs (r c : Bool × Bool × Bool × Bool) : GInt :=
  GInt.add (kron4 sp2 sp2 sm2 sm2 r c) (kron4 sm2 sm2 sp2 sp2 r c)

def strings : List (Int × M2 × M2 × M2 × M2) :=
  [(1, pX, pX, pX, pX), (-1, pX, pX, pY, pY), (1, pX, pY, pX, pY), (1, pX, pY, pY, pX),
   (1, pY, pX, pX, pY), (1, pY, pX, pY, pX), (-1, pY, pY, pX, pX), (1, pY, pY, pY, pY)]

def rhs (r c : Bool × Bool × Bool × Bool) : GInt :=
  GInt.smul 2 (strings.foldl (fun acc s => GInt.add acc (GInt.smul s.1 (kron4 s.2.1 s.2.2.1 s.2.2.2.1 s.2.2.2.2 r c))) GInt.zero)

/-- 16 (U + U^dag) = 2 (XXXX - XXYY + XYXY + XYYX + YXXY + YXYX - YYXX + YYYY). -/
theorem plaquette_pauli_expansion :
    (idx.all fun r => idx.all fun c => decide (lhs r c = rhs r c)) = true := by decide +kernel

/-- RK term. P0 = (1 + Z)/2, P1 = (1 - Z)/2, so
    16 (U U^dag + U^dag U) = (1+Z)(1+Z)(1-Z)(1-Z) + (1-Z)(1-Z)(1+Z)(1+Z)
                           = 2 (1 + ZZII + IIZZ - ZIZI - ZIIZ - IZZI - IZIZ + ZZZZ). -/
def onePlusZ : M2 := fun r c => GInt.add (pI r c) (pZ r c)
def oneMinusZ : M2 := fun r c => GInt.add (pI r c) (GInt.smul (-1) (pZ r c))

def rkLhs (r c : Bool × Bool × Bool × Bool) : GInt :=
  GInt.add (kron4 onePlusZ onePlusZ oneMinusZ oneMinusZ r c)
           (kron4 oneMinusZ oneMinusZ onePlusZ onePlusZ r c)

def rkStrings : List (Int × M2 × M2 × M2 × M2) :=
  [(1, pI, pI, pI, pI), (1, pZ, pZ, pI, pI), (1, pI, pI, pZ, pZ), (-1, pZ, pI, pZ, pI),
   (-1, pZ, pI, pI, pZ), (-1, pI, pZ, pZ, pI), (-1, pI, pZ, pI, pZ), (1, pZ, pZ, pZ, pZ)]

def rkRhs (r c : Bool × Bool × Bool × Bool) : GInt :=
  GInt.smul 2 (rkStrings.foldl (fun acc s => GInt.add acc (GInt.smul s.1 (kron4 s.2.1 s.2.2.1 s.2.2.2.1 s.2.2.2.2 r c))) GInt.zero)

theorem rk_pauli_expansion :
    (idx.all fun r => idx.all fun c => decide (rkLhs r c = rkRhs r c)) = true := by decide +kernel

end Casimir
