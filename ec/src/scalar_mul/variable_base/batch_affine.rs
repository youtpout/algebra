//! Pippenger MSM with batched-affine bucket accumulation (wasm patch).
//!
//! The default `msm_bigint` accumulates buckets with mixed
//! projective+affine additions (~11 field multiplications each). Here the
//! points of each bucket are reduced by a balanced binary tree whose
//! additions run in AFFINE coordinates: one field inversion per addition,
//! amortized across every pair of the tree level with one Montgomery batch
//! inversion (~3 multiplications per element), for ~6 multiplications per
//! addition. All pairs of a level are independent by construction — no
//! collision scheduling is needed.
//!
//! Degenerate additions are classified per pair before the batch
//! inversion: identity operands, equal-x doublings (sharing the batch
//! inversion with `2y` as denominator) and `P + (-P) = identity` all take
//! explicit paths, so the kernel is correct on adversarial inputs, not
//! just random ones.
//!
//! Signed-digit recoding, window count and the final window combination
//! are the ones of `msm_bigint_wnaf`. Compiled on every target so the
//! differential tests run on native; only wasm32 call sites dispatch here
//! (native prefers the default path, where inversions are relatively
//! cheaper and mixed additions faster).

use crate::short_weierstrass::{Affine, Projective, SWCurveConfig};
use crate::AffineRepr;
use ark_ff::{batch_inversion, AdditiveGroup, Field, PrimeField, Zero};
use ark_std::{cfg_into_iter, vec, vec::Vec};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// `a + b`, distinct x coordinates: `den = b.x - a.x`.
    Add,
    /// `a + a` (equal coordinates, y != 0): `den = 2 a.y`.
    Double,
    /// `b` is the identity (also used to move an odd leftover): result `a`.
    TakeA,
    /// `a` is the identity: result `b`.
    TakeB,
    /// `a = -b` or a doubling with y = 0: result identity.
    Identity,
}

struct Pending<P: SWCurveConfig> {
    a: Affine<P>,
    b: Affine<P>,
    out: usize,
    kind: Kind,
}

/// MSM over signed windowed digits with batched-affine bucket trees.
pub fn msm_bigint_batch_affine<P: SWCurveConfig>(
    bases: &[Affine<P>],
    bigints: &[<P::ScalarField as PrimeField>::BigInt],
) -> Projective<P> {
    let size = ark_std::cmp::min(bases.len(), bigints.len());
    let scalars = &bigints[..size];
    let bases = &bases[..size];
    if size == 0 {
        return Projective::zero();
    }

    let c = if size < 32 {
        3
    } else {
        super::super::ln_without_floats(size) + 2
    };
    let num_bits = P::ScalarField::MODULUS_BIT_SIZE as usize;
    let digits_count = (num_bits + c - 1) / c;

    #[cfg(feature = "parallel")]
    let scalar_digits = scalars
        .into_par_iter()
        .flat_map_iter(|s| super::make_digits(s, c, num_bits))
        .collect::<Vec<_>>();
    #[cfg(not(feature = "parallel"))]
    let scalar_digits = scalars
        .iter()
        .flat_map(|s| super::make_digits(s, c, num_bits))
        .collect::<Vec<_>>();

    let window_sums: Vec<_> = cfg_into_iter!(0..digits_count)
        .map(|w| window_sum::<P>(bases, &scalar_digits, digits_count, w, c))
        .collect();

    let lowest = *window_sums.first().unwrap();
    lowest
        + &window_sums[1..]
            .iter()
            .rev()
            .fold(Projective::zero(), |mut total, sum_i| {
                total += sum_i;
                for _ in 0..c {
                    total.double_in_place();
                }
                total
            })
}

/// One window: counting-sort the (sign-applied) points by bucket, reduce
/// each bucket with level-batched affine additions, then the usual
/// running-sum bucket reduction.
fn window_sum<P: SWCurveConfig>(
    bases: &[Affine<P>],
    scalar_digits: &[i64],
    stride: usize,
    w: usize,
    c: usize,
) -> Projective<P> {
    // Full 2^c buckets like msm_bigint_wnaf: the recoding's LAST digit is
    // not recentered and can reach 2^c - 1 in absolute value.
    let n_buckets = 1usize << c;

    let mut lens = vec![0u32; n_buckets];
    for (j, base) in bases.iter().enumerate() {
        let d = scalar_digits[j * stride + w];
        if d != 0 && !base.is_zero() {
            lens[(d.unsigned_abs() - 1) as usize] += 1;
        }
    }
    let mut starts = vec![0u32; n_buckets];
    let mut acc = 0u32;
    for b in 0..n_buckets {
        starts[b] = acc;
        acc += lens[b];
    }
    let mut fill = starts.clone();
    let mut pts: Vec<Affine<P>> = vec![Affine::zero(); acc as usize];
    for (j, base) in bases.iter().enumerate() {
        let d = scalar_digits[j * stride + w];
        if d == 0 || base.is_zero() {
            continue;
        }
        let idx = (d.unsigned_abs() - 1) as usize;
        pts[fill[idx] as usize] = if d < 0 { -*base } else { *base };
        fill[idx] += 1;
    }

    // Tree reduction. Reads all happen while collecting the level (operand
    // values are copied into `pending`), writes all happen in the batch
    // application — in-place layout per bucket, no compaction: results of
    // a length-l bucket land at offsets 0..ceil(l/2) of the same bucket.
    let mut pending: Vec<Pending<P>> = Vec::new();
    let mut dens: Vec<P::BaseField> = Vec::new();
    loop {
        pending.clear();
        for b in 0..n_buckets {
            let s = starts[b] as usize;
            let l = lens[b] as usize;
            if l < 2 {
                continue;
            }
            let pairs = l / 2;
            for k in 0..pairs {
                pending.push(Pending {
                    a: pts[s + 2 * k],
                    b: pts[s + 2 * k + 1],
                    out: s + k,
                    kind: Kind::Add,
                });
            }
            if l % 2 == 1 && l > 1 {
                // Odd leftover moves down to close the level's layout.
                pending.push(Pending {
                    a: pts[s + l - 1],
                    b: Affine::zero(),
                    out: s + pairs,
                    kind: Kind::TakeA,
                });
            }
            lens[b] = (pairs + l % 2) as u32;
        }
        if pending.is_empty() {
            break;
        }
        batch_apply::<P>(&mut pts, &mut pending, &mut dens);
    }

    let mut running = Projective::<P>::zero();
    let mut res = Projective::<P>::zero();
    for b in (0..n_buckets).rev() {
        if lens[b] == 1 {
            running += &pts[starts[b] as usize];
        }
        res += &running;
    }
    res
}

/// Classifies every pending addition, batch-inverts the denominators of
/// the non-degenerate ones, then computes and writes all results.
fn batch_apply<P: SWCurveConfig>(
    pts: &mut [Affine<P>],
    pending: &mut [Pending<P>],
    dens: &mut Vec<P::BaseField>,
) {
    dens.clear();
    for p in pending.iter_mut() {
        p.kind = if p.a.is_zero() {
            Kind::TakeB
        } else if p.b.is_zero() {
            Kind::TakeA
        } else if p.a.x != p.b.x {
            Kind::Add
        } else if p.a.y == p.b.y && !p.a.y.is_zero() {
            Kind::Double
        } else {
            Kind::Identity
        };
        dens.push(match p.kind {
            Kind::Add => p.b.x - p.a.x,
            Kind::Double => p.a.y.double(),
            _ => P::BaseField::ONE,
        });
    }
    batch_inversion(dens);
    for (p, inv) in pending.iter().zip(dens.iter()) {
        let out = match p.kind {
            Kind::TakeA => p.a,
            Kind::TakeB => p.b,
            Kind::Identity => Affine::zero(),
            Kind::Add => {
                let lambda = (p.b.y - p.a.y) * inv;
                let x3 = lambda.square() - p.a.x - p.b.x;
                let y3 = lambda * (p.a.x - x3) - p.a.y;
                Affine::new_unchecked(x3, y3)
            },
            Kind::Double => {
                let sq = p.a.x.square();
                let lambda = (sq.double() + sq + P::COEFF_A) * inv;
                let x3 = lambda.square() - p.a.x.double();
                let y3 = lambda * (p.a.x - x3) - p.a.y;
                Affine::new_unchecked(x3, y3)
            },
        };
        pts[p.out] = out;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CurveConfig, CurveGroup, VariableBaseMSM};
    use ark_ff::{BigInt, Fp, MontBackend, MontConfig};

    /// Self-contained 255-bit field (bls12-381 Fr modulus) so the tests
    /// run on any branch without ark-test-curves (whose ark-ec is a
    /// different crate instance).
    struct TestField;
    impl MontConfig<4> for TestField {
        const MODULUS: BigInt<4> = BigInt::new([
            0xFFFFFFFF00000001,
            0x53BDA402FFFE5BFE,
            0x3339D80809A1D805,
            0x73EDA753299D7D48,
        ]);
        const GENERATOR: Fp<MontBackend<Self, 4>, 4> =
            Fp::new_unchecked(BigInt::new([7, 0, 0, 0]));
        const TWO_ADIC_ROOT_OF_UNITY: Fp<MontBackend<Self, 4>, 4> =
            Fp::new_unchecked(BigInt::new([0, 0, 0, 0]));
    }
    type F = Fp<MontBackend<TestField, 4>, 4>;

    /// Group law of y^2 = x^3 + 3 through the generator (1, 2). The a=0
    /// addition/doubling formulas never read COEFF_B, so the placeholder
    /// is harmless: every point below is produced by group operations
    /// from (1, 2) and stays on the real curve.
    struct TestCurve;
    impl CurveConfig for TestCurve {
        type BaseField = F;
        type ScalarField = F;
        const COFACTOR: &'static [u64] = &[1];
        const COFACTOR_INV: F = <F as Field>::ONE;
    }
    impl SWCurveConfig for TestCurve {
        type ZeroFlag = bool;
        const COEFF_A: F = <F as AdditiveGroup>::ZERO;
        const COEFF_B: F = <F as AdditiveGroup>::ZERO; // unused by the formulas
        const GENERATOR: Affine<TestCurve> = Affine::new_unchecked(
            <F as Field>::ONE,
            <F as Field>::ONE, // placeholder, unused
        );
    }
    type A = Affine<TestCurve>;
    type G = Projective<TestCurve>;

    fn gen_points(n: usize) -> Vec<A> {
        let g = A::new_unchecked(F::from(1u64), F::from(2u64));
        let mut acc: G = g.into_group();
        let proj: Vec<G> = (0..n)
            .map(|_| {
                let cur = acc;
                acc = acc.double() + g;
                cur
            })
            .collect();
        G::normalize_batch(&proj)
    }

    fn gen_scalars(n: usize) -> Vec<BigInt<4>> {
        let y = F::from(0x9e3779b97f4a7c15u64);
        let mut sc = F::from(3u64);
        (0..n)
            .map(|_| {
                sc.square_in_place();
                sc += y;
                sc.into_bigint()
            })
            .collect()
    }

    /// Random-ish points and scalars agree with the default MSM across
    /// sizes spanning the small-size window choice and tree depths.
    #[test]
    fn batch_affine_matches_default() {
        for n in [1usize, 2, 5, 31, 32, 100, 1000] {
            let bases = gen_points(n);
            let bigints = gen_scalars(n);
            let want = G::msm_bigint(&bases, &bigints);
            let got = msm_bigint_batch_affine(&bases, &bigints);
            assert_eq!(got, want, "mismatch at n={n}");
        }
    }

    /// Adversarial bucket contents: repeated points (doublings all the
    /// way up the tree), P and -P in the same bucket (identity mid-tree),
    /// zero and unit scalars, identity bases, empty input.
    #[test]
    fn batch_affine_handles_degenerate_additions() {
        let pts = gen_points(4);
        let (p, q) = (pts[1], pts[2]);
        let s = gen_scalars(1)[0];

        let bases = vec![p; 64];
        let bigints = vec![s; 64];
        assert_eq!(
            msm_bigint_batch_affine(&bases, &bigints),
            G::msm_bigint(&bases, &bigints),
            "repeated point"
        );

        let bases = vec![p, -p, q, p, A::zero(), q];
        let bigints = [
            s,
            s,
            gen_scalars(3)[2],
            F::from(0u64).into_bigint(),
            gen_scalars(2)[1],
            F::from(1u64).into_bigint(),
        ]
        .to_vec();
        assert_eq!(
            msm_bigint_batch_affine(&bases, &bigints),
            G::msm_bigint(&bases, &bigints),
            "mixed degenerate"
        );

        assert_eq!(
            msm_bigint_batch_affine::<TestCurve>(&[], &[]),
            G::zero(),
            "empty"
        );
    }
}
