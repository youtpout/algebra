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
use ark_std::{vec, vec::Vec};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Below this size the default mixed-addition path wins: measured on
/// V8/x64 (31-thread pool, Vesta): 2^12 +28%, 2^13 +6%, 2^14 -13%,
/// 2^15 -17%, 2^16 -17.5% vs the default MSM.
pub const WASM_BATCH_AFFINE_MIN: usize = 1 << 14;

/// Runtime switch for the wasm32 batched-affine MSM dispatch (default
/// on). Kept for one-build A/B measurement and as a production
/// kill-switch, like the lazy-FFT one.
static BATCH_AFFINE_ENABLED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(true);

/// Enables or disables the wasm32 batched-affine MSM dispatch.
pub fn set_wasm_batch_affine_msm(enabled: bool) {
    BATCH_AFFINE_ENABLED.store(enabled, core::sync::atomic::Ordering::Relaxed);
}

#[inline]
pub(crate) fn batch_affine_enabled() -> bool {
    BATCH_AFFINE_ENABLED.load(core::sync::atomic::Ordering::Relaxed)
}

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

    // Per-thread scratch reused across windows: at small sizes the
    // per-window allocations (bucket tables + sorted copy) under the
    // single-threaded wasm allocator lock dominate the arithmetic.
    #[cfg(feature = "parallel")]
    let window_sums: Vec<_> = (0..digits_count)
        .into_par_iter()
        .map_init(Scratch::<P>::default, |sc, w| {
            window_sum::<P>(sc, bases, &scalar_digits, digits_count, w, c)
        })
        .collect();
    #[cfg(not(feature = "parallel"))]
    let window_sums: Vec<_> = {
        let mut sc = Scratch::<P>::default();
        (0..digits_count)
            .map(|w| window_sum::<P>(&mut sc, bases, &scalar_digits, digits_count, w, c))
            .collect::<Vec<_>>()
    };

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

/// Reusable per-thread buffers (see the map_init above).
struct Scratch<P: SWCurveConfig> {
    lens: Vec<u32>,
    starts: Vec<u32>,
    fill: Vec<u32>,
    pts: Vec<Affine<P>>,
    kinds: Vec<Kind>,
    dens: Vec<P::BaseField>,
    active: Vec<u32>,
}

impl<P: SWCurveConfig> Default for Scratch<P> {
    fn default() -> Self {
        Scratch {
            lens: Vec::new(),
            starts: Vec::new(),
            fill: Vec::new(),
            pts: Vec::new(),
            kinds: Vec::new(),
            dens: Vec::new(),
            active: Vec::new(),
        }
    }
}

/// One window: counting-sort the (sign-applied) points by bucket, reduce
/// each bucket with level-batched affine additions, then the usual
/// running-sum bucket reduction.
fn window_sum<P: SWCurveConfig>(
    sc: &mut Scratch<P>,
    bases: &[Affine<P>],
    scalar_digits: &[i64],
    stride: usize,
    w: usize,
    c: usize,
) -> Projective<P> {
    // Full 2^c buckets like msm_bigint_wnaf: the recoding's LAST digit is
    // not recentered and can reach 2^c - 1 in absolute value.
    let n_buckets = 1usize << c;

    let Scratch {
        lens,
        starts,
        fill,
        pts,
        kinds,
        dens,
        active,
    } = sc;
    lens.clear();
    lens.resize(n_buckets, 0);
    for (j, base) in bases.iter().enumerate() {
        let d = scalar_digits[j * stride + w];
        if d != 0 && !base.is_zero() {
            lens[(d.unsigned_abs() - 1) as usize] += 1;
        }
    }
    starts.clear();
    let mut acc = 0u32;
    for b in 0..n_buckets {
        starts.push(acc);
        acc += lens[b];
    }
    fill.clear();
    fill.extend_from_slice(starts);
    pts.clear();
    pts.resize(acc as usize, Affine::zero());
    for (j, base) in bases.iter().enumerate() {
        let d = scalar_digits[j * stride + w];
        if d == 0 || base.is_zero() {
            continue;
        }
        let idx = (d.unsigned_abs() - 1) as usize;
        pts[fill[idx] as usize] = if d < 0 { -*base } else { *base };
        fill[idx] += 1;
    }

    // Tree reduction, one batch inversion per level, in place and without
    // materializing operands: within a bucket the pair k reads offsets
    // 2k/2k+1 and writes offset k, so processing pairs in order keeps
    // every write strictly below all remaining reads. Pass 1 classifies
    // each pair (one byte) and collects the denominators; pass 2 re-reads
    // the untouched operands and writes the results.
    active.clear();
    active.extend((0..n_buckets as u32).filter(|&b| lens[b as usize] > 1));
    while !active.is_empty() {
        kinds.clear();
        dens.clear();
        for &b in active.iter() {
            let s = starts[b as usize] as usize;
            let l = lens[b as usize] as usize;
            for k in 0..l / 2 {
                let (a, q) = (&pts[s + 2 * k], &pts[s + 2 * k + 1]);
                let kind = classify::<P>(a, q);
                dens.push(match kind {
                    Kind::Add => q.x - a.x,
                    Kind::Double => a.y.double(),
                    _ => P::BaseField::ONE,
                });
                kinds.push(kind);
            }
        }
        batch_inversion(dens);
        let mut i = 0usize;
        for &b in active.iter() {
            let s = starts[b as usize] as usize;
            let l = lens[b as usize] as usize;
            let pairs = l / 2;
            for k in 0..pairs {
                let (a, q) = (pts[s + 2 * k], pts[s + 2 * k + 1]);
                pts[s + k] = apply::<P>(&a, &q, kinds[i], &dens[i]);
                i += 1;
            }
            if l % 2 == 1 {
                pts[s + pairs] = pts[s + l - 1];
            }
            lens[b as usize] = (pairs + l % 2) as u32;
        }
        active.retain(|&b| lens[b as usize] > 1);
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

#[inline(always)]
fn classify<P: SWCurveConfig>(a: &Affine<P>, b: &Affine<P>) -> Kind {
    if a.is_zero() {
        Kind::TakeB
    } else if b.is_zero() {
        Kind::TakeA
    } else if a.x != b.x {
        Kind::Add
    } else if a.y == b.y && !a.y.is_zero() {
        Kind::Double
    } else {
        Kind::Identity
    }
}

/// One affine addition with its batch-inverted denominator.
#[inline(always)]
fn apply<P: SWCurveConfig>(a: &Affine<P>, b: &Affine<P>, kind: Kind, inv: &P::BaseField) -> Affine<P> {
    match kind {
        Kind::TakeA => *a,
        Kind::TakeB => *b,
        Kind::Identity => Affine::zero(),
        Kind::Add => {
            let lambda = (b.y - a.y) * inv;
            let x3 = lambda.square() - a.x - b.x;
            let y3 = lambda * (a.x - x3) - a.y;
            Affine::new_unchecked(x3, y3)
        },
        Kind::Double => {
            let sq = a.x.square();
            let lambda = (sq.double() + sq + P::COEFF_A) * inv;
            let x3 = lambda.square() - a.x.double();
            let y3 = lambda * (a.x - x3) - a.y;
            Affine::new_unchecked(x3, y3)
        },
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
