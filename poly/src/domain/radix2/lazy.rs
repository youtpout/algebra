//! wasm32 FFT kernels in the lazy-carry 29-bit-limb domain
//! (`ark_ff::lazy29`): the butterfly networks and the parallel
//! decomposition are copied from `fft.rs` operation for operation, only the
//! element representation changes — arrays are converted at the boundaries.
//! On wasm this trades the 64-bit carry chains (emulated 64x64->128
//! multiplies) for carry-free u64 columns, a measured ~1.6x on butterfly
//! throughput.
//!
//! Dispatched from `io_helper`/`oi_helper` on wasm32 only — every FFT entry
//! point (fft/ifft/coset/degree-aware) funnels through those two. Compiled
//! on every target so the differential tests below run on native.
//!
//! The dispatch is runtime-checked, not type-directed: [`detect`] accepts
//! any `F` that is a 4-limb prime field in standard Montgomery
//! representation (`x * 2^256 mod p`), verified by comparing the
//! representations of 1 and 2 against `R`/`2R` derived from
//! `F::characteristic()` alone. Only after that proof do we reinterpret
//! element memory as `[u64; 4]` limbs.
#![allow(unsafe_code)]
// Dispatched only on wasm32; native builds keep the code for the tests.
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

use super::Radix2EvaluationDomain;
use crate::domain::DomainCoeff;
use ark_ff::{lazy29 as lz, FftField};
use ark_std::{cfg_chunks_mut, cfg_into_iter, cfg_iter, cfg_iter_mut, vec, vec::*};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

type L = [u64; lz::LIMBS];

/// Below this size the setup (constant derivation + conversions) is not
/// worth amortizing over the stages.
pub(crate) const MIN_LAZY_FFT_SIZE: usize = 128;

// Same empirical thresholds as fft.rs (private to that module).
const MIN_NUM_CHUNKS_FOR_COMPACTION: usize = 1 << 7;
const MIN_GAP_SIZE_FOR_PARALLELIZATION: usize = 1 << 10;
const MIN_INPUT_SIZE_FOR_PARALLELIZATION: usize = 1 << 10;

/// Reinterprets `&mut [T]` as `&mut [F]` when `T` is exactly `F` (the only
/// `DomainCoeff` the lazy kernels handle).
#[inline]
pub(crate) fn as_field_mut<F: FftField, T: DomainCoeff<F>>(x: &mut [T]) -> Option<&mut [F]> {
    if core::any::TypeId::of::<T>() == core::any::TypeId::of::<F>() {
        // SAFETY: T and F are the same type.
        Some(unsafe { &mut *(x as *mut [T] as *mut [F]) })
    } else {
        None
    }
}

/// Runtime capability probe. `Some` only when `F` is a prime field of four
/// u64 limbs whose in-memory representation is standard Montgomery form:
/// the layout is checked structurally (size/alignment/extension degree) and
/// the representation semantically (repr(1) == R and repr(2) == 2R, with R
/// computed from the characteristic by modular doubling).
pub(crate) fn detect<F: FftField>() -> Option<lz::Params> {
    if core::mem::size_of::<F>() != 32
        || core::mem::align_of::<F>() < core::mem::align_of::<u64>()
        || F::extension_degree() != 1
    {
        return None;
    }
    let ch = F::characteristic();
    if ch.len() != 4 {
        return None;
    }
    let pr = lz::Params::from_modulus([ch[0], ch[1], ch[2], ch[3]])?;
    let one = F::one();
    let two = one + one;
    if *repr(&one) != pr.r || *repr(&two) != pr.two_r {
        return None;
    }
    Some(pr)
}

/// Only sound for `F` accepted by [`detect`] (all callers below).
#[inline(always)]
fn repr<F>(x: &F) -> &[u64; 4] {
    unsafe { &*(x as *const F as *const [u64; 4]) }
}

/// Only sound for `F` accepted by [`detect`]; the written limbs are a fully
/// reduced Montgomery representation (< p), i.e. a valid element.
#[inline(always)]
fn repr_mut<F>(x: &mut F) -> &mut [u64; 4] {
    unsafe { &mut *(x as *mut F as *mut [u64; 4]) }
}

/// `fft.rs::butterfly_fn_io`, element ops in the lazy domain.
#[inline(always)]
fn butterfly_io(pr: &lz::Params, lo: &mut L, hi: &mut L, root: &L) {
    let neg = lz::sub_p(pr, lo, hi);
    *lo = lz::add_p(pr, lo, hi);
    *hi = lz::mont_mul_p(pr, &neg, root);
}

/// `fft.rs::butterfly_fn_oi`, element ops in the lazy domain.
#[inline(always)]
fn butterfly_oi(pr: &lz::Params, lo: &mut L, hi: &mut L, root: &L) {
    let t = lz::mont_mul_p(pr, hi, root);
    let neg = lz::sub_p(pr, lo, &t);
    *lo = lz::add_p(pr, lo, &t);
    *hi = neg;
}

/// `fft.rs::apply_butterfly` with lazy elements — identical chunking and
/// parallelization decisions.
#[allow(clippy::too_many_arguments)]
fn apply_butterfly_lazy<G: Fn(&lz::Params, &mut L, &mut L, &L) + Copy + Sync + Send>(
    g: G,
    pr: &lz::Params,
    xi: &mut [L],
    roots: &[L],
    step: usize,
    chunk_size: usize,
    num_chunks: usize,
    max_threads: usize,
    gap: usize,
) {
    if xi.len() <= MIN_INPUT_SIZE_FOR_PARALLELIZATION {
        xi.chunks_mut(chunk_size).for_each(|cxi| {
            let (lo, hi) = cxi.split_at_mut(gap);
            lo.iter_mut()
                .zip(hi)
                .zip(roots.iter().step_by(step))
                .for_each(|((lo, hi), root)| g(pr, lo, hi, root));
        });
    } else {
        cfg_chunks_mut!(xi, chunk_size).for_each(|cxi| {
            let (lo, hi) = cxi.split_at_mut(gap);
            // If the chunk is sufficiently big that parallelism helps,
            // we parallelize the butterfly operation within the chunk.
            if gap > MIN_GAP_SIZE_FOR_PARALLELIZATION && num_chunks < max_threads {
                cfg_iter_mut!(lo)
                    .zip(hi)
                    .zip(cfg_iter!(roots).step_by(step))
                    .for_each(|((lo, hi), root)| g(pr, lo, hi, root));
            } else {
                lo.iter_mut()
                    .zip(hi)
                    .zip(roots.iter().step_by(step))
                    .for_each(|((lo, hi), root)| g(pr, lo, hi, root));
            }
        });
    }
}

impl<F: FftField> Radix2EvaluationDomain<F> {
    /// `fft.rs::io_helper` with butterflies in the lazy domain: identical
    /// butterfly network, so bit-for-bit the same (out-of-order) output
    /// layout. `F` must have been accepted by [`detect`] (which produced
    /// `pr`).
    pub(crate) fn io_helper_lazy(&self, xi: &mut [F], root: F, pr: &lz::Params) {
        let roots_f = self.roots_of_unity(root);
        let mut roots: Vec<L> = cfg_iter!(roots_f).map(|r| lz::enter_p(pr, repr(r))).collect();
        let mut d: Vec<L> = cfg_iter!(xi).map(|x| lz::enter_p(pr, repr(x))).collect();

        let mut step = 1;
        let mut first = true;

        #[cfg(feature = "parallel")]
        let max_threads = rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let max_threads = 1;

        let mut gap = d.len() / 2;
        while gap > 0 {
            // each butterfly cluster uses 2*gap positions
            let chunk_size = 2 * gap;
            let num_chunks = d.len() / chunk_size;

            if num_chunks >= MIN_NUM_CHUNKS_FOR_COMPACTION {
                if !first {
                    roots = cfg_into_iter!(roots).step_by(step * 2).collect();
                }
                step = 1;
                roots.shrink_to_fit();
            } else {
                step = num_chunks;
            }
            first = false;

            apply_butterfly_lazy(
                butterfly_io,
                pr,
                &mut d,
                &roots,
                step,
                chunk_size,
                num_chunks,
                max_threads,
                gap,
            );

            gap /= 2;
        }

        cfg_iter_mut!(xi)
            .zip(d)
            .for_each(|(x, v)| *repr_mut(x) = lz::exit_p(pr, &v));
    }

    /// `fft.rs::oi_helper` with butterflies in the lazy domain (same output
    /// layout). `F` must have been accepted by [`detect`].
    pub(crate) fn oi_helper_lazy(&self, xi: &mut [F], root: F, start_gap: usize, pr: &lz::Params) {
        let roots_f = self.roots_of_unity(root);
        let roots_cache: Vec<L> = cfg_iter!(roots_f).map(|r| lz::enter_p(pr, repr(r))).collect();
        let mut d: Vec<L> = cfg_iter!(xi).map(|x| lz::enter_p(pr, repr(x))).collect();

        let compaction_max_size = core::cmp::min(
            roots_cache.len() / 2,
            roots_cache.len() / MIN_NUM_CHUNKS_FOR_COMPACTION,
        );
        let mut compacted_roots = vec![[0u64; lz::LIMBS]; compaction_max_size];

        #[cfg(feature = "parallel")]
        let max_threads = rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let max_threads = 1;

        let mut gap = start_gap;
        while gap < d.len() {
            // each butterfly cluster uses 2*gap positions
            let chunk_size = 2 * gap;
            let num_chunks = d.len() / chunk_size;

            let (roots, step) = if num_chunks >= MIN_NUM_CHUNKS_FOR_COMPACTION && gap < d.len() / 2
            {
                cfg_iter!(roots_cache)
                    .step_by(num_chunks)
                    .zip(&mut compacted_roots[..gap])
                    .for_each(|(b, a)| *a = *b);

                (&compacted_roots[..gap], 1)
            } else {
                (&roots_cache[..], num_chunks)
            };

            apply_butterfly_lazy(
                butterfly_oi,
                pr,
                &mut d,
                roots,
                step,
                chunk_size,
                num_chunks,
                max_threads,
                gap,
            );

            gap *= 2;
        }

        cfg_iter_mut!(xi)
            .zip(d)
            .for_each(|(x, v)| *repr_mut(x) = lz::exit_p(pr, &v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{BigInt, Field, Fp, MontBackend, MontConfig, One};

    /// bls12-381 Fr (255-bit) with placeholder fft constants: the
    /// differential tests below drive the helpers directly with an
    /// arbitrary "generator", which exercises the identical butterfly
    /// networks without needing a real root of unity.
    struct NoCarry255;
    impl MontConfig<4> for NoCarry255 {
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
    type F = Fp<MontBackend<NoCarry255, 4>, 4>;

    fn fake_domain(n: usize, g: F) -> Radix2EvaluationDomain<F> {
        Radix2EvaluationDomain {
            size: n as u64,
            log_size_of_group: n.trailing_zeros(),
            size_as_field_element: F::from(n as u64),
            size_inv: F::from(n as u64).inverse().unwrap(),
            group_gen: g,
            group_gen_inv: g.inverse().unwrap(),
            offset: F::one(),
            offset_inv: F::one(),
            offset_pow_size: F::one(),
        }
    }

    fn varied_data(n: usize) -> Vec<F> {
        let y = F::from(0x9e3779b97f4a7c15u64);
        let mut x = F::one() + y;
        (0..n)
            .map(|_| {
                x.square_in_place();
                x += y;
                x
            })
            .collect()
    }

    #[test]
    fn detect_accepts_mont4_field() {
        let pr = detect::<F>().expect("4-limb Montgomery field must be accepted");
        assert_eq!(pr, ark_ff::lazy29::params::<NoCarry255>());
    }

    /// The lazy helpers produce bit-for-bit the layouts of the generic
    /// helpers, across sizes that exercise both parallelization branches
    /// and the root-compaction path, and for `start_gap > 1`
    /// (degree-aware FFT).
    #[test]
    fn lazy_helpers_match_generic_helpers() {
        let g = F::from(0xc2b2ae3d27d4eb4fu64);
        let pr = detect::<F>().unwrap();
        for log_n in [7usize, 10, 12] {
            let n = 1 << log_n;
            let domain = fake_domain(n, g);
            let data = varied_data(n);

            let mut want = data.clone();
            domain.io_helper(&mut want, g);
            let mut got = data.clone();
            domain.io_helper_lazy(&mut got, g, &pr);
            assert_eq!(got, want, "io mismatch at n={n}");

            for start_gap in [1usize, 8] {
                let mut want = data.clone();
                domain.oi_helper(&mut want, g, start_gap);
                let mut got = data.clone();
                domain.oi_helper_lazy(&mut got, g, start_gap, &pr);
                assert_eq!(got, want, "oi mismatch at n={n}, start_gap={start_gap}");
            }
        }
    }
}
