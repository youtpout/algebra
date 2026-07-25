//! Two-lane NEON Montgomery multiplication over the 29-bit-limb domain.
//!
//! aarch64 has no 64x64->128 SIMD multiply, so the only way to use NEON for a
//! 256-bit field is a reduced radix: `vmlal_u32` widens two 32x32 products to
//! 64 bits per instruction, and 29-bit limbs leave enough headroom to
//! accumulate a whole column without propagating a carry. That is exactly the
//! shape of [`super::lazy29`], which this module vectorises: the scalar
//! routine multiplies one element per pass, this one multiplies two.
//!
//! Unlike the 64-bit CIOS, nothing here touches the condition flags, so there
//! is no carry chain to serialise -- the reason to expect anything at all from
//! SIMD on this architecture.
//!
//! Inputs and outputs are in the `2^261` Montgomery domain and bounded by
//! `2 * p`, with the same contract as [`super::lazy29::mont_mul_p`]; entering
//! and leaving the domain is the caller's business (see `lazy29::enter`/`exit`)
//! and only pays off when amortised over many operations.

#![cfg(target_arch = "aarch64")]

use core::arch::aarch64::*;

use super::lazy29::{Params, LIMBS};

const W: i32 = 29;
const MASK: u32 = (1 << W) - 1;

/// Two independent Montgomery products, one per NEON lane: lane 0 holds the
/// first element, lane 1 the second.
///
/// Same algorithm as the scalar 29-bit routine, so the results are identical;
/// the differential test in this module checks it on random inputs.
#[allow(unsafe_code)] // the point of the module
#[inline(always)]
pub fn mont_mul2(
    pr: &Params,
    a: &[[u64; LIMBS]; 2],
    b: &[[u64; LIMBS]; 2],
) -> [[u64; LIMBS]; 2] {
    unsafe {
        let mask = vdup_n_u32(MASK);
        let inv = vdup_n_u32(pr.inv as u32);

        // Limbs of both elements, interleaved two per vector.
        let mut a_limbs = [vdup_n_u32(0); LIMBS];
        let mut b_limbs = [vdup_n_u32(0); LIMBS];
        let mut p_limbs = [vdup_n_u32(0); LIMBS];
        for j in 0..LIMBS {
            a_limbs[j] = pack(a[0][j], a[1][j]);
            b_limbs[j] = pack(b[0][j], b[1][j]);
            p_limbs[j] = pack(pr.p[j], pr.p[j]);
        }

        let mut t = [vdupq_n_u64(0); LIMBS];
        for i in 0..LIMBS {
            let ai = a_limbs[i];

            // t0 = t[0] + a[i] * b[0]
            let t0 = vmlal_u32(t[0], ai, b_limbs[0]);
            // m = ((t0 mod 2^29) * inv) mod 2^29
            let m = vand_u32(vmul_u32(vand_u32(vmovn_u64(t0), mask), inv), mask);
            // c = (t0 + m * p[0]) >> 29 -- the low 29 bits vanish by
            // construction of m, so only the shifted part carries over.
            let c = vshrq_n_u64::<W>(vmlal_u32(t0, m, p_limbs[0]));

            // Column 0 additionally absorbs the carry; the others are two
            // multiply-accumulates each, with no carry in between.
            t[0] = vaddq_u64(
                vmlal_u32(vmlal_u32(t[1], ai, b_limbs[1]), m, p_limbs[1]),
                c,
            );
            for j in 1..LIMBS - 1 {
                t[j] = vmlal_u32(vmlal_u32(t[j + 1], ai, b_limbs[j + 1]), m, p_limbs[j + 1]);
            }
            t[LIMBS - 1] = vdupq_n_u64(0);
        }

        // One propagation pass, both lanes at once.
        let mask64 = vdupq_n_u64(MASK as u64);
        let mut carry = vdupq_n_u64(0);
        let mut out = [[0u64; LIMBS]; 2];
        for j in 0..LIMBS {
            let v = vaddq_u64(t[j], carry);
            let limb = vandq_u64(v, mask64);
            carry = vshrq_n_u64::<W>(v);
            out[0][j] = vgetq_lane_u64::<0>(limb);
            out[1][j] = vgetq_lane_u64::<1>(limb);
        }
        debug_assert_eq!(vgetq_lane_u64::<0>(carry), 0);
        debug_assert_eq!(vgetq_lane_u64::<1>(carry), 0);
        out
    }
}

/// Two 29-bit limbs into the two 32-bit lanes of one vector.
#[allow(unsafe_code)]
#[inline(always)]
unsafe fn pack(low: u64, high: u64) -> uint32x2_t {
    let pair = [low as u32, high as u32];
    vld1_u32(pair.as_ptr())
}
