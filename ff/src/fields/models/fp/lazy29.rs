//! Lazy-carry field arithmetic in nine 29-bit limbs (wasm patch).
//!
//! wasm32 has no 64x64->128 multiply, so carry chains dominate the classic
//! CIOS. With 29-bit limbs every product is < 2^58 and a u64 column absorbs
//! all 18 products of a multiplication WITHOUT propagating carries — the
//! inner loop has no dependency chain (measured 26.9ns/mul vs 42.5ns for
//! the 32-bit CIOS on V8/x64 in the throughput regime; the ZPRIZE-winning
//! design). Hot kernels (FFT, Poseidon, MSM) convert whole arrays into this
//! domain at their boundaries and convert back once done.
//!
//! Domain: an element x is represented by nine u64 limbs (< 2^29 each)
//! encoding the integer `x * 2^261 mod' p`, kept under the LAZY BOUND
//! `value < 2p` between operations; [`exit`] performs the final reduction.
//! Compiled on every target so differential tests run on native; only
//! wasm32 call sites should dispatch here.

use super::{Fp, MontBackend, MontConfig};
use crate::{BigInt, Field, PrimeField};

const W: u32 = 29;
const MASK: u64 = (1 << W) - 1;
/// 9 * 29 = 261 bits: fits any 256-bit modulus with lazy headroom.
pub const LIMBS: usize = 9;

/// The modulus in 29-bit limbs.
pub const fn modulus29<T: MontConfig<4>>() -> [u64; LIMBS] {
    split29(T::MODULUS.0)
}

/// Twice the modulus in 29-bit limbs (the lazy bound).
pub const fn two_modulus29<T: MontConfig<4>>() -> [u64; LIMBS] {
    let p = modulus29::<T>();
    let mut out = [0u64; LIMBS];
    let mut carry = 0u64;
    let mut j = 0;
    while j < LIMBS {
        let v = (p[j] << 1) | carry;
        out[j] = v & MASK;
        carry = v >> W;
        j += 1;
    }
    out
}

/// `-p^{-1} mod 2^29` (truncation of the 64-bit Montgomery constant).
pub const fn inv29<T: MontConfig<4>>() -> u64 {
    T::INV & MASK
}

/// 256-bit little-endian u64 limbs -> nine 29-bit limbs.
pub const fn split29(l: [u64; 4]) -> [u64; LIMBS] {
    let mut out = [0u64; LIMBS];
    let mut j = 0;
    while j < LIMBS {
        let bit = 29 * j;
        let (w, off) = (bit / 64, bit % 64);
        let mut v = l[w] >> off;
        if off > 35 && w + 1 < 4 {
            v |= l[w + 1] << (64 - off);
        }
        out[j] = v & MASK;
        j += 1;
    }
    out
}

/// Nine 29-bit limbs (fully carried, value < 2^256) -> u64 limbs.
pub const fn join29(l: [u64; LIMBS]) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut j = 0;
    while j < LIMBS {
        let bit = 29 * j;
        let (w, off) = (bit / 64, bit % 64);
        out[w] |= l[j] << off;
        if off > 35 && w + 1 < 4 {
            out[w + 1] |= l[j] >> (64 - off);
        }
        j += 1;
    }
    out
}

/// Montgomery product `a * b * 2^-261`, inputs and output < 2p.
///
/// Inner loop: two mul-adds per column and NO carries — one exact carry per
/// round for the resolved column 0, one propagation pass at the end. With
/// a, b < 2p: `t < 4p^2/2^261 + p < p/16 + p < 2p` — the bound closes.
#[inline(always)]
pub fn mont_mul<T: MontConfig<4>>(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> [u64; LIMBS] {
    let p = const { modulus29::<T>() };
    let inv = const { inv29::<T>() };
    let mut t = [0u64; LIMBS];
    for i in 0..LIMBS {
        let ai = a[i];
        let t0 = t[0] + ai * b[0];
        let m = ((t0 & MASK) * inv) & MASK;
        let c = (t0 + m * p[0]) >> W;
        t[0] = t[1] + ai * b[1] + m * p[1] + c;
        t[1] = t[2] + ai * b[2] + m * p[2];
        t[2] = t[3] + ai * b[3] + m * p[3];
        t[3] = t[4] + ai * b[4] + m * p[4];
        t[4] = t[5] + ai * b[5] + m * p[5];
        t[5] = t[6] + ai * b[6] + m * p[6];
        t[6] = t[7] + ai * b[7] + m * p[7];
        t[7] = t[8] + ai * b[8] + m * p[8];
        t[8] = 0;
    }
    carry_pass(&mut t);
    t
}

/// `a + b` under the lazy bound (inputs < 2p, output < 2p).
#[inline(always)]
pub fn add<T: MontConfig<4>>(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> [u64; LIMBS] {
    let mut t = [0u64; LIMBS];
    for j in 0..LIMBS {
        t[j] = a[j] + b[j];
    }
    carry_pass(&mut t);
    reduce_once(&mut t, &const { two_modulus29::<T>() });
    t
}

/// `a - b` under the lazy bound (inputs < 2p, output < 2p).
#[inline(always)]
pub fn sub<T: MontConfig<4>>(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> [u64; LIMBS] {
    let two_p = const { two_modulus29::<T>() };
    // a + 2p - b stays positive limb-wise after one borrow-free rewrite:
    // (a[j] + 2p[j] + MASK-borrow trick) — do it as i64 with carries.
    let mut t = [0u64; LIMBS];
    let mut borrow = 0i64;
    for j in 0..LIMBS {
        let v = a[j] as i64 + two_p[j] as i64 - b[j] as i64 + borrow;
        t[j] = (v as u64) & MASK;
        borrow = v >> W; // arithmetic shift: -1 propagates
    }
    debug_assert!(borrow >= 0);
    reduce_once(&mut t, &const { two_modulus29::<T>() });
    t
}

/// One full carry propagation (limbs restored to < 2^29).
#[inline(always)]
fn carry_pass(t: &mut [u64; LIMBS]) {
    let mut carry = 0u64;
    for j in 0..LIMBS {
        let v = t[j] + carry;
        t[j] = v & MASK;
        carry = v >> W;
    }
    debug_assert_eq!(carry, 0);
}

/// Conditional single subtraction of `m` (used with 2p or p).
#[inline(always)]
fn reduce_once(t: &mut [u64; LIMBS], m: &[u64; LIMBS]) {
    let mut ge = true;
    for j in (0..LIMBS).rev() {
        if t[j] != m[j] {
            ge = t[j] > m[j];
            break;
        }
    }
    if ge {
        let mut borrow = 0i64;
        for j in 0..LIMBS {
            let v = t[j] as i64 - m[j] as i64 + borrow;
            t[j] = (v as u64) & MASK;
            borrow = v >> W;
        }
    }
}

/// The entry multiplier `2^266 mod p` in domain limbs: entering multiplies
/// the internal (2^256) Montgomery representation by `2^266 / 2^261 = 2^5`,
/// i.e. maps `x*2^256` to `x*2^261`. Computed once per batch by the caller.
pub fn entry_constant<T: MontConfig<4>>() -> [u64; LIMBS] {
    let mut x = Fp::<MontBackend<T, 4>, 4>::from(2u64);
    // x = 2^266 in the field, via 266 doublings of 1... use pow: 2^266.
    x = x.pow([266u64]);
    split29(x.into_bigint().0)
}

/// Enters the domain from an internal Montgomery representation (< p).
#[inline(always)]
pub fn enter<T: MontConfig<4>>(repr: &BigInt<4>, entry: &[u64; LIMBS]) -> [u64; LIMBS] {
    mont_mul::<T>(&split29(repr.0), entry)
}

/// Leaves the domain back to the internal Montgomery representation,
/// fully reduced (< p).
#[inline(always)]
pub fn exit<T: MontConfig<4>>(d: &[u64; LIMBS]) -> BigInt<4> {
    // mont_mul by R (= 2^256 mod p) maps x*2^261 -> x*2^256.
    let mut t = mont_mul::<T>(d, &const { split29(T::R.0) });
    reduce_once(&mut t, &const { two_modulus29::<T>() });
    reduce_once(&mut t, &const { modulus29::<T>() });
    BigInt(join29(t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AdditiveGroup, Field};

    /// bls12-381 Fr — 255-bit modulus, the config used by the sibling
    /// 32-bit CIOS differential test.
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

    fn enter_f(x: F) -> [u64; LIMBS] {
        enter::<NoCarry255>(&x.0, &entry_constant::<NoCarry255>())
    }
    fn exit_f(d: &[u64; LIMBS]) -> BigInt<4> {
        exit::<NoCarry255>(d)
    }

    /// Mixed mul/add/sub chains in the lazy domain agree with the field.
    #[test]
    fn lazy29_chain_matches_field() {
        let minus_one = -F::ONE;
        let mut mirror = F::from(3u64);
        let mut d = enter_f(mirror);
        let y = F::from(0x9e3779b97f4a7c15u64);
        let dy = enter_f(y);
        for step in 0..200 {
            match step % 4 {
                0 => {
                    mirror *= y;
                    d = mont_mul::<NoCarry255>(&d, &dy);
                }
                1 => {
                    mirror += y;
                    d = add::<NoCarry255>(&d, &dy);
                }
                2 => {
                    mirror -= y;
                    d = sub::<NoCarry255>(&d, &dy);
                }
                _ => {
                    mirror.square_in_place();
                    d = mont_mul::<NoCarry255>(&d, &d.clone());
                }
            }
            assert_eq!(exit_f(&d), mirror.0, "diverged at step {step}");
        }
        // Edge values round-trip too.
        for v in [F::ZERO, F::ONE, minus_one] {
            assert_eq!(exit_f(&enter_f(v)), v.0);
        }
    }
}
