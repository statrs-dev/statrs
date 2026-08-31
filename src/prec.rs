#![allow(unused_macros, unused_imports)]
//! Provides utility functions for working with floating point precision.
//!
//! This module is intended for internal use within the `statrs` crate to ensure consistent
//! precision checking across all statistical computations. While it is currently public
//! for historical reasons, it will be made private in a future breaking release.
//!
//! # Usage
//!
//! The module provides three main types of precision checks:
//!
//! 1. Absolute difference checks (`abs_diff_eq!`) - Use when comparing values that should
//!    be close in absolute terms, e.g., when checking if a value is close to zero
//!
//! 2. Relative difference checks (`relative_eq!`) - Use when comparing values that scale
//!    with the input, e.g., when comparing probability densities or statistical moments
//!
//! 3. ULPs (Units in Last Place) checks (`ulps_eq!`) - Use for comparing values that
//!    should be different due only to rounding effects, not algorithmic ones. This can
//!    include cases where FMA use depends on compilation parameters. Usually test-only.
//!
//! Each check type has both a non-asserting version (e.g., `abs_diff_eq!`) and an
//! asserting version (e.g., `assert_abs_diff_eq!`).
//!
//! # Default Precision Levels
//!
//! The module defines default precision levels that are carefully chosen to balance
//! correctness and performance:
//!
//! - `DEFAULT_RELATIVE_ACC`: 1e-14 for relative comparisons
//! - `DEFAULT_EPS`: 1e-9 for absolute comparisons
//! - `DEFAULT_ULPS`: 5 for ULPs comparisons
//!
//! These defaults should be used unless there is a specific reason to use different
//! precision levels.
//!
//! # Module-Specific Precision
//!
//! Some modules may require different precision levels than the crate defaults. In such
//! cases, the module should define its own precision constants using the same names as
//! defined here (e.g., `MODULE_RELATIVE_ACC`, `MODULE_EPS`) to maintain consistency
//! and searchability.
//!
//! # Deprecated Functionality
//!
//! The following items are deprecated and will be removed in a future release:
//! - `almost_eq` function - Use `abs_diff_eq!` macro instead
//! - `assert_almost_eq!` macro - Use `assert_abs_diff_eq!` macro instead

/// Standard epsilon, maximum relative precision of IEEE 754 double-precision
/// floating point numbers (64 bit) e.g. `2^-53`
pub const F64_PREC: f64 = 0.00000000000000011102230246251565;

/// Default accuracy for `f64`, equivalent to `0.0 * F64_PREC`
pub const DEFAULT_F64_ACC: f64 = 0.0000000000000011102230246251565;

/// Default and target relative accuracy for f64 operations
pub const DEFAULT_RELATIVE_ACC: f64 = 1e-14;

/// Default and target absolute accuracy for f64 operations
pub const DEFAULT_EPS: f64 = 1e-9;

/// Default and target ULPs accuracy for f64 operations
pub const DEFAULT_ULPS: u32 = 5;

/// Compares if two floats are close via `approx::abs_diff_eq`
/// using a maximum absolute difference (epsilon) of `acc`.
#[deprecated(since = "0.19.0", note = "Use abs_diff_eq! macro instead")]
pub fn almost_eq(a: f64, b: f64, acc: f64) -> bool {
    use approx::AbsDiffEq;
    if a.is_infinite() && b.is_infinite() {
        return a == b;
    }
    a.abs_diff_eq(&b, acc)
}

/// Compares if two floats are close via `prec::relative_eq!`
/// Updates first argument to value of second argument
pub(crate) fn convergence(x: &mut f64, x_new: f64) -> bool {
    let res = relative_eq!(*x, x_new);
    *x = x_new;
    res
}

/// Decomposes `x` into a mantissa in `[0.5, 1)` and an exponent such that
/// `x == mantissa * 2^exponent`, avoiding overflow when multiplying two
/// widely-scaled values together before taking their logarithm.
pub(crate) fn frexp(x: f64) -> (f64, i32) {
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000_0000_0000;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let mantissa_bits = bits & 0x000f_ffff_ffff_ffff;
    if exponent == 0 {
        if mantissa_bits == 0 {
            return (x, 0);
        }
        // subnormal: rescale by 2^54 to bring it into the normal range first
        let (m, e) = frexp(x * f64::from_bits(0x4350000000000000));
        return (m, e - 54);
    }
    let m = f64::from_bits(sign | (1022u64 << 52) | mantissa_bits);
    (m, exponent - 1022)
}

/// Splits the exact rounding error out of the product `a * b`
/// (Dekker's algorithm): `a * b == p + dekker_product_err(a, b, p)` exactly.
///
/// Used instead of `f64::mul_add`, which falls back to a slow software FMA on
/// targets without the hardware instruction (e.g. baseline x86-64).
#[inline]
pub(crate) fn dekker_product_err(a: f64, b: f64, p: f64) -> f64 {
    const SPLIT: f64 = 134_217_729.0; // 2^27 + 1
    // Veltkamp's split below multiplies by `SPLIT`, which overflows to `inf`
    // once an argument passes about `1.3e300` and then yields `inf - inf`, i.e.
    // NaN. Scaling by a power of two is exact and the residual scales with it,
    // so rescale rather than give up: `err(a b) = 2^k err((a 2^-k) b)`.
    const BIG: f64 = 1e300;
    const DOWN: f64 = 9.313225746154785e-10; // 2^-30, exact
    let (mut a, mut b, mut p) = (a, b, p);
    let mut scale = 1.0;
    if a.abs() > BIG {
        a *= DOWN;
        p *= DOWN;
        scale /= DOWN;
    }
    if b.abs() > BIG {
        b *= DOWN;
        p *= DOWN;
        scale /= DOWN;
    }
    let ca = SPLIT * a;
    let a_hi = ca - (ca - a);
    let a_lo = a - a_hi;
    let cb = SPLIT * b;
    let b_hi = cb - (cb - b);
    let b_lo = b - b_hi;
    scale * (((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo)
}

/// Knuth's two-sum for a difference: returns `(s, e)` with
/// `a - b == s + e` exactly, where `s = a - b` rounded.
#[inline]
pub(crate) fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let s = a - b;
    if !s.is_finite() {
        // the residual is not representable; 0 is the safe choice, and the
        // alternative is `inf - inf == NaN` poisoning every caller
        return (s, 0.0);
    }
    let v = s - a;
    (s, (a - (s - v)) + (-b - v))
}

#[cfg(test)]
mod tests {
    use super::frexp;

    #[test]
    fn frexp_decomposes_zero_and_subnormal_values() {
        assert_eq!(frexp(0.0), (0.0, 0));
        assert_eq!(frexp(-0.0), (-0.0, 0));
        assert_eq!(frexp(f64::from_bits(1)), (0.5, -1073));
        assert_eq!(frexp(f64::from_bits(1).copysign(-1.0)), (-0.5, -1073));
    }
}

macro_rules! redefine_one_opt_approx_macro {
    (
        $approx_macro:ident,
        { epsilon: $default_eps:expr }
    ) => {
        macro_rules! $approx_macro {
            // Caller provides an override for epsilon.
            ($a:expr, $b:expr, epsilon = $user_eps:expr) => {
                approx::$approx_macro!($a, $b, epsilon = $user_eps)
            };
            // No override: use default.
            ($a:expr, $b:expr) => {
                approx::$approx_macro!($a, $b, epsilon = $default_eps)
            };
        }
    };
}

macro_rules! redefine_two_opt_approx_macro {
    (
        $approx_macro:ident,
        { epsilon: $default_eps:expr, $second_key:ident: $default_second:expr }
    ) => {
        macro_rules! $approx_macro {
            // Caller provides both options.
            ($a:expr, $b:expr, epsilon = $user_eps:expr, $second_key = $user_second:expr) => {
                approx::$approx_macro!($a, $b, epsilon = $user_eps, $second_key = $user_second)
            };
            // Caller provides epsilon only; use default for second.
            ($a:expr, $b:expr, epsilon = $user_eps:expr) => {
                approx::$approx_macro!($a, $b, epsilon = $user_eps, $second_key = $default_second)
            };
            // Caller provides the second option only; use default for epsilon.
            ($a:expr, $b:expr, $second_key = $user_second:expr) => {
                approx::$approx_macro!($a, $b, epsilon = $default_eps, $second_key = $user_second)
            };
            // Caller provides neither: use both defaults.
            ($a:expr, $b:expr) => {
                approx::$approx_macro!(
                    $a,
                    $b,
                    epsilon = $default_eps,
                    $second_key = $default_second
                )
            };
        }
    };
}
mod macros {
    pub(crate) use redefine_one_opt_approx_macro;
    pub(crate) use redefine_two_opt_approx_macro;

    // Non-asserting wrappers:
    redefine_one_opt_approx_macro!(
        abs_diff_eq,
        { epsilon: crate::prec::DEFAULT_EPS }
    );
    redefine_two_opt_approx_macro!(
        relative_eq,
        { epsilon: crate::prec::DEFAULT_EPS, max_relative: crate::prec::DEFAULT_RELATIVE_ACC }
    );
    redefine_two_opt_approx_macro!(
        ulps_eq,
        { epsilon: 0.0, max_ulps: crate::prec::DEFAULT_ULPS }
    );

    pub(crate) use abs_diff_eq;
    pub(crate) use relative_eq;
    pub(crate) use ulps_eq;

    // Asserting wrappers:
    redefine_one_opt_approx_macro!(
        assert_abs_diff_eq,
        { epsilon: crate::prec::DEFAULT_EPS }
    );
    redefine_two_opt_approx_macro!(
        assert_relative_eq,
        { epsilon: crate::prec::DEFAULT_EPS, max_relative: crate::prec::DEFAULT_RELATIVE_ACC }
    );
    redefine_two_opt_approx_macro!(
        assert_ulps_eq,
        { epsilon: 0.0, max_ulps: crate::prec::DEFAULT_ULPS }
    );

    pub(crate) use assert_abs_diff_eq;
    pub(crate) use assert_relative_eq;
    pub(crate) use assert_ulps_eq;

    #[deprecated(since = "0.19.0", note = "Use assert_abs_diff_eq! macro instead")]
    macro_rules! assert_almost_eq {
        ($a:expr, $b:expr, $eps:expr $(,)?) => {
            approx::assert_abs_diff_eq!($a, $b, epsilon = $eps)
        };
    }
}

pub(crate) use macros::*;
