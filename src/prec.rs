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
//!    should be close in terms of floating-point representation
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
