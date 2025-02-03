#![allow(unused_macros, unused_imports)]
//! Provides utility functions for working with floating point precision

/// Standard epsilon, maximum relative precision of IEEE 754 double-precision
/// floating point numbers (64 bit) e.g. `2^-53`
pub const F64_PREC: f64 = 0.00000000000000011102230246251565;

/// Default accuracy for `f64`, equivalent to `0.0 * F64_PREC`
pub const DEFAULT_F64_ACC: f64 = 0.0000000000000011102230246251565;

/// Targeted accuracy over `f64` results used in tests
pub const DEFAULT_RELATIVE_ACC: f64 = 1e-10;
pub const DEFAULT_EPS: f64 = 1e-9;
pub const DEFAULT_ULPS: u32 = 5;

/// Compares if two floats are close via `approx::abs_diff_eq`
/// using a maximum absolute difference (epsilon) of `acc`.
pub fn almost_eq(a: f64, b: f64, acc: f64) -> bool {
    use approx::AbsDiffEq;
    if a.is_infinite() && b.is_infinite() {
        return a == b;
    }
    a.abs_diff_eq(&b, acc)
}

/// Compares if two floats are close via `approx::relative_eq!`
/// and `crate::consts::ACC` relative precision.
/// Updates first argument to value of second argument
pub fn convergence(x: &mut f64, x_new: f64) -> bool {
    let res = relative_eq!(*x, x_new);
    *x = x_new;
    res
}

#[macro_export(local_inner_macros)]
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

#[macro_export(local_inner_macros)]
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

#[macro_use]
pub mod macros {
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
        { epsilon: crate::prec::DEFAULT_EPS, max_ulps: crate::prec::DEFAULT_ULPS }
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
        { epsilon: crate::prec::DEFAULT_EPS, max_ulps: crate::prec::DEFAULT_ULPS }
    );

    pub(crate) use assert_abs_diff_eq;
    pub(crate) use assert_relative_eq;
    pub(crate) use assert_ulps_eq;

    #[deprecated = "phasing this macro out from internal testing for consistency"]
    macro_rules! assert_almost_eq {
        ($a:expr, $b:expr, $eps:expr $(,)?) => {
            approx::assert_abs_diff_eq!($a, $b, epsilon = $eps)
        };
    }

    pub(crate) use assert_almost_eq;
}

pub use macros::*;
