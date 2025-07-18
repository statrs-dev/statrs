//! Provides functions for calculating
//! [harmonic](https://en.wikipedia.org/wiki/Harmonic_number)
//! numbers

use crate::consts;
use crate::function::gamma;

/// Computes the `t`-th harmonic number
///
/// # Remarks
///
/// Returns `1` as a special case when `t == 0`
pub fn harmonic(t: u64) -> f64 {
    match t {
        0 => 1.0,
        _ => consts::EULER_MASCHERONI + gamma::digamma(t as f64 + 1.0),
    }
}

/// Computes the generalized harmonic number of  order `n` of `m`
/// e.g. `(1 + 1/2^m + 1/3^m + ... + 1/n^m)`
///
/// # Remarks
///
/// Returns `1` as a special case when `n == 0`
pub fn gen_harmonic(n: u64, m: f64) -> f64 {
    match n {
        0 => 1.0,
        _ => (0..n).fold(0.0, |acc, x| acc + (x as f64 + 1.0).powf(-m)),
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use core::f64;
    use crate::prec;
    use super::*;

    #[test]
    fn test_harmonic() {
        prec::assert_ulps_eq!(harmonic(0), 1.0, max_ulps = 0);
        prec::assert_abs_diff_eq!(harmonic(1), 1.0, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(harmonic(2), 1.5, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(harmonic(4), 2.083333333333333333333, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(harmonic(8), 2.717857142857142857143, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(harmonic(16), 3.380728993228993228993, epsilon = 1e-14);
    }

    #[test]
    fn test_gen_harmonic() {
        assert_eq!(gen_harmonic(0, 0.0), 1.0);
        assert_eq!(gen_harmonic(0, f64::INFINITY), 1.0);
        assert_eq!(gen_harmonic(0, f64::NEG_INFINITY), 1.0);
        assert_eq!(gen_harmonic(1, 0.0), 1.0);
        assert_eq!(gen_harmonic(1, f64::INFINITY), 1.0);
        assert_eq!(gen_harmonic(1, f64::NEG_INFINITY), 1.0);
        assert_eq!(gen_harmonic(2, 1.0), 1.5);
        assert_eq!(gen_harmonic(2, 3.0), 1.125);
        assert_eq!(gen_harmonic(2, f64::INFINITY), 1.0);
        assert_eq!(gen_harmonic(2, f64::NEG_INFINITY), f64::INFINITY);
        prec::assert_abs_diff_eq!(gen_harmonic(4, 1.0), 2.083333333333333333333, epsilon = 1e-14);
        assert_eq!(gen_harmonic(4, 3.0), 1.177662037037037037037);
        assert_eq!(gen_harmonic(4, f64::INFINITY), 1.0);
        assert_eq!(gen_harmonic(4, f64::NEG_INFINITY), f64::INFINITY);
    }
}
