//! Provides functions related to factorial calculations (e.g. binomial
//! coefficient, factorial, multinomial)

use crate::function::gamma;

/// The maximum factorial representable
/// by a 64-bit floating point without
/// overflowing
pub const MAX_FACTORIAL: usize = 170;

/// Computes the factorial function `x -> x!` for
/// `170 >= x >= 0`. All factorials larger than `170!`
/// will overflow an `f64`.
///
/// # Remarks
///
/// Returns `f64::INFINITY` if `x > 170`
pub fn factorial(x: u64) -> f64 {
    let x = x as usize;
    FCACHE.get(x).map_or(f64::INFINITY, |&fac| fac)
}

/// Computes the logarithmic factorial function `x -> ln(x!)`
/// for `x >= 0`.
///
/// # Remarks
///
/// Returns `0.0` if `x <= 1`
pub fn ln_factorial(x: u64) -> f64 {
    let x = x as usize;
    FCACHE
        .get(x)
        .map_or_else(|| gamma::ln_gamma(x as f64 + 1.0), |&fac| fac.ln())
}

/// Computes the binomial coefficient `n choose k`
/// where `k` and `n` are non-negative values.
///
/// # Remarks
///
/// Returns `0.0` if `k > n`
pub fn binomial(n: u64, k: u64) -> f64 {
    if k > n {
        0.0
    } else {
        (0.5 + (ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)).exp()).floor()
    }
}

/// Computes the natural logarithm of the binomial coefficient
/// `ln(n choose k)` where `k` and `n` are non-negative values
///
/// # Remarks
///
/// Returns `f64::NEG_INFINITY` if `k > n`
pub fn ln_binomial(n: u64, k: u64) -> f64 {
    if k > n {
        f64::NEG_INFINITY
    } else {
        ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
    }
}

/// Computes the multinomial coefficient: `n choose n1, n2, n3, ...`
///
/// # Panics
///
/// If the elements in `ni` do not sum to `n`
pub fn multinomial(n: u64, ni: &[u64]) -> f64 {
    checked_multinomial(n, ni).unwrap()
}

/// Computes the multinomial coefficient: `n choose n1, n2, n3, ...`
///
/// Returns `None` if the elements in `ni` do not sum to `n`.
pub fn checked_multinomial(n: u64, ni: &[u64]) -> Option<f64> {
    let (sum, ret) = ni.iter().fold((0, ln_factorial(n)), |acc, &x| {
        (acc.0 + x, acc.1 - ln_factorial(x))
    });

    if sum == n {
        Some((0.5 + ret.exp()).floor())
    } else {
        None
    }
}

// Initialization for pre-computed cache of 171 factorial
// values 0!...170!
const FCACHE: [f64; MAX_FACTORIAL + 1] = {
    let mut fcache = [1.0; MAX_FACTORIAL + 1];

    // `const` only allow while loops
    let mut i = 1;
    while i < MAX_FACTORIAL + 1 {
        fcache[i] = fcache[i - 1] * i as f64;
        i += 1;
    }

    fcache
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fcache() {
        assert!((FCACHE[0] - 1.0).abs() < f64::EPSILON);
        assert!((FCACHE[1] - 1.0).abs() < f64::EPSILON);
        assert!((FCACHE[2] - 2.0).abs() < f64::EPSILON);
        assert!((FCACHE[3] - 6.0).abs() < f64::EPSILON);
        assert!((FCACHE[4] - 24.0).abs() < f64::EPSILON);
        assert!((FCACHE[70] - 1197857166996989e85).abs() < f64::EPSILON);
        assert!((FCACHE[170] - 7257415615307994e291).abs() < f64::EPSILON);
    }

    #[test]
    fn test_factorial_and_ln_factorial() {
        let mut fac = 1.0;
        assert_eq!(factorial(0), fac);
        for i in 1..171 {
            fac *= i as f64;
            assert_eq!(factorial(i), fac);
            assert_eq!(ln_factorial(i), fac.ln());
        }
    }

    #[test]
    fn test_factorial_overflow() {
        assert_eq!(factorial(172), f64::INFINITY);
        assert_eq!(factorial(u64::MAX), f64::INFINITY);
    }

    #[test]
    fn test_ln_factorial_does_not_overflow() {
        assert_eq!(ln_factorial(1 << 10), 6078.2118847500501140);
        assert_almost_eq!(ln_factorial(1 << 12), 29978.648060844048236, 1e-11);
        assert_eq!(ln_factorial(1 << 15), 307933.81973375485425);
        assert_eq!(ln_factorial(1 << 17), 1413421.9939462073242);
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(1, 1), 1.0);
        assert_eq!(binomial(5, 2), 10.0);
        assert_eq!(binomial(7, 3), 35.0);
        assert_eq!(binomial(1, 0), 1.0);
        assert_eq!(binomial(0, 1), 0.0);
        assert_eq!(binomial(5, 7), 0.0);
    }

    #[test]
    fn test_ln_binomial() {
        assert_eq!(ln_binomial(1, 1), 1f64.ln());
        assert_almost_eq!(ln_binomial(5, 2), 10f64.ln(), 1e-14);
        assert_almost_eq!(ln_binomial(7, 3), 35f64.ln(), 1e-14);
        assert_eq!(ln_binomial(1, 0), 1f64.ln());
        assert_eq!(ln_binomial(0, 1), 0f64.ln());
        assert_eq!(ln_binomial(5, 7), 0f64.ln());
    }

    #[test]
    fn test_multinomial() {
        assert_eq!(1.0, multinomial(1, &[1, 0]));
        assert_eq!(10.0, multinomial(5, &[3, 2]));
        assert_eq!(10.0, multinomial(5, &[2, 3]));
        assert_eq!(35.0, multinomial(7, &[3, 4]));
    }

    #[test]
    #[should_panic]
    fn test_multinomial_bad_ni() {
        multinomial(1, &[1, 1]);
    }

    #[test]
    fn test_checked_multinomial_bad_ni() {
        assert!(checked_multinomial(1, &[1, 1]).is_none());
    }
}
