//! Provides functions related to factorial calculations (e.g. binomial
//! coefficient, factorial, multinomial)

use crate::function::gamma;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

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
/// Returns `0.0` if `k > n`.
///
/// The result is exact whenever `C(n, k)` fits in a `u128`, and correctly
/// rounded to the nearest `f64` beyond `2^53`. Larger coefficients fall back
/// to `exp(ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k))`, which
/// is accurate to about `1e-12` relative.
pub fn binomial(n: u64, k: u64) -> f64 {
    if k > n {
        return 0.0;
    }
    match binomial_u128(n, k) {
        Some(exact) => exact as f64,
        // Overflow: the result exceeds ~3.4e38 / n, far past 2^53, so
        // exactness is unattainable anyway; fall back to logs.
        None => (ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k))
            .exp()
            .round(),
    }
}

/// Computes `C(n, k)` exactly in integer arithmetic, or `None` if an
/// intermediate value overflows a `u128`. Requires `k <= n`.
///
/// `C(n, i + 1) = C(n, i) * (n - i) / (i + 1)`, and the division is exact at
/// every step, so the whole computation stays in integer arithmetic. The
/// previous `f64` implementation rounded `exp(ln n! - ln k! - ln (n-k)!)` to
/// the nearest integer, which was off by up to ~1e5 (57 ulp) for coefficients
/// near 2^63 that f64 could still represent to full precision.
fn binomial_u128(n: u64, k: u64) -> Option<u128> {
    // C(n, k) == C(n, n - k); walking up the smaller side minimises both the
    // iteration count and the chance of overflowing the integer path.
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k {
        acc = acc.checked_mul((n - i) as u128)? / (i as u128 + 1);
    }
    Some(acc)
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
///
/// # Remarks
///
/// The result is exact whenever the coefficient (and every intermediate
/// prefix product) fits in a `u128`, and correctly rounded to the nearest
/// `f64` beyond `2^53`. Larger coefficients fall back to
/// `exp(ln n! - sum ln ni!)`, accurate to about `1e-12` relative.
pub fn checked_multinomial(n: u64, ni: &[u64]) -> Option<f64> {
    if ni.iter().sum::<u64>() != n {
        return None;
    }

    // n! / (n1! n2! ... nk!) == prod_i C(s_i, n_i) with s_i = n_1 + ... + n_i:
    // a product of binomial coefficients, each computed exactly.
    let mut acc: u128 = 1;
    let mut prefix: u64 = 0;
    for &k in ni {
        prefix += k;
        let Some(product) = binomial_u128(prefix, k).and_then(|c| acc.checked_mul(c)) else {
            // Overflow: fall back to logs (the old implementation's only path).
            let ret = ni.iter().fold(ln_factorial(n), |a, &x| a - ln_factorial(x));
            return Some(ret.exp().round());
        };
        acc = product;
    }
    Some(acc as f64)
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

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::prec;

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
        prec::assert_abs_diff_eq!(
            ln_factorial(1 << 12),
            29978.648060844048236,
            epsilon = 1e-11
        );
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

    /// Every `C(n, k)` that fits in a `u128` must round-trip exactly (the
    /// conversion `u128 -> f64` is correctly rounded, so `expected as f64` is
    /// the best possible double). The old `exp(ln ...)`-based implementation
    /// failed this for e.g. `C(50, 25)` (off by 2) and `C(67, 33)` (off by
    /// 116736).
    #[test]
    fn test_binomial_is_exact_where_representable() {
        for n in 0..=170u64 {
            let mut expected: u128 = 1;
            for k in 0..=n / 2 {
                assert_eq!(
                    binomial(n, k),
                    expected as f64,
                    "C({n}, {k}) should be {expected}"
                );
                assert_eq!(binomial(n, n - k), expected as f64, "C({n}, {}) symmetry", n - k);
                let Some(product) = expected.checked_mul((n - k) as u128) else {
                    break;
                };
                expected = product / (k as u128 + 1);
            }
        }
    }

    /// Coefficients too large for the integer path fall back to logs; check the
    /// fallback is close and consistent with `ln_binomial`.
    #[test]
    fn test_binomial_log_fallback() {
        // C(200, 100) = 9.0548514656103281165404177077e58 (overflows u128)
        prec::assert_relative_eq!(
            binomial(200, 100),
            9.0548514656103281165e58,
            epsilon = 0.0,
            max_relative = 1e-11
        );
        prec::assert_relative_eq!(
            binomial(1000, 500),
            ln_binomial(1000, 500).exp(),
            epsilon = 0.0,
            max_relative = 1e-11
        );
    }

    #[test]
    fn test_ln_binomial() {
        assert_eq!(ln_binomial(1, 1), 1f64.ln());
        prec::assert_abs_diff_eq!(ln_binomial(5, 2), 10f64.ln(), epsilon = 1e-14);
        prec::assert_abs_diff_eq!(ln_binomial(7, 3), 35f64.ln(), epsilon = 1e-14);
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

    /// A two-part multinomial is a binomial coefficient; a three-part one has
    /// the closed form `C(n, a) * C(n - a, b)`. Both must be exact where
    /// representable (mirrors `test_binomial_is_exact_where_representable`).
    #[test]
    fn test_multinomial_is_exact_where_representable() {
        for n in 0..=170u64 {
            for k in 0..=n / 2 {
                assert_eq!(multinomial(n, &[k, n - k]), binomial(n, k), "n={n} k={k}");
            }
        }
        // 60! / (20!)^3 = 577831214478475823831865900 (fits u128); the
        // conversion `u128 -> f64` is correctly rounded:
        assert_eq!(
            multinomial(60, &[20, 20, 20]),
            577831214478475823831865900_u128 as f64
        );
    }

    #[test]
    fn test_multinomial_log_fallback() {
        // 300! / (100!)^3 overflows u128; check against ln-space value
        let ln_ref = ln_factorial(300) - 3.0 * ln_factorial(100);
        prec::assert_relative_eq!(
            multinomial(300, &[100, 100, 100]),
            ln_ref.exp(),
            epsilon = 0.0,
            max_relative = 1e-11
        );
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
