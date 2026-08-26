//! Provides the [beta](https://en.wikipedia.org/wiki/Beta_function) and related
//! function
//!
//! This module sets the default precision more tightly than crate defaults for `DEFAULT_EPS`

mod double_double;
mod large_params;
mod temme;

use crate::function::gamma;
use crate::prec;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// sample case of module level precision
#[cfg(test)]
const MODULE_EPS: f64 = 1e-15;

/// Represents the errors that can occur when computing the natural logarithm
/// of the beta function or the regularized lower incomplete beta function.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum BetaFuncError {
    /// `a` is zero or less than zero.
    ANotGreaterThanZero,

    /// `b` is zero or less than zero.
    BNotGreaterThanZero,

    /// `x` is not in `[0, 1]`.
    XOutOfRange,

    /// The numerical method did not converge.
    ConvergenceFailed,
}

impl core::fmt::Display for BetaFuncError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            BetaFuncError::ANotGreaterThanZero => write!(f, "a is zero or less than zero"),
            BetaFuncError::BNotGreaterThanZero => write!(f, "b is zero or less than zero"),
            BetaFuncError::XOutOfRange => write!(f, "x is not in [0, 1]"),
            BetaFuncError::ConvergenceFailed => write!(f, "computation did not converge"),
        }
    }
}

impl core::error::Error for BetaFuncError {}

/// Computes the natural logarithm
/// of the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter
/// and `a > 0`, `b > 0`.
///
/// # Panics
///
/// if `a <= 0.0` or `b <= 0.0`
pub fn ln_beta(a: f64, b: f64) -> f64 {
    checked_ln_beta(a, b).unwrap()
}

/// Computes the natural logarithm
/// of the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter
/// and `a > 0`, `b > 0`.
///
/// # Errors
///
/// if `a <= 0.0` or `b <= 0.0`
pub fn checked_ln_beta(a: f64, b: f64) -> Result<f64, BetaFuncError> {
    if a <= 0.0 {
        Err(BetaFuncError::ANotGreaterThanZero)
    } else if b <= 0.0 {
        Err(BetaFuncError::BNotGreaterThanZero)
    } else {
        Ok(gamma::ln_gamma(a) + gamma::ln_gamma(b) - gamma::ln_gamma(a + b))
    }
}

/// Computes the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter.
///
///
/// # Panics
///
/// if `a <= 0.0` or `b <= 0.0`
pub fn beta(a: f64, b: f64) -> f64 {
    checked_beta(a, b).unwrap()
}

/// Computes the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter.
///
///
/// # Errors
///
/// if `a <= 0.0` or `b <= 0.0`
pub fn checked_beta(a: f64, b: f64) -> Result<f64, BetaFuncError> {
    checked_ln_beta(a, b).map(|x| x.exp())
}

/// Computes the lower incomplete (unregularized) beta function
/// `B(a,b,x) = int(t^(a-1)*(1-t)^(b-1),t=0..x)` for `a > 0, b > 0, 1 >= x >= 0`
/// where `a` is the first beta parameter, `b` is the second beta parameter, and
/// `x` is the upper limit of the integral
///
/// # Panics
///
/// If `a <= 0.0`, `b <= 0.0`, `x < 0.0`, `x > 1.0`, or the numerical method
/// does not converge.
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    checked_beta_inc(a, b, x).unwrap()
}

/// Computes the lower incomplete (unregularized) beta function
/// `B(a,b,x) = int(t^(a-1)*(1-t)^(b-1),t=0..x)` for `a > 0, b > 0, 1 >= x >= 0`
/// where `a` is the first beta parameter, `b` is the second beta parameter, and
/// `x` is the upper limit of the integral
///
/// # Errors
///
/// If `a <= 0.0`, `b <= 0.0`, `x < 0.0`, `x > 1.0`, or the numerical method
/// does not converge.
pub fn checked_beta_inc(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    checked_beta_reg(a, b, x).and_then(|x| checked_beta(a, b).map(|y| x * y))
}

/// Computes the regularized lower incomplete beta function
/// `I_x(a,b) = 1/Beta(a,b) * int(t^(a-1)*(1-t)^(b-1), t=0..x)`
/// `a > 0`, `b > 0`, `1 >= x >= 0` where `a` is the first beta parameter,
/// `b` is the second beta parameter, and `x` is the upper limit of the
/// integral.
///
/// # Panics
///
/// If `a <= 0.0`, `b <= 0.0`, `x < 0.0`, `x > 1.0`, or the numerical method
/// does not converge.
pub fn beta_reg(a: f64, b: f64, x: f64) -> f64 {
    checked_beta_reg(a, b, x).unwrap()
}

fn beta_reg_use_complement(a: f64, b: f64, x: f64) -> bool {
    let denominator = a + b + 2.0;
    if denominator.is_finite() {
        return x >= (a + 1.0) / denominator;
    }
    let scale = a.max(b);
    let inverse_scale = 1.0 / scale;
    let scaled_a = a / scale;
    let scaled_b = b / scale;
    x >= (scaled_a + inverse_scale) / (scaled_a + scaled_b + 2.0 * inverse_scale)
}

fn beta_reg_symmetric_central(a: f64, b: f64, x: f64) -> Option<f64> {
    if a != b || a < 100.0 {
        return None;
    }

    let delta = x - 0.5;
    let delta_squared = delta * delta;
    if 4.0 * delta_squared * a > 0.5 {
        return None;
    }

    let inverse_a = 1.0 / a;
    let mut gamma_ratio: f64 = 869.0 / 4_194_304.0;
    for coefficient in [
        -399.0 / 262_144.0,
        -21.0 / 32_768.0,
        5.0 / 1_024.0,
        1.0 / 128.0,
        -1.0 / 8.0,
        1.0,
    ] {
        gamma_ratio = gamma_ratio.mul_add(inverse_a, coefficient);
    }
    let central_density = 2.0 * (a / core::f64::consts::PI).sqrt() * gamma_ratio;

    let mut term = delta;
    let mut integral = term;
    for index in 1..=32 {
        let n = f64::from(index);
        term *= -4.0 * delta_squared * (a - n) * (2.0 * n - 1.0) / (n * (2.0 * n + 1.0));
        let previous = integral;
        integral += term;
        if integral == previous {
            break;
        }
    }
    Some(central_density.mul_add(integral, 0.5))
}

fn beta_continued_fraction(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    let eps = prec::F64_PREC;
    let fpmin = f64::MIN_POSITIVE / eps;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;

    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..100_001 {
        let m = f64::from(m);
        let m2 = m * 2.0;
        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;

        if d.abs() < fpmin {
            d = fpmin;
        }

        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }

        d = 1.0 / d;
        h = h * d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;

        if d.abs() < fpmin {
            d = fpmin;
        }

        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }

        d = 1.0 / d;
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() <= eps {
            return Ok(h);
        }
    }

    Err(BetaFuncError::ConvergenceFailed)
}

fn beta_reg_from_fraction(log_prefactor: (f64, f64), fraction: f64, a: f64) -> f64 {
    let quotient = fraction / a;
    let log_quotient = if quotient.is_finite() && quotient > 0.0 {
        quotient.ln()
    } else {
        fraction.ln() - a.ln()
    };
    double_double::exp(double_double::add(log_prefactor, (log_quotient, 0.0)))
}

/// Computes the regularized lower incomplete beta function
/// `I_x(a,b) = 1/Beta(a,b) * int(t^(a-1)*(1-t)^(b-1), t=0..x)`
/// `a > 0`, `b > 0`, `1 >= x >= 0` where `a` is the first beta parameter,
/// `b` is the second beta parameter, and `x` is the upper limit of the
/// integral.
///
/// # Errors
///
/// If `a <= 0.0`, `b <= 0.0`, `x < 0.0`, `x > 1.0`, or the numerical method
/// does not converge.
pub fn checked_beta_reg(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    if a <= 0.0 {
        return Err(BetaFuncError::ANotGreaterThanZero);
    }

    if b <= 0.0 {
        return Err(BetaFuncError::BNotGreaterThanZero);
    }

    if !(0.0..=1.0).contains(&x) {
        return Err(BetaFuncError::XOutOfRange);
    }

    if let Some(result) = beta_reg_symmetric_central(a, b, x) {
        return Ok(result);
    }

    if let Some(result) = temme::beta_reg_temme(a, b, x) {
        return Ok(result);
    }

    if x == 0.0 || x == 1.0 {
        return Ok(x);
    }

    let log_prefactor = match large_params::log_prefactor(a, b, x) {
        Some(large_params::LogPrefactor::Value(logarithm)) => logarithm,
        Some(large_params::LogPrefactor::Underflow) => {
            return Ok(if beta_reg_use_complement(a, b, x) {
                1.0
            } else {
                0.0
            });
        }
        None => (
            gamma::ln_gamma(a + b) - gamma::ln_gamma(a) - gamma::ln_gamma(b)
                + a * x.ln()
                + b * (-x).ln_1p(),
            0.0,
        ),
    };

    if !beta_reg_use_complement(a, b, x) {
        let fraction = beta_continued_fraction(a, b, x)?;
        return Ok(beta_reg_from_fraction(log_prefactor, fraction, a));
    }

    let complement_fraction = beta_continued_fraction(b, a, 1.0 - x)?;
    let complement = beta_reg_from_fraction(log_prefactor, complement_fraction, b);
    let result = 1.0 - complement;
    if result > f64::EPSILON.sqrt() && result.is_finite() {
        return Ok(result);
    }

    let fraction = beta_continued_fraction(a, b, x)?;
    Ok(beta_reg_from_fraction(log_prefactor, fraction, a))
}

/// Computes the inverse of the regularized incomplete beta function
// This code is based on the implementation in the ["special"][1] crate,
// which in turn is based on a [C implementation][2] by John Burkardt. The
// original algorithm was published in Applied Statistics and is known as
// [Algorithm AS 64][3] and [Algorithm AS 109][4].
//
// [1]: https://docs.rs/special/0.8.1/
// [2]: http://people.sc.fsu.edu/~jburkardt/c_src/asa109/asa109.html
// [3]: http://www.jstor.org/stable/2346798
// [4]: http://www.jstor.org/stable/2346887
//
// > Copyright 2014–2019 The special Developers
// >
// > Permission is hereby granted, free of charge, to any person obtaining a copy of
// > this software and associated documentation files (the "Software"), to deal in
// > the Software without restriction, including without limitation the rights to
// > use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// > the Software, and to permit persons to whom the Software is furnished to do so,
// > subject to the following conditions:
// >
// > The above copyright notice and this permission notice shall be included in all
// > copies or substantial portions of the Software.
// >
// > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// > IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// > FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// > COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// > IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// > CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
pub fn inv_beta_reg(mut a: f64, mut b: f64, mut x: f64) -> f64 {
    // Algorithm AS 64
    // http://www.jstor.org/stable/2346798
    //
    // An approximation x₀ to x if found from (cf. Scheffé and Tukey, 1944)
    //
    // 1 + x₀   4p + 2q - 2
    // ------ = -----------
    // 1 - x₀      χ²(α)
    //
    // where χ²(α) is the upper α point of the χ² distribution with 2q
    // degrees of freedom and is obtained from Wilson and Hilferty's
    // approximation (cf. Wilson and Hilferty, 1931)
    //
    // χ²(α) = 2q (1 - 1 / (9q) + y(α) sqrt(1 / (9q)))^3,
    //
    // y(α) being Hastings' approximation (cf. Hastings, 1955) for the upper
    // α point of the standard normal distribution. If χ²(α) < 0, then
    //
    // x₀ = 1 - ((1 - α)q B(p, q))^(1 / q).
    //
    // Again if (4p + 2q - 2) / χ²(α) does not exceed 1, x₀ is obtained from
    //
    // x₀ = (αp B(p, q))^(1 / p).
    //
    // The final solution is obtained by the Newton–Raphson method from the
    // relation
    //
    //                    f(x[i - 1])
    // x[i] = x[i - 1] - ------------
    //                   f'(x[i - 1])
    //
    // where
    //
    // f(x) = I(x, p, q) - α.
    let ln_beta = ln_beta(a, b);

    // Remark AS R83
    // http://www.jstor.org/stable/2347779
    const SAE: i32 = -30;
    const FPU: f64 = 1e-30; // 10^SAE

    debug_assert!((0.0..=1.0).contains(&x) && a > 0.0 && b > 0.0);

    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }

    let mut p;
    let mut q;

    let flip = 0.5 < x;
    if flip {
        p = a;
        a = b;
        b = p;
        x = 1.0 - x;
    }

    p = (-(x * x).ln()).sqrt();
    q = p - (2.30753 + 0.27061 * p) / (1.0 + (0.99229 + 0.04481 * p) * p);

    if 1.0 < a && 1.0 < b {
        // Remark AS R19 and Algorithm AS 109
        // http://www.jstor.org/stable/2346887
        //
        // For a and b > 1, the approximation given by Carter (1947), which
        // improves the Fisher–Cochran formula, is generally better. For
        // other values of a and b en empirical investigation has shown that
        // the approximation given in AS 64 is adequate.
        let r = (q * q - 3.0) / 6.0;
        let s = 1.0 / (2.0 * a - 1.0);
        let t = 1.0 / (2.0 * b - 1.0);
        let h = 2.0 / (s + t);
        let w = q * (h + r).sqrt() / h - (t - s) * (r + 5.0 / 6.0 - 2.0 / (3.0 * h));
        p = a / (a + b * (2.0 * w).exp());
    } else {
        let mut t = 1.0 / (9.0 * b);
        t = 2.0 * b * (1.0 - t + q * t.sqrt()).powf(3.0);
        if t <= 0.0 {
            p = 1.0 - ((((1.0 - x) * b).ln() + ln_beta) / b).exp();
        } else {
            t = 2.0 * (2.0 * a + b - 1.0) / t;
            if t <= 1.0 {
                p = (((x * a).ln() + ln_beta) / a).exp();
            } else {
                p = 1.0 - 2.0 / (t + 1.0);
            }
        }
    }

    p = p.clamp(0.0001, 0.9999);

    // Remark AS R83
    // http://www.jstor.org/stable/2347779
    let e = (-5.0 / a / a - 1.0 / x.powf(0.2) - 13.0) as i32;
    let acu = if e > SAE { f64::powi(10.0, e) } else { FPU };

    let mut pnext;
    let mut qprev = 0.0;
    let mut sq = 1.0;
    let mut prev = 1.0;

    'outer: loop {
        // Remark AS R19 and Algorithm AS 109
        // http://www.jstor.org/stable/2346887
        q = beta_reg(a, b, p);
        q = (q - x) * (ln_beta + (1.0 - a) * p.ln() + (1.0 - b) * (1.0 - p).ln()).exp();

        // Remark AS R83
        // http://www.jstor.org/stable/2347779
        if q * qprev <= 0.0 {
            prev = if sq > FPU { sq } else { FPU };
        }

        // Remark AS R19 and Algorithm AS 109
        // http://www.jstor.org/stable/2346887
        let mut g = 1.0;
        loop {
            loop {
                let adj = g * q;
                sq = adj * adj;

                if sq < prev {
                    pnext = p - adj;
                    if (0.0..=1.0).contains(&pnext) {
                        break;
                    }
                }
                g /= 3.0;
            }

            if prev <= acu || q * q <= acu {
                p = pnext;
                break 'outer;
            }

            if pnext != 0.0 && pnext != 1.0 {
                break;
            }

            g /= 3.0;
        }

        if pnext == p {
            break;
        }

        p = pnext;
        qprev = q;
    }

    if flip { 1.0 - p } else { p }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prec;
    use core::f64::consts as f64_consts;
    const MODULE_RELATIVE_ACC: f64 = 1e-14;

    fn beta_assert_relative_eq(a: f64, b: f64) {
        prec::assert_relative_eq!(
            a,
            b,
            epsilon = MODULE_EPS,
            max_relative = MODULE_RELATIVE_ACC
        );
    }

    fn beta_assert_abs_diff_eq(a: f64, b: f64) {
        prec::assert_abs_diff_eq!(a, b, epsilon = MODULE_EPS);
    }

    #[test]
    fn test_ln_beta() {
        beta_assert_relative_eq(ln_beta(0.5, 0.5), 1.144729885849400174144);
        beta_assert_relative_eq(ln_beta(1.0, 0.5), f64_consts::LN_2);
        beta_assert_relative_eq(ln_beta(2.5, 0.5), 0.163900632837673937284);
        beta_assert_relative_eq(ln_beta(0.5, 1.0), f64_consts::LN_2);
        beta_assert_relative_eq(ln_beta(1.0, 1.0), 0.0);
        beta_assert_relative_eq(ln_beta(2.5, 1.0), -0.9162907318741550651835);
        beta_assert_relative_eq(ln_beta(0.5, 2.5), 0.163900632837673937284);
        beta_assert_relative_eq(ln_beta(1.0, 2.5), -0.9162907318741550651835);
        beta_assert_relative_eq(ln_beta(2.5, 2.5), -2.608688089402107300388);
    }

    #[test]
    #[should_panic]
    fn test_ln_beta_a_lte_0() {
        ln_beta(0.0, 0.5);
    }

    #[test]
    #[should_panic]
    fn test_ln_beta_b_lte_0() {
        ln_beta(0.5, 0.0);
    }

    #[test]
    fn test_checked_ln_beta_a_lte_0() {
        assert!(checked_ln_beta(0.0, 0.5).is_err());
    }

    #[test]
    fn test_checked_ln_beta_b_lte_0() {
        assert!(checked_ln_beta(0.5, 0.0).is_err());
    }

    #[test]
    #[should_panic]
    fn test_beta_a_lte_0() {
        beta(0.0, 0.5);
    }

    #[test]
    #[should_panic]
    fn test_beta_b_lte_0() {
        beta(0.5, 0.0);
    }

    #[test]
    fn test_checked_beta_a_lte_0() {
        assert!(checked_beta(0.0, 0.5).is_err());
    }

    #[test]
    fn test_checked_beta_b_lte_0() {
        assert!(checked_beta(0.5, 0.0).is_err());
    }

    #[test]
    fn test_beta() {
        beta_assert_relative_eq(beta(0.5, 0.5), f64_consts::PI);
        beta_assert_relative_eq(beta(1.0, 0.5), 2.0);
        beta_assert_relative_eq(beta(2.5, 0.5), 1.17809724509617246442);
        beta_assert_relative_eq(beta(0.5, 1.0), 2.0);
        beta_assert_relative_eq(beta(1.0, 1.0), 1.0);
        beta_assert_relative_eq(beta(2.5, 1.0), 0.4);
        beta_assert_relative_eq(beta(0.5, 2.5), 1.17809724509617246442);
        beta_assert_relative_eq(beta(1.0, 2.5), 0.4);
        beta_assert_relative_eq(beta(2.5, 2.5), 0.073631077818510779026);
    }

    #[test]
    fn test_beta_inc() {
        beta_assert_relative_eq(beta_inc(0.5, 0.5, 0.5), f64_consts::FRAC_PI_2);
        beta_assert_relative_eq(beta_inc(0.5, 0.5, 1.0), f64_consts::PI);
        beta_assert_relative_eq(beta_inc(1.0, 0.5, 0.5), 0.5857864376269049511983);
        beta_assert_relative_eq(beta_inc(1.0, 0.5, 1.0), 2.0);
        beta_assert_relative_eq(beta_inc(2.5, 0.5, 0.5), 0.0890486225480862322117);
        beta_assert_relative_eq(beta_inc(2.5, 0.5, 1.0), 1.17809724509617246442);
        beta_assert_relative_eq(beta_inc(0.5, 1.0, 0.5), f64_consts::SQRT_2);
        beta_assert_relative_eq(beta_inc(0.5, 1.0, 1.0), 2.0);
        beta_assert_relative_eq(beta_inc(1.0, 1.0, 0.5), 0.5);
        beta_assert_relative_eq(beta_inc(1.0, 1.0, 1.0), 1.0);
        beta_assert_relative_eq(beta_inc(2.5, 1.0, 0.5), 0.0707106781186547524401);
        beta_assert_relative_eq(beta_inc(2.5, 1.0, 1.0), 0.4);
        beta_assert_relative_eq(beta_inc(0.5, 2.5, 0.5), 1.08904862254808623221);
        beta_assert_relative_eq(beta_inc(0.5, 2.5, 1.0), 1.17809724509617246442);
        beta_assert_relative_eq(beta_inc(1.0, 2.5, 0.5), 0.32928932188134524756);
        beta_assert_relative_eq(beta_inc(1.0, 2.5, 1.0), 0.4);
        beta_assert_relative_eq(beta_inc(2.5, 2.5, 0.5), 0.03681553890925538951323);
        beta_assert_relative_eq(beta_inc(2.5, 2.5, 1.0), 0.073631077818510779026);
    }

    #[test]
    #[should_panic]
    fn test_beta_inc_a_lte_0() {
        beta_inc(0.0, 1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_inc_b_lte_0() {
        beta_inc(1.0, 0.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_inc_x_lt_0() {
        beta_inc(1.0, 1.0, -1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_inc_x_gt_1() {
        beta_inc(1.0, 1.0, 2.0);
    }

    #[test]
    fn test_checked_beta_inc_a_lte_0() {
        assert!(checked_beta_inc(0.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_beta_inc_b_lte_0() {
        assert!(checked_beta_inc(1.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_beta_inc_x_lt_0() {
        assert!(checked_beta_inc(1.0, 1.0, -1.0).is_err());
    }

    #[test]
    fn test_checked_beta_inc_x_gt_1() {
        assert!(checked_beta_inc(1.0, 1.0, 2.0).is_err());
    }

    #[test]
    fn test_beta_reg() {
        beta_assert_abs_diff_eq(beta_reg(0.5, 0.5, 0.5), 0.5);
        assert_eq!(beta_reg(0.5, 0.5, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(1.0, 0.5, 0.5), 0.292893218813452475599);
        assert_eq!(beta_reg(1.0, 0.5, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(2.5, 0.5, 0.5), 0.07558681842161243795);
        assert_eq!(beta_reg(2.5, 0.5, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(0.5, 1.0, 0.5), f64_consts::FRAC_1_SQRT_2);
        assert_eq!(beta_reg(0.5, 1.0, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(1.0, 1.0, 0.5), 0.5);
        assert_eq!(beta_reg(1.0, 1.0, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(2.5, 1.0, 0.5), 0.1767766952966368811);
        assert_eq!(beta_reg(2.5, 1.0, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(0.5, 2.5, 0.5), 0.92441318157838756205);
        assert_eq!(beta_reg(0.5, 2.5, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(1.0, 2.5, 0.5), 0.8232233047033631189);
        assert_eq!(beta_reg(1.0, 2.5, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(2.5, 2.5, 0.5), 0.5);
        assert_eq!(beta_reg(2.5, 2.5, 1.0), 1.0);
    }

    #[test]
    fn test_beta_reg_large_symmetric_center() {
        for shape in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e308] {
            assert_eq!(beta_reg(shape, shape, 0.5), 0.5);
        }
    }

    #[test]
    fn test_beta_reg_large_shape_boundaries() {
        assert_eq!(beta_reg(1e8, 2e8, 0.0), 0.0);
        assert_eq!(beta_reg(1e8, 2e8, 1.0), 1.0);
    }

    #[test]
    fn test_beta_reg_next_down_from_one_is_not_a_boundary() {
        let x = f64::from_bits(1.0_f64.to_bits() - 1);
        let expected = 0.30744526594453764_f64;
        let actual = beta_reg(1.0, 0.01, x);
        assert!(
            actual.to_bits().abs_diff(expected.to_bits()) <= 16,
            "actual={actual:?}, expected={expected:?}"
        );
    }

    #[test]
    fn test_beta_reg_preserves_extreme_asymmetric_lower_tail() {
        let a = 1e8;
        let x = (a + 1.0) / (a + 2.0);
        let expected = 2.193839407155793e-301;
        let actual = beta_reg(a, 1e-300, x);
        assert!(
            ((actual - expected) / expected).abs() <= 1e-8,
            "actual={actual:?}, expected={expected:?}"
        );

        let expected = f64::from_bits(0x1bc);
        let actual = beta_reg(a, 1e-320, x);
        assert!(
            actual.to_bits().abs_diff(expected.to_bits()) <= 4,
            "actual={actual:?}, expected={expected:?}"
        );
    }

    #[test]
    fn test_beta_reg_extreme_shapes_and_smallest_x() {
        let x = f64::from_bits(1);
        assert_eq!(checked_beta_reg(1e308, 1e308, x), Ok(0.0));
    }

    #[test]
    fn test_beta_reg_large_symmetric_adjacent_to_center() {
        let lower = f64::from_bits(0.5_f64.to_bits() - 1);
        let upper = f64::from_bits(0.5_f64.to_bits() + 1);
        let cases = [
            (1e2, 0x3fdffffffffffff5_u64, 0x3fe000000000000b_u64),
            (1e4, 0x3fdfffffffffff8f, 0x3fe0000000000071),
            (1e6, 0x3fdffffffffffb98, 0x3fe0000000000468),
            (1e8, 0x3fdfffffffffd3ec, 0x3fe0000000002c14),
        ];
        for (shape, lower_expected, upper_expected) in cases {
            for (x, expected) in [(lower, lower_expected), (upper, upper_expected)] {
                let actual = beta_reg(shape, shape, x).to_bits();
                assert!(
                    actual.abs_diff(expected) <= 4,
                    "shape={shape:?}, x={x:?}, actual={actual:#018x}, expected={expected:#018x}"
                );
            }
        }
    }

    #[test]
    fn test_beta_reg_large_symmetric_central_range() {
        let cases = [
            (
                100.0,
                0x3fddbcbcf5c0139f_u64,
                0x3fc44eca5b83b728_u64,
                0x3fe121a1851ff630_u64,
                0x3feaec4d691f1233_u64,
            ),
            (
                1e6,
                0x3fdffa3516f00033,
                0x3fc44ed0bb7cb51c,
                0x3fe002e57487ffe6,
                0x3feaec4bd120d163,
            ),
        ];
        for (shape, lower_x, lower_expected, upper_x, upper_expected) in cases {
            for (x, expected) in [(lower_x, lower_expected), (upper_x, upper_expected)] {
                let actual = beta_reg(shape, shape, f64::from_bits(x)).to_bits();
                assert!(
                    actual.abs_diff(expected) <= 4,
                    "shape={shape:?}, x={x:#018x}, actual={actual:#018x}, expected={expected:#018x}"
                );
            }
        }
    }

    #[test]
    fn test_beta_reg_large_asymmetric_central_range() {
        let cases = [
            (
                100_000_000.0,
                200_000_000.0,
                0x3fd554e32dc84e59_u64,
                0x3fc44ed0bc353f04_u64,
            ),
            (
                12_000_000.0,
                108_000_000.0,
                0x3fb99926bbf81f29,
                0x3fd9af470acc030a,
            ),
            (
                40_000_000.0,
                80_000_000.0,
                0x3fd5555555555555,
                0x3fe00012006e3f11,
            ),
            (
                108_000_000.0,
                12_000_000.0,
                0x3fecccbe71189d7f,
                0x3fd9ae504ae8a2e1,
            ),
            (
                100_000_000.0,
                900_000_000.0,
                0x3fb998fa6ff66eb1,
                0x3fc44ed0bb21a1af,
            ),
            (
                100_000_000.0,
                900_000_000.0,
                0x3fb999999999999a,
                0x3fe00017846dc7c6,
            ),
            (
                100_000_000.0,
                900_000_000.0,
                0x3fb99a38c33cc483,
                0x3feaec4bd137a086,
            ),
            (
                333_333_333.333_333_3,
                666_666_666.666_666_7,
                0x3fd55516ceef6eb5,
                0x3fc44ed0bbb46ae2,
            ),
            (
                333_333_333.333_333_3,
                666_666_666.666_666_7,
                0x3fd5555555555555,
                0x3fe000063c68549a,
            ),
            (
                333_333_333.333_333_3,
                666_666_666.666_666_7,
                0x3fd55593dbbb3bf5,
                0x3feaec4bd112fd8f,
            ),
        ];
        for (a, b, x, expected) in cases {
            let actual = beta_reg(a, b, f64::from_bits(x)).to_bits();
            assert!(
                actual.abs_diff(expected) <= 4,
                "a={a:?}, b={b:?}, x={x:#018x}, actual={actual:#018x}, expected={expected:#018x}"
            );
        }
    }

    #[test]
    fn test_beta_reg_temme_deviance_boundary() {
        let groups = [
            (
                1_000_000.0,
                9_000_000.0,
                [
                    (0x3fb97fe3e2209f30_u64, 0x3ef23a83b513c998_u64),
                    (0x3fb97fe3e2209f31, 0x3ef23a83b513d668),
                    (0x3fb97fe3e2209f32, 0x3ef23a83b513e337),
                    (0x3fb97fe3e2209f33, 0x3ef23a83b513f006),
                    (0x3fb97fe3e2209f34, 0x3ef23a83b513fcd6),
                ],
            ),
            (
                1_000_100.0,
                98_999_900.0,
                [
                    (0x3f846555255c20b9, 0x3ee7d0fb4a5ec8c2),
                    (0x3f846555255c20ba, 0x3ee7d0fb4a5edd23),
                    (0x3f846555255c20bb, 0x3ee7d0fb4a5ef184),
                    (0x3f846555255c20bc, 0x3ee7d0fb4a5f05e5),
                    (0x3f846555255c20bd, 0x3ee7d0fb4a5f1a46),
                ],
            ),
        ];
        for (a, b, cases) in groups {
            let mut previous = 0_u64;
            for (x, expected) in cases {
                let actual = beta_reg(a, b, f64::from_bits(x)).to_bits();
                assert!(
                    actual.abs_diff(expected) <= 4,
                    "a={a:?}, b={b:?}, x={x:#018x}, actual={actual:#018x}, expected={expected:#018x}"
                );
                assert!(actual > previous, "a={a:?}, b={b:?}, x={x:#018x}");
                previous = actual;
            }
        }
    }

    #[test]
    fn test_beta_reg_temme_shape_boundaries() {
        let groups = [
            [
                (
                    3_333_333.0,
                    6_666_666.0,
                    0x3fd5555555555555_u64,
                    0x3fe0003e5c12c87a_u64,
                    4_u64,
                ),
                (
                    3_333_333.0,
                    6_666_667.0,
                    0x3fd55555318abc87,
                    0x3fe0003e5c138137,
                    4,
                ),
                (
                    3_333_334.0,
                    6_666_667.0,
                    0x3fd55555791fede8,
                    0x3fe0003e5c117848,
                    4,
                ),
            ],
            [
                (
                    999_999.0,
                    19_000_001.0,
                    0x3fa99997ec1a6fee,
                    0x3fe0010183687a50,
                    4,
                ),
                (
                    1_000_000.0,
                    19_000_000.0,
                    0x3fa999999999999a,
                    0x3fe00101835e9c34,
                    4,
                ),
                (
                    1_000_001.0,
                    18_999_999.0,
                    0x3fa9999b4718c345,
                    0x3fe001018354bc19,
                    4,
                ),
            ],
            [
                (
                    999_999.0,
                    99_000_001.0,
                    0x3f847adff0152658,
                    0x3fe00112ae2480be,
                    64,
                ),
                (
                    1_000_000.0,
                    99_000_000.0,
                    0x3f847ae147ae147b,
                    0x3fe00112ae1b39b3,
                    4,
                ),
                (
                    1_000_001.0,
                    98_999_999.0,
                    0x3f847ae29f47029e,
                    0x3fe00112ae11f2a9,
                    4,
                ),
            ],
        ];
        for cases in groups {
            for (a, b, x, expected, max_ulp) in cases {
                let actual = beta_reg(a, b, f64::from_bits(x)).to_bits();
                assert!(
                    actual.abs_diff(expected) <= max_ulp,
                    "a={a:?}, b={b:?}, x={x:#018x}, actual={actual:#018x}, expected={expected:#018x}"
                );
            }
        }
    }

    #[test]
    fn test_beta_reg_temme_sum_boundary() {
        let cases = [
            (
                333_333.0,
                666_667.0,
                1.0 / 3.0,
                0x3fe00314cb61cb08_u64,
                8_u64,
            ),
            (333_332.0, 666_667.0, 1.0 / 3.0, 0x3fe007b3fc6bdec5, 8),
            (100_000.0, 900_000.0, 0.1, 0x3fe002e7aeb53fe7, 8),
            (99_999.0, 900_000.0, 0.1, 0x3fe00cb59c53af1f, 8),
            (33_333.0, 66_667.0, 1.0 / 3.0, 0x3fe009be64ef20eb, 64),
            (33_332.0, 66_667.0, 1.0 / 3.0, 0x3fe0185bfb455dda, 64),
            (10_000.0, 90_000.0, 0.1, 0x3fe0092fbc02d93b, 2_048),
            (9_999.0, 90_000.0, 0.1, 0x3fe02830d5e59996, 2_048),
            (3_333.0, 6_667.0, 1.0 / 3.0, 0x3fe01ed031aeb5fe, 512),
            (3_332.0, 6_667.0, 1.0 / 3.0, 0x3fe04d085a8c499b, 512),
            (1_000.0, 9_000.0, 0.1, 0x3fe01d0cf6c4eab9, 512),
            (999.0, 9_000.0, 0.1, 0x3fe07f18a31ed127, 512),
        ];
        for (a, b, x, expected, max_ulp) in cases {
            let actual = beta_reg(a, b, x).to_bits();
            assert!(
                actual.abs_diff(expected) <= max_ulp,
                "a={a:?}, b={b:?}, x={x:?}, actual={actual:#018x}, expected={expected:#018x}"
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_beta_reg_a_lte_0() {
        beta_reg(0.0, 1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_reg_b_lte_0() {
        beta_reg(1.0, 0.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_reg_x_lt_0() {
        beta_reg(1.0, 1.0, -1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_reg_x_gt_1() {
        beta_reg(1.0, 1.0, 2.0);
    }

    #[test]
    fn test_checked_beta_reg_a_lte_0() {
        assert!(checked_beta_reg(0.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_beta_reg_b_lte_0() {
        assert!(checked_beta_reg(1.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_beta_reg_x_lt_0() {
        assert!(checked_beta_reg(1.0, 1.0, -1.0).is_err());
    }

    #[test]
    fn test_checked_beta_reg_x_gt_1() {
        assert!(checked_beta_reg(1.0, 1.0, 2.0).is_err());
    }

    #[test]
    fn test_error_is_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<BetaFuncError>();
    }
}
