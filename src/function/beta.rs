//! Provides the [beta](https://en.wikipedia.org/wiki/Beta_function) and related
//! function

use crate::error::StatsError;
use crate::function::gamma;
use crate::is_zero;
use crate::prec;
use crate::Result;
use std::f64;

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
pub fn checked_ln_beta(a: f64, b: f64) -> Result<f64> {
    if a <= 0.0 {
        Err(StatsError::ArgMustBePositive("a"))
    } else if b <= 0.0 {
        Err(StatsError::ArgMustBePositive("b"))
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
pub fn checked_beta(a: f64, b: f64) -> Result<f64> {
    checked_ln_beta(a, b).map(|x| x.exp())
}

/// Computes the lower incomplete (unregularized) beta function
/// `B(a,b,x) = int(t^(a-1)*(1-t)^(b-1),t=0..x)` for `a > 0, b > 0, 1 >= x >= 0`
/// where `a` is the first beta parameter, `b` is the second beta parameter, and
/// `x` is the upper limit of the integral
///
/// # Panics
///
/// If `a <= 0.0`, `b <= 0.0`, `x < 0.0`, or `x > 1.0`
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
/// If `a <= 0.0`, `b <= 0.0`, `x < 0.0`, or `x > 1.0`
pub fn checked_beta_inc(a: f64, b: f64, x: f64) -> Result<f64> {
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
/// if `a <= 0.0`, `b <= 0.0`, `x < 0.0`, or `x > 1.0`
pub fn beta_reg(a: f64, b: f64, x: f64) -> f64 {
    checked_beta_reg(a, b, x).unwrap()
}

/// Computes the regularized lower incomplete beta function
/// `I_x(a,b) = 1/Beta(a,b) * int(t^(a-1)*(1-t)^(b-1), t=0..x)`
/// `a > 0`, `b > 0`, `1 >= x >= 0` where `a` is the first beta parameter,
/// `b` is the second beta parameter, and `x` is the upper limit of the
/// integral.
///
/// # Errors
///
/// if `a <= 0.0`, `b <= 0.0`, `x < 0.0`, or `x > 1.0`
pub fn checked_beta_reg(a: f64, b: f64, x: f64) -> Result<f64> {
    if a <= 0.0 {
        Err(StatsError::ArgMustBePositive("a"))
    } else if b <= 0.0 {
        Err(StatsError::ArgMustBePositive("b"))
    } else if !(0.0..=1.0).contains(&x) {
        Err(StatsError::ArgIntervalIncl("x", 0.0, 1.0))
    } else {
        let bt = if is_zero(x) || ulps_eq!(x, 1.0) {
            0.0
        } else {
            (gamma::ln_gamma(a + b) - gamma::ln_gamma(a) - gamma::ln_gamma(b)
                + a * x.ln()
                + b * (1.0 - x).ln())
            .exp()
        };
        let symm_transform = x >= (a + 1.0) / (a + b + 2.0);
        let eps = prec::F64_PREC;
        let fpmin = f64::MIN_POSITIVE / eps;

        let mut a = a;
        let mut b = b;
        let mut x = x;
        if symm_transform {
            let swap = a;
            x = 1.0 - x;
            a = b;
            b = swap;
        }

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

        for m in 1..141 {
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
            let del = d * c;
            h *= del;

            if (del - 1.0).abs() <= eps {
                return if symm_transform {
                    Ok(1.0 - bt * h / a)
                } else {
                    Ok(bt * h / a)
                };
            }
        }

        if symm_transform {
            Ok(1.0 - bt * h / a)
        } else {
            Ok(bt * h / a)
        }
    }
}

/// Computes the inverse of the regularized incomplete beta function
//
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
// > this software and associated documentation files (the “Software”), to deal in
// > the Software without restriction, including without limitation the rights to
// > use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// > the Software, and to permit persons to whom the Software is furnished to do so,
// > subject to the following conditions:
// >
// > The above copyright notice and this permission notice shall be included in all
// > copies or substantial portions of the Software.
// >
// > THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
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
    // degrees of freedom and is obtained from Wilson and Hilferty’s
    // approximation (cf. Wilson and Hilferty, 1931)
    //
    // χ²(α) = 2q (1 - 1 / (9q) + y(α) sqrt(1 / (9q)))^3,
    //
    // y(α) being Hastings’ approximation (cf. Hastings, 1955) for the upper
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

    if p < 0.0001 {
        p = 0.0001;
    } else if 0.9999 < p {
        p = 0.9999;
    }

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

    if flip {
        1.0 - p
    } else {
        p
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    #[test]
    fn test_ln_beta() {
        assert_almost_eq!(super::ln_beta(0.5, 0.5), 1.144729885849400174144, 1e-15);
        assert_almost_eq!(super::ln_beta(1.0, 0.5), 0.6931471805599453094172, 1e-14);
        assert_almost_eq!(super::ln_beta(2.5, 0.5), 0.163900632837673937284, 1e-15);
        assert_almost_eq!(super::ln_beta(0.5, 1.0), 0.6931471805599453094172, 1e-14);
        assert_almost_eq!(super::ln_beta(1.0, 1.0), 0.0, 1e-15);
        assert_almost_eq!(super::ln_beta(2.5, 1.0), -0.9162907318741550651835, 1e-14);
        assert_almost_eq!(super::ln_beta(0.5, 2.5), 0.163900632837673937284, 1e-15);
        assert_almost_eq!(super::ln_beta(1.0, 2.5), -0.9162907318741550651835, 1e-14);
        assert_almost_eq!(super::ln_beta(2.5, 2.5), -2.608688089402107300388, 1e-14);
    }

    #[test]
    #[should_panic]
    fn test_ln_beta_a_lte_0() {
        super::ln_beta(0.0, 0.5);
    }

    #[test]
    #[should_panic]
    fn test_ln_beta_b_lte_0() {
        super::ln_beta(0.5, 0.0);
    }

    #[test]
    fn test_checked_ln_beta_a_lte_0() {
        assert!(super::checked_ln_beta(0.0, 0.5).is_err());
    }

    #[test]
    fn test_checked_ln_beta_b_lte_0() {
        assert!(super::checked_ln_beta(0.5, 0.0).is_err());
    }

    #[test]
    #[should_panic]
    fn test_beta_a_lte_0() {
        super::beta(0.0, 0.5);
    }

    #[test]
    #[should_panic]
    fn test_beta_b_lte_0() {
        super::beta(0.5, 0.0);
    }

    #[test]
    fn test_checked_beta_a_lte_0() {
        assert!(super::checked_beta(0.0, 0.5).is_err());
    }

    #[test]
    fn test_checked_beta_b_lte_0() {
        assert!(super::checked_beta(0.5, 0.0).is_err());
    }

    #[test]
    fn test_beta() {
        assert_almost_eq!(super::beta(0.5, 0.5), 3.141592653589793238463, 1e-15);
        assert_almost_eq!(super::beta(1.0, 0.5), 2.0, 1e-14);
        assert_almost_eq!(super::beta(2.5, 0.5), 1.17809724509617246442, 1e-15);
        assert_almost_eq!(super::beta(0.5, 1.0), 2.0, 1e-14);
        assert_almost_eq!(super::beta(1.0, 1.0), 1.0, 1e-15);
        assert_almost_eq!(super::beta(2.5, 1.0), 0.4, 1e-14);
        assert_almost_eq!(super::beta(0.5, 2.5), 1.17809724509617246442, 1e-15);
        assert_almost_eq!(super::beta(1.0, 2.5), 0.4, 1e-14);
        assert_almost_eq!(super::beta(2.5, 2.5), 0.073631077818510779026, 1e-15);
    }

    #[test]
    fn test_beta_inc() {
        assert_almost_eq!(super::beta_inc(0.5, 0.5, 0.5), 1.570796326794896619231, 1e-14);
        assert_almost_eq!(super::beta_inc(0.5, 0.5, 1.0), 3.141592653589793238463, 1e-15);
        assert_almost_eq!(super::beta_inc(1.0, 0.5, 0.5), 0.5857864376269049511983, 1e-15);
        assert_almost_eq!(super::beta_inc(1.0, 0.5, 1.0), 2.0, 1e-14);
        assert_almost_eq!(super::beta_inc(2.5, 0.5, 0.5), 0.0890486225480862322117, 1e-16);
        assert_almost_eq!(super::beta_inc(2.5, 0.5, 1.0), 1.17809724509617246442, 1e-15);
        assert_almost_eq!(super::beta_inc(0.5, 1.0, 0.5), 1.414213562373095048802, 1e-14);
        assert_almost_eq!(super::beta_inc(0.5, 1.0, 1.0), 2.0, 1e-14);
        assert_almost_eq!(super::beta_inc(1.0, 1.0, 0.5), 0.5, 1e-15);
        assert_almost_eq!(super::beta_inc(1.0, 1.0, 1.0), 1.0, 1e-15);
        assert_eq!(super::beta_inc(2.5, 1.0, 0.5), 0.0707106781186547524401);
        assert_almost_eq!(super::beta_inc(2.5, 1.0, 1.0), 0.4, 1e-14);
        assert_almost_eq!(super::beta_inc(0.5, 2.5, 0.5), 1.08904862254808623221, 1e-15);
        assert_almost_eq!(super::beta_inc(0.5, 2.5, 1.0), 1.17809724509617246442, 1e-15);
        assert_almost_eq!(super::beta_inc(1.0, 2.5, 0.5), 0.32928932188134524756, 1e-14);
        assert_almost_eq!(super::beta_inc(1.0, 2.5, 1.0), 0.4, 1e-14);
        assert_almost_eq!(super::beta_inc(2.5, 2.5, 0.5), 0.03681553890925538951323, 1e-15);
        assert_almost_eq!(super::beta_inc(2.5, 2.5, 1.0), 0.073631077818510779026, 1e-15);
    }

    #[test]
    #[should_panic]
    fn test_beta_inc_a_lte_0() {
        super::beta_inc(0.0, 1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_inc_b_lte_0() {
        super::beta_inc(1.0, 0.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_inc_x_lt_0() {
        super::beta_inc(1.0, 1.0, -1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_inc_x_gt_1() {
        super::beta_inc(1.0, 1.0, 2.0);
    }

    #[test]
    fn test_checked_beta_inc_a_lte_0() {
        assert!(super::checked_beta_inc(0.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_beta_inc_b_lte_0() {
        assert!(super::checked_beta_inc(1.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_beta_inc_x_lt_0() {
        assert!(super::checked_beta_inc(1.0, 1.0, -1.0).is_err());
    }

    #[test]
    fn test_checked_beta_inc_x_gt_1() {
        assert!(super::checked_beta_inc(1.0, 1.0, 2.0).is_err());
    }

    #[test]
    fn test_beta_reg() {
        assert_almost_eq!(super::beta_reg(0.5, 0.5, 0.5), 0.5, 1e-15);
        assert_eq!(super::beta_reg(0.5, 0.5, 1.0), 1.0);
        assert_almost_eq!(super::beta_reg(1.0, 0.5, 0.5), 0.292893218813452475599, 1e-15);
        assert_eq!(super::beta_reg(1.0, 0.5, 1.0), 1.0);
        assert_almost_eq!(super::beta_reg(2.5, 0.5, 0.5), 0.07558681842161243795, 1e-16);
        assert_eq!(super::beta_reg(2.5, 0.5, 1.0), 1.0);
        assert_almost_eq!(super::beta_reg(0.5, 1.0, 0.5), 0.7071067811865475244, 1e-15);
        assert_eq!(super::beta_reg(0.5, 1.0, 1.0), 1.0);
        assert_almost_eq!(super::beta_reg(1.0, 1.0, 0.5), 0.5, 1e-15);
        assert_eq!(super::beta_reg(1.0, 1.0, 1.0), 1.0);
        assert_almost_eq!(super::beta_reg(2.5, 1.0, 0.5), 0.1767766952966368811, 1e-15);
        assert_eq!(super::beta_reg(2.5, 1.0, 1.0), 1.0);
        assert_eq!(super::beta_reg(0.5, 2.5, 0.5), 0.92441318157838756205);
        assert_eq!(super::beta_reg(0.5, 2.5, 1.0), 1.0);
        assert_almost_eq!(super::beta_reg(1.0, 2.5, 0.5), 0.8232233047033631189, 1e-15);
        assert_eq!(super::beta_reg(1.0, 2.5, 1.0), 1.0);
        assert_almost_eq!(super::beta_reg(2.5, 2.5, 0.5), 0.5, 1e-15);
        assert_eq!(super::beta_reg(2.5, 2.5, 1.0), 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_reg_a_lte_0() {
        super::beta_reg(0.0, 1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_reg_b_lte_0() {
        super::beta_reg(1.0, 0.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_reg_x_lt_0() {
        super::beta_reg(1.0, 1.0, -1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_reg_x_gt_1() {
        super::beta_reg(1.0, 1.0, 2.0);
    }

    #[test]
    fn test_checked_beta_reg_a_lte_0() {
        assert!(super::checked_beta_reg(0.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_beta_reg_b_lte_0() {
        assert!(super::checked_beta_reg(1.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_beta_reg_x_lt_0() {
        assert!(super::checked_beta_reg(1.0, 1.0, -1.0).is_err());
    }

    #[test]
    fn test_checked_beta_reg_x_gt_1() {
        assert!(super::checked_beta_reg(1.0, 1.0, 2.0).is_err());
    }
}
