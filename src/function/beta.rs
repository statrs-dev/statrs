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
    } else if x < 0.0 || x > 1.0 {
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

#[rustfmt::skip]
#[cfg(test)]
mod test {
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
