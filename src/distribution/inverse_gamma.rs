use crate::distribution::{Continuous, ContinuousCDF};
use crate::function::gamma;
use crate::statistics::*;
use crate::{Result, StatsError};
use rand::Rng;
use std::f64;

/// Implements the [Inverse
/// Gamma](https://en.wikipedia.org/wiki/Inverse-gamma_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{InverseGamma, Continuous};
/// use statrs::statistics::Distribution;
/// use statrs::prec;
///
/// let n = InverseGamma::new(1.1, 0.1).unwrap();
/// assert!(prec::almost_eq(n.mean().unwrap(), 1.0, 1e-14));
/// assert_eq!(n.pdf(1.0), 0.07554920138253064);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct InverseGamma {
    shape: f64,
    rate: f64,
}

impl InverseGamma {
    /// Constructs a new inverse gamma distribution with a shape (α)
    /// of `shape` and a rate (β) of `rate`
    ///
    /// # Errors
    ///
    /// Returns an error if `shape` or `rate` are `NaN`.
    /// Also returns an error if `shape` or `rate` are not in `(0, +inf)`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::InverseGamma;
    ///
    /// let mut result = InverseGamma::new(3.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = InverseGamma::new(0.0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(shape: f64, rate: f64) -> Result<InverseGamma> {
        let is_nan = shape.is_nan() || rate.is_nan();
        match (shape, rate, is_nan) {
            (_, _, true) => Err(StatsError::BadParams),
            (_, _, false) if shape <= 0.0 || rate <= 0.0 => Err(StatsError::BadParams),
            (_, _, false) if shape.is_infinite() || rate.is_infinite() => {
                Err(StatsError::BadParams)
            }
            (_, _, false) => Ok(InverseGamma { shape, rate }),
        }
    }

    /// Returns the shape (α) of the inverse gamma distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::InverseGamma;
    ///
    /// let n = InverseGamma::new(3.0, 1.0).unwrap();
    /// assert_eq!(n.shape(), 3.0);
    /// ```
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Returns the rate (β) of the inverse gamma distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::InverseGamma;
    ///
    /// let n = InverseGamma::new(3.0, 1.0).unwrap();
    /// assert_eq!(n.rate(), 1.0);
    /// ```
    pub fn rate(&self) -> f64 {
        self.rate
    }
}

impl ::rand::distributions::Distribution<f64> for InverseGamma {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> f64 {
        1.0 / super::gamma::sample_unchecked(r, self.shape, self.rate)
    }
}

impl ContinuousCDF<f64, f64> for InverseGamma {
    /// Calculates the cumulative distribution function for the inverse gamma
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// Γ(α, β / x) / Γ(α)
    /// ```
    ///
    /// where the numerator is the upper incomplete gamma function,
    /// the denominator is the gamma function, `α` is the shape,
    /// and `β` is the rate
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else if x.is_infinite() {
            1.0
        } else {
            gamma::gamma_ur(self.shape, self.rate / x)
        }
    }

    /// Calculates the survival function for the inverse gamma
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// Γ(α, β / x) / Γ(α)
    /// ```
    ///
    /// where the numerator is the lower incomplete gamma function,
    /// the denominator is the gamma function, `α` is the shape,
    /// and `β` is the rate
    fn sf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            1.0
        } else if x.is_infinite() {
            0.0
        } else {
            gamma::gamma_lr(self.shape, self.rate / x)
        }
    }
}

impl Min<f64> for InverseGamma {
    /// Returns the minimum value in the domain of the
    /// inverse gamma distribution representable by a double precision
    /// float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    fn min(&self) -> f64 {
        0.0
    }
}

impl Max<f64> for InverseGamma {
    /// Returns the maximum value in the domain of the
    /// inverse gamma distribution representable by a double precision
    /// float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// INF
    /// ```
    fn max(&self) -> f64 {
        f64::INFINITY
    }
}

impl Distribution<f64> for InverseGamma {
    /// Returns the mean of the inverse distribution
    ///
    /// # None
    ///
    /// If `shape <= 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// β / (α - 1)
    /// ```
    ///
    /// where `α` is the shape and `β` is the rate
    fn mean(&self) -> Option<f64> {
        if self.shape <= 1.0 {
            None
        } else {
            Some(self.rate / (self.shape - 1.0))
        }
    }
    /// Returns the variance of the inverse gamma distribution
    ///
    /// # None
    ///
    /// If `shape <= 2.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// β^2 / ((α - 1)^2 * (α - 2))
    /// ```
    ///
    /// where `α` is the shape and `β` is the rate
    fn variance(&self) -> Option<f64> {
        if self.shape <= 2.0 {
            None
        } else {
            let val = self.rate * self.rate
                / ((self.shape - 1.0) * (self.shape - 1.0) * (self.shape - 2.0));
            Some(val)
        }
    }
    /// Returns the entropy of the inverse gamma distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// α + ln(β * Γ(α)) - (1 + α) * ψ(α)
    /// ```
    ///
    /// where `α` is the shape, `β` is the rate, `Γ` is the gamma function,
    /// and `ψ` is the digamma function
    fn entropy(&self) -> Option<f64> {
        let entr = self.shape + self.rate.ln() + gamma::ln_gamma(self.shape)
            - (1.0 + self.shape) * gamma::digamma(self.shape);
        Some(entr)
    }
    /// Returns the skewness of the inverse gamma distribution
    ///
    /// # None
    ///
    /// If `shape <= 3`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 4 * sqrt(α - 2) / (α - 3)
    /// ```
    ///
    /// where `α` is the shape
    fn skewness(&self) -> Option<f64> {
        if self.shape <= 3.0 {
            None
        } else {
            Some(4.0 * (self.shape - 2.0).sqrt() / (self.shape - 3.0))
        }
    }
}

impl Mode<Option<f64>> for InverseGamma {
    /// Returns the mode of the inverse gamma distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// β / (α + 1)
    /// ```
    ///
    /// /// where `α` is the shape and `β` is the rate
    fn mode(&self) -> Option<f64> {
        Some(self.rate / (self.shape + 1.0))
    }
}

impl Continuous<f64, f64> for InverseGamma {
    /// Calculates the probability density function for the
    /// inverse gamma distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (β^α / Γ(α)) * x^(-α - 1) * e^(-β / x)
    /// ```
    ///
    /// where `α` is the shape, `β` is the rate, and `Γ` is the gamma function
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x.is_infinite() {
            0.0
        } else if ulps_eq!(self.shape, 1.0) {
            self.rate / (x * x) * (-self.rate / x).exp()
        } else {
            self.rate.powf(self.shape) * x.powf(-self.shape - 1.0) * (-self.rate / x).exp()
                / gamma::gamma(self.shape)
        }
    }

    /// Calculates the probability density function for the
    /// inverse gamma distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln((β^α / Γ(α)) * x^(-α - 1) * e^(-β / x))
    /// ```
    ///
    /// where `α` is the shape, `β` is the rate, and `Γ` is the gamma function
    fn ln_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }
}

#[rustfmt::skip]
#[cfg(all(test, feature = "nightly"))]
mod tests {
    use crate::statistics::*;
    use crate::distribution::{ContinuousCDF, Continuous, InverseGamma};
    use crate::distribution::internal::*;
    use crate::consts::ACC;

    fn try_create(shape: f64, rate: f64) -> InverseGamma {
        let n = InverseGamma::new(shape, rate);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(shape: f64, rate: f64) {
        let n = try_create(shape, rate);
        assert_eq!(shape, n.shape());
        assert_eq!(rate, n.rate());
    }

    fn bad_create_case(shape: f64, rate: f64) {
        let n = InverseGamma::new(shape, rate);
        assert!(n.is_err());
    }

    fn get_value<F>(shape: f64, rate: f64, eval: F) -> f64
        where F: Fn(InverseGamma) -> f64
    {
        let n = try_create(shape, rate);
        eval(n)
    }

    fn test_case<F>(shape: f64, rate: f64, expected: f64, eval: F)
        where F: Fn(InverseGamma) -> f64
    {
        let x = get_value(shape, rate, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(shape: f64, rate: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(InverseGamma) -> f64
    {
        let x = get_value(shape, rate, eval);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_create() {
        create_case(0.1, 0.1);
        create_case(1.0, 1.0);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(0.0, 1.0);
        bad_create_case(-1.0, 1.0);
        bad_create_case(-100.0, 1.0);
        bad_create_case(f64::NEG_INFINITY, 1.0);
        bad_create_case(f64::NAN, 1.0);
        bad_create_case(1.0, 0.0);
        bad_create_case(1.0, -1.0);
        bad_create_case(1.0, -100.0);
        bad_create_case(1.0, f64::NEG_INFINITY);
        bad_create_case(1.0, f64::NAN);
        bad_create_case(f64::INFINITY, 1.0);
        bad_create_case(1.0, f64::INFINITY);
        bad_create_case(f64::INFINITY, f64::INFINITY);
    }

    #[test]
    fn test_mean() {
        let mean = |x: InverseGamma| x.mean().unwrap();
        test_almost(1.1, 0.1, 1.0, 1e-14, mean);
        test_almost(1.1, 1.0, 10.0, 1e-14, mean);
    }

    #[test]
    #[should_panic]
    fn test_mean_with_shape_lte_1() {
        let mean = |x: InverseGamma| x.mean().unwrap();
        get_value(0.1, 0.1, mean);
    }

    #[test]
    fn test_variance() {
        let variance = |x: InverseGamma| x.variance().unwrap();
        test_almost(2.1, 0.1, 0.08264462809917355371901, 1e-15, variance);
        test_almost(2.1, 1.0, 8.264462809917355371901, 1e-13, variance);
    }

    #[test]
    #[should_panic]
    fn test_variance_with_shape_lte_2() {
        let variance = |x: InverseGamma| x.variance().unwrap();
        get_value(0.1, 0.1, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: InverseGamma| x.entropy().unwrap();
        test_almost(0.1, 0.1, 11.51625799319234475054, 1e-14, entropy);
        test_almost(1.0, 1.0, 2.154431329803065721213, 1e-14, entropy);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: InverseGamma| x.skewness().unwrap();
        test_almost(3.1, 0.1, 41.95235392680606187966, 1e-13, skewness);
        test_almost(3.1, 1.0, 41.95235392680606187966, 1e-13, skewness);
        test_case(5.0, 0.1, 3.464101615137754587055, skewness);
    }

    #[test]
    #[should_panic]
    fn test_skewness_with_shape_lte_3() {
        let skewness = |x: InverseGamma| x.skewness().unwrap();
        get_value(0.1, 0.1, skewness);
    }

    #[test]
    fn test_mode() {
        let mode = |x: InverseGamma| x.mode().unwrap();
        test_case(0.1, 0.1, 0.09090909090909090909091, mode);
        test_case(1.0, 1.0, 0.5, mode);
    }

    #[test]
    fn test_min_max() {
        let min = |x: InverseGamma| x.min();
        let max = |x: InverseGamma| x.max();
        test_case(1.0, 1.0, 0.0, min);
        test_case(1.0, 1.0, f64::INFINITY, max);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: f64| move |x: InverseGamma| x.pdf(arg);
        test_almost(0.1, 0.1, 0.0628591853882328004197, 1e-15, pdf(1.2));
        test_almost(0.1, 1.0, 0.0297426109178248997426, 1e-15, pdf(2.0));
        test_case(1.0, 0.1, 0.04157808822362745501024, pdf(1.5));
        test_case(1.0, 1.0, 0.3018043114632487660842, pdf(1.2));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: InverseGamma| x.ln_pdf(arg);
        test_almost(0.1, 0.1, 0.0628591853882328004197f64.ln(), 1e-15, ln_pdf(1.2));
        test_almost(0.1, 1.0, 0.0297426109178248997426f64.ln(), 1e-15, ln_pdf(2.0));
        test_case(1.0, 0.1, 0.04157808822362745501024f64.ln(), ln_pdf(1.5));
        test_case(1.0, 1.0, 0.3018043114632487660842f64.ln(), ln_pdf(1.2));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: InverseGamma| x.cdf(arg);
        test_almost(0.1, 0.1, 0.1862151961946054271994, 1e-14, cdf(1.2));
        test_almost(0.1, 1.0, 0.05859755410986647796141, 1e-14, cdf(2.0));
        test_case(1.0, 0.1, 0.9355069850316177377304, cdf(1.5));
        test_almost(1.0, 1.0, 0.4345982085070782231613, 1e-14, cdf(1.2));
    }


    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: InverseGamma| x.sf(arg);
        test_almost(0.1, 0.1, 0.8137848038053936, 1e-14, sf(1.2));
        test_almost(0.1, 1.0, 0.9414024458901327, 1e-14, sf(2.0));
        test_almost(1.0, 0.1, 0.0644930149683822, 1e-14, sf(1.5));
        test_almost(1.0, 1.0, 0.565401791492922, 1e-14, sf(1.2));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&try_create(1.0, 0.5), 0.0, 100.0);
        test::check_continuous_distribution(&try_create(9.0, 2.0), 0.0, 100.0);
    }
}
