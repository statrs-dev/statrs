use crate::distribution::{Continuous, ContinuousCDF};
use crate::function::gamma;
use crate::statistics::*;
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
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct InverseGamma {
    shape: f64,
    rate: f64,
}

/// Represents the errors that can occur when creating an [`InverseGamma`].
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum InverseGammaError {
    /// The shape is NaN, infinite, zero or less than zero.
    ShapeInvalid,

    /// The rate is NaN, infinite, zero or less than zero.
    RateInvalid,
}

impl std::fmt::Display for InverseGammaError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            InverseGammaError::ShapeInvalid => {
                write!(f, "Shape is NaN, infinite, zero or less than zero")
            }
            InverseGammaError::RateInvalid => {
                write!(f, "Rate is NaN, infinite, zero or less than zero")
            }
        }
    }
}

impl std::error::Error for InverseGammaError {}

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
    pub fn new(shape: f64, rate: f64) -> Result<InverseGamma, InverseGammaError> {
        if shape.is_nan() || shape.is_infinite() || shape <= 0.0 {
            return Err(InverseGammaError::ShapeInvalid);
        }

        if rate.is_nan() || rate.is_infinite() || rate <= 0.0 {
            return Err(InverseGammaError::RateInvalid);
        }

        Ok(InverseGamma { shape, rate })
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

impl std::fmt::Display for InverseGamma {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Inv-Gamma({}, {})", self.shape, self.rate)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distributions::Distribution<f64> for InverseGamma {
    fn sample<R: ::rand::Rng + ?Sized>(&self, r: &mut R) -> f64 {
        1.0 / super::gamma::sample_unchecked(r, self.shape, self.rate)
    }
}

impl ContinuousCDF<f64, f64> for InverseGamma {
    /// Calculates the cumulative distribution function for the inverse gamma
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
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
    /// ```text
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
    /// ```text
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
    /// ```text
    /// f64::INFINITY
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
    /// ```text
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
    /// ```text
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
    /// ```text
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
    /// ```text
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
    /// ```text
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
    /// ```text
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
    /// ```text
    /// ln((β^α / Γ(α)) * x^(-α - 1) * e^(-β / x))
    /// ```
    ///
    /// where `α` is the shape, `β` is the rate, and `Γ` is the gamma function
    fn ln_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::internal::*;
    use crate::testing_boiler;

    testing_boiler!(shape: f64, rate: f64; InverseGamma; InverseGammaError);

    #[test]
    fn test_create() {
        create_ok(0.1, 0.1);
        create_ok(1.0, 1.0);
    }

    #[test]
    fn test_bad_create() {
        test_create_err(0.0, 1.0, InverseGammaError::ShapeInvalid);
        test_create_err(1.0, -1.0, InverseGammaError::RateInvalid);
        create_err(-1.0, 1.0);
        create_err(-100.0, 1.0);
        create_err(f64::NEG_INFINITY, 1.0);
        create_err(f64::NAN, 1.0);
        create_err(1.0, 0.0);
        create_err(1.0, -100.0);
        create_err(1.0, f64::NEG_INFINITY);
        create_err(1.0, f64::NAN);
        create_err(f64::INFINITY, 1.0);
        create_err(1.0, f64::INFINITY);
        create_err(f64::INFINITY, f64::INFINITY);
    }

    #[test]
    fn test_mean() {
        let mean = |x: InverseGamma| x.mean().unwrap();
        test_absolute(1.1, 0.1, 1.0, 1e-14, mean);
        test_absolute(1.1, 1.0, 10.0, 1e-14, mean);
    }

    #[test]
    fn test_mean_with_shape_lte_1() {
        test_none(0.1, 0.1, |dist| dist.mean());
    }

    #[test]
    fn test_variance() {
        let variance = |x: InverseGamma| x.variance().unwrap();
        test_absolute(2.1, 0.1, 0.08264462809917355371901, 1e-15, variance);
        test_absolute(2.1, 1.0, 8.264462809917355371901, 1e-13, variance);
    }

    #[test]
    fn test_variance_with_shape_lte_2() {
        test_none(0.1, 0.1, |dist| dist.variance());
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: InverseGamma| x.entropy().unwrap();
        test_absolute(0.1, 0.1, 11.51625799319234475054, 1e-14, entropy);
        test_absolute(1.0, 1.0, 2.154431329803065721213, 1e-14, entropy);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: InverseGamma| x.skewness().unwrap();
        test_absolute(3.1, 0.1, 41.95235392680606187966, 1e-13, skewness);
        test_absolute(3.1, 1.0, 41.95235392680606187966, 1e-13, skewness);
        test_exact(5.0, 0.1, 3.464101615137754587055, skewness);
    }

    #[test]
    fn test_skewness_with_shape_lte_3() {
        test_none(0.1, 0.1, |dist| dist.skewness());
    }

    #[test]
    fn test_mode() {
        let mode = |x: InverseGamma| x.mode().unwrap();
        test_exact(0.1, 0.1, 0.09090909090909090909091, mode);
        test_exact(1.0, 1.0, 0.5, mode);
    }

    #[test]
    fn test_min_max() {
        let min = |x: InverseGamma| x.min();
        let max = |x: InverseGamma| x.max();
        test_exact(1.0, 1.0, 0.0, min);
        test_exact(1.0, 1.0, f64::INFINITY, max);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: f64| move |x: InverseGamma| x.pdf(arg);
        test_absolute(0.1, 0.1, 0.0628591853882328004197, 1e-15, pdf(1.2));
        test_absolute(0.1, 1.0, 0.0297426109178248997426, 1e-15, pdf(2.0));
        test_exact(1.0, 0.1, 0.04157808822362745501024, pdf(1.5));
        test_exact(1.0, 1.0, 0.3018043114632487660842, pdf(1.2));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: InverseGamma| x.ln_pdf(arg);
        test_absolute(0.1, 0.1, 0.0628591853882328004197f64.ln(), 1e-15, ln_pdf(1.2));
        test_absolute(0.1, 1.0, 0.0297426109178248997426f64.ln(), 1e-15, ln_pdf(2.0));
        test_exact(1.0, 0.1, 0.04157808822362745501024f64.ln(), ln_pdf(1.5));
        test_exact(1.0, 1.0, 0.3018043114632487660842f64.ln(), ln_pdf(1.2));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: InverseGamma| x.cdf(arg);
        test_absolute(0.1, 0.1, 0.1862151961946054271994, 1e-14, cdf(1.2));
        test_absolute(0.1, 1.0, 0.05859755410986647796141, 1e-14, cdf(2.0));
        test_exact(1.0, 0.1, 0.9355069850316177377304, cdf(1.5));
        test_absolute(1.0, 1.0, 0.4345982085070782231613, 1e-14, cdf(1.2));
    }


    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: InverseGamma| x.sf(arg);
        test_absolute(0.1, 0.1, 0.8137848038053936, 1e-14, sf(1.2));
        test_absolute(0.1, 1.0, 0.9414024458901327, 1e-14, sf(2.0));
        test_absolute(1.0, 0.1, 0.0644930149683822, 1e-14, sf(1.5));
        test_absolute(1.0, 1.0, 0.565401791492922, 1e-14, sf(1.2));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&create_ok(1.0, 0.5), 0.0, 100.0);
        test::check_continuous_distribution(&create_ok(9.0, 2.0), 0.0, 100.0);
    }
}
