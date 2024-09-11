use core::f64;
use std::f64::consts::PI;

use crate::{
    consts::EULER_MASCHERONI,
    statistics::{Distribution, Max, Median, Min, Mode},
};

use super::{Continuous, ContinuousCDF};

/// https://en.wikipedia.org/wiki/Gumbel_distribution
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Gumbel {
    location: f64,
    scale: f64,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum GumbelError {
    /// The location is invalid (NAN)
    LocationInvalid,

    /// The scale is NAN, zero or less than zero
    ScaleInvalid,
}

impl std::fmt::Display for GumbelError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GumbelError::LocationInvalid => write!(f, "Location is NAN"),
            GumbelError::ScaleInvalid => write!(f, "Scale is NAN, zero or less than zero"),
        }
    }
}

impl std::error::Error for GumbelError {}

impl Gumbel {
    pub fn new(location: f64, scale: f64) -> Result<Self, GumbelError> {
        if location.is_nan() {
            return Err(GumbelError::LocationInvalid);
        }

        if scale.is_nan() || scale <= 0.0 {
            return Err(GumbelError::ScaleInvalid);
        }

        Ok(Self { location, scale })
    }

    pub fn location(&self) -> f64 {
        self.location
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl std::fmt::Display for Gumbel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gumbel({:?}, {:?})", self.location, self.scale)
    }
}

impl ::rand::distributions::Distribution<f64> for Gumbel {
    fn sample<R: rand::Rng + ?Sized>(&self, r: &mut R) -> f64 {
        // Check: Quantile formula: mu - beta*ln(-ln(p))
        self.location - self.scale * ((-(r.gen::<f64>())).ln()).ln()
    }
}

impl ContinuousCDF<f64, f64> for Gumbel {
    /// Calculates the cumulative distribution function for the
    /// gumbel distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// e^(-e^(-(x - μ) / β))
    /// ```
    ///
    /// where `μ` is the location and `β` is the scale
    fn cdf(&self, x: f64) -> f64 {
        (-(-(x - self.location) / self.scale).exp()).exp()
    }

    /// Calculates the inverse cumulative distribution function for the
    /// gumbel distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// μ - β ln(-ln(p)) where 0 < p < 1
    /// -INF             where p <= 0
    /// INF              otherwise
    /// ```
    ///
    /// where `μ` is the location and `β` is the scale
    fn inverse_cdf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            f64::NEG_INFINITY
        } else if p >= 1.0 {
            f64::INFINITY
        } else {
            self.location - self.scale * ((-(p.ln())).ln())
        }
    }

    /// Calculates the survival function for the
    /// gumbel distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// 1 - e^(-e^(-(x - μ) / β))
    /// ```
    ///
    /// where `μ` is the location and `β` is the scale
    fn sf(&self, x: f64) -> f64 {
        1.0 - (-(-(x - self.location) / self.scale).exp()).exp()
    }
}

impl Min<f64> for Gumbel {
    /// Returns the minimum value in the domain of the gumbel
    /// distribution representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```text
    /// NEG_INF
    /// ```
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
}

impl Max<f64> for Gumbel {
    /// Returns the maximum value in the domain of the gumbel
    /// distribution representable by a double precision float
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

impl Distribution<f64> for Gumbel {
    /// Returns the entropy of the gumbel distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// ln(β) + γ + 1
    /// ```
    ///
    /// where `β` is the scale
    /// and `γ` is the Euler-Mascheroni constant (approx 0.57721)
    fn entropy(&self) -> Option<f64> {
        Some(1.0 + EULER_MASCHERONI + (self.scale).ln())
    }

    /// Returns the mean of the gumbel distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// μ + γβ
    /// ```
    ///
    /// where `μ` is the location, `β` is the scale
    /// and `γ` is the Euler-Mascheroni constant (approx 0.57721)
    fn mean(&self) -> Option<f64> {
        Some(self.location + (EULER_MASCHERONI * self.scale))
    }

    /// Returns the skewness of the gumbel distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// 12 * sqrt(6) * ζ(3) / π^3 ≈ 1.13955
    /// ```
    /// ζ(3) is the Riemann zeta function evaluated at 3 (approx 1.20206)
    /// π is the constant PI (approx 3.14159)
    /// This approximately evaluates to 1.13955
    fn skewness(&self) -> Option<f64> {
        Some(1.13955)
    }

    /// Returns the variance of the gumbel distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (π^2 / 6) * β^2
    /// ```
    ///
    /// where `β` is the scale and `π` is the constant PI (approx 3.14159)
    fn variance(&self) -> Option<f64> {
        Some(((PI * PI) / 6.0) * self.scale * self.scale)
    }

    /// Returns the standard deviation of the gumbel distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// β * π / sqrt(6)
    /// ```
    ///
    /// where `β` is the scale and `π` is the constant PI (approx 3.14159)
    fn std_dev(&self) -> Option<f64> {
        Some(self.scale * PI / 6.0_f64.sqrt())
    }
}

impl Median<f64> for Gumbel {
    /// Returns the median of the gumbel distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// μ - β ln(ln(2))
    /// ```
    ///
    /// where `μ` is the location and `β` is the scale parameter
    fn median(&self) -> f64 {
        self.location - self.scale * (((2.0_f64).ln()).ln())
    }
}

impl Mode<f64> for Gumbel {
    /// Returns the mode of the gumbel distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// μ
    /// ```
    ///
    /// where `μ` is the location
    fn mode(&self) -> f64 {
        self.location
    }
}

impl Continuous<f64, f64> for Gumbel {
    /// Calculates the probability density function for the gumbel
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (1/β) * exp(-(x - μ)/β) * exp(-exp(-(x - μ)/β))
    /// ```
    ///
    /// where `μ` is the location, `β` is the scale
    fn pdf(&self, x: f64) -> f64 {
        (1.0_f64 / self.scale)
            * (-(x - self.location) / (self.scale)).exp()
            * (-((-(x - self.location) / self.scale).exp())).exp()
    }

    /// Calculates the log probability density function for the gumbel
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// ln((1/β) * exp(-(x - μ)/β) * exp(-exp(-(x - μ)/β)))
    /// ```
    ///
    /// where `μ` is the location, `β` is the scale
    fn ln_pdf(&self, x: f64) -> f64 {
        ((1.0_f64 / self.scale)
            * (-(x - self.location) / (self.scale)).exp()
            * (-((-(x - self.location) / self.scale).exp())).exp())
        .ln()
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::internal::*;
    use crate::testing_boiler;

    testing_boiler!(location: f64, scale: f64; Gumbel; GumbelError);

    #[test]
    fn test_create() {
        create_ok(0.0, 0.1);
        create_ok(0.0, 1.0);
        create_ok(0.0, 10.0);
        create_ok(10.0, 11.0);
        create_ok(-5.0, 100.0);
        create_ok(0.0, f64::INFINITY);
    }

    #[test]
    fn test_bad_create() {
        let invalid = [
            (f64::NAN, 1.0, GumbelError::LocationInvalid),
            (1.0, f64::NAN, GumbelError::ScaleInvalid),
            (f64::NAN, f64::NAN, GumbelError::LocationInvalid),
            (1.0, 0.0, GumbelError::ScaleInvalid),
            (0.0, f64::NEG_INFINITY, GumbelError::ScaleInvalid)
        ];

        for (location, scale, err) in invalid {
            test_create_err(location, scale, err);
        }
    }

    #[test]
    fn test_min_max() {
        let min = |x: Gumbel| x.min();
        let max = |x:Gumbel| x.max();

        test_exact(0.0, 1.0, f64::NEG_INFINITY, min);
        test_exact(0.0, 1.0, f64::INFINITY, max);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Gumbel| x.entropy().unwrap();
        test_exact(0.0, 2.0, 2.270362845461478, entropy);
        test_exact(0.1, 4.0, 2.9635100260214235, entropy);
        test_exact(1.0, 10.0, 3.8798007578955787, entropy);
        test_exact(10.0, 11.0, 3.9751109376999034, entropy); 
    }

    #[test]
    fn test_mean() {
        let mean = |x: Gumbel| x.mean().unwrap();
        test_exact(0.0, 2.0, 1.1544313298030658, mean);
        test_exact(0.1, 4.0, 2.4088626596061316, mean);
        test_exact(1.0, 10.0, 6.772156649015328, mean);
        test_exact(10.0, 11.0, 16.34937231391686, mean);
        test_exact(10.0, f64::INFINITY, f64::INFINITY, mean);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: Gumbel| x.skewness().unwrap();
        test_exact(0.0, 2.0, 1.13955, skewness);
        test_exact(0.1, 4.0, 1.13955, skewness);
        test_exact(1.0, 10.0, 1.13955, skewness);
        test_exact(10.0, 11.0, 1.13955, skewness);
        test_exact(10.0, f64::INFINITY, 1.13955, skewness);
    }

    #[test]
    fn test_variance() {
        let variance = |x: Gumbel| x.variance().unwrap();
        test_exact(0.0, 2.0, 6.579736267392906, variance);
        test_exact(0.1, 4.0, 26.318945069571624, variance);
        test_exact(1.0, 10.0, 164.49340668482265, variance);
        test_exact(10.0, 11.0, 199.03702208863538, variance);
    }

    #[test]
    fn test_std_dev() {
        let std_dev = |x: Gumbel| x.std_dev().unwrap();
        test_exact(0.0, 2.0, 2.565099660323728, std_dev);
        test_exact(0.1, 4.0, 5.130199320647456, std_dev);
        test_exact(1.0, 10.0, 12.82549830161864, std_dev);
        test_exact(10.0, 11.0, 14.108048131780505, std_dev);
    }

    #[test]
    fn test_median() {
        let median = |x: Gumbel| x.median();
        test_exact(0.0, 2.0, 0.7330258411633287, median);
        test_exact(0.1, 4.0, 1.5660516823266574, median);
        test_exact(1.0, 10.0, 4.665129205816644, median);
        test_exact(10.0, 11.0, 14.031642126398307, median);
        test_exact(10.0, f64::INFINITY, f64::INFINITY, median);
    }

    #[test]
    fn test_mode() {
        let mode = |x: Gumbel| x.mode();
        test_exact(0.0, 2.0, 0.0, mode);
        test_exact(0.1, 4.0, 0.1, mode);
        test_exact(1.0, 10.0, 1.0, mode);
        test_exact(10.0, 11.0, 10.0, mode);
        test_exact(10.0, f64::INFINITY, 10.0, mode);
    }

    #[test]
    fn test_cdf() {
        let cdf = |a: f64| move |x: Gumbel| x.cdf(a);
        test_exact(0.0, 0.1, 0.0, cdf(-5.0));
        test_exact(0.0, 0.1, 0.0, cdf(-1.0));
        test_exact(0.0, 0.1, 0.36787944117144233, cdf(0.0));
        test_exact(0.0, 0.1, 0.9999546011007987, cdf(1.0));
        test_absolute(0.0, 0.1, 0.99999999999999999, 1e-12, cdf(5.0));
        test_absolute(0.0, 1.0, 0.06598803584531253, 1e-12, cdf(-1.0));
        test_exact(0.0, 1.0, 0.36787944117144233, cdf(0.0));
        test_absolute(0.0, 10.0, 0.192295645547964928, 1e-12, cdf(-5.0));
        test_absolute(0.0, 10.0, 0.3311542771529088, 1e-12, cdf(-1.0));
        test_exact(0.0, 10.0, 0.36787944117144233, cdf(0.0));
        test_absolute(0.0, 10.0, 0.4046076616641318, 1e-12, cdf(1.0));
        test_absolute(0.0, 10.0, 0.545239211892605, 1e-12, cdf(5.0));
        test_exact(-2.0, f64::INFINITY, 0.36787944117144233, cdf(-5.0));
        test_exact(-2.0, f64::INFINITY, 0.36787944117144233, cdf(-1.0));
        test_exact(-2.0, f64::INFINITY, 0.36787944117144233, cdf(0.0));
        test_exact(-2.0, f64::INFINITY, 0.36787944117144233, cdf(1.0));
        test_exact(-2.0, f64::INFINITY, 0.36787944117144233, cdf(5.0));
        test_exact(f64::INFINITY, 1.0, 0.0, cdf(-5.0));
        test_exact(f64::INFINITY, 1.0, 0.0, cdf(-1.0));
        test_exact(f64::INFINITY, 1.0, 0.0, cdf(0.0));
        test_exact(f64::INFINITY, 1.0, 0.0, cdf(1.0));
        test_exact(f64::INFINITY, 1.0, 0.0, cdf(5.0));
    }

    #[test]
    fn test_inverse_cdf() {
        let inv_cdf = |a: f64| move |x: Gumbel| x.inverse_cdf(a);
        test_exact(0.0, 0.1, f64::NEG_INFINITY, inv_cdf(-5.0));
        test_exact(0.0, 0.1, f64::NEG_INFINITY, inv_cdf(-1.0));
        test_exact(0.0, 0.1, f64::NEG_INFINITY, inv_cdf(0.0));
        test_exact(0.0, 0.1, f64::INFINITY, inv_cdf(1.0));
        test_exact(0.0, 0.1, f64::INFINITY, inv_cdf(5.0));
        test_absolute(0.0, 1.0, -0.8340324452479557, 1e-12, inv_cdf(0.1));
        test_absolute(0.0, 10.0, 3.6651292058166436, 1e-12, inv_cdf(0.5));
        test_absolute(0.0, 10.0, 22.503673273124456, 1e-12, inv_cdf(0.9));
        test_exact(2.0, f64::INFINITY, f64::NEG_INFINITY, inv_cdf(0.1));
        test_exact(-2.0, f64::INFINITY, f64::INFINITY, inv_cdf(0.5));
        test_exact(f64::INFINITY, 1.0, f64::INFINITY, inv_cdf(0.1));
    }
}
