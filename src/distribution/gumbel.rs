use core::f64;
use std::f64::consts::PI;

use crate::{
    consts::EULER_MASCHERONI,
    statistics::{Distribution, Max, MeanN, Median, Min, Mode},
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

        if scale.is_nan() && scale <= 0.0 {
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
    /// μ - β ln(-ln(p))
    /// ```
    ///
    /// where `μ` is the location and `β` is the scale
    fn inverse_cdf(&self, p: f64) -> f64 {
        self.location - self.scale * ((-(p.ln())).ln())
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
