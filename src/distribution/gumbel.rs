use core::f64;

use crate::statistics::{Max, Min};

use super::ContinuousCDF;

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
        self.location - self.scale * (((-(r.gen::<f64>())).ln()).ln())
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
