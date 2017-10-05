use {Result, StatsError};
use distribution::{Continuous, Distribution, Univariate};
use rand::Rng;
use rand::distributions::{IndependentSample, Sample};
use statistics::*;
use std::f64;


/// Implements the [Pareto](https://en.wikipedia.org/wiki/Pareto_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Pareto, Continuous};
/// use statrs::statistics::Mean;
/// use statrs::prec;
///
/// let n = Pareto::new(1.0, 2.0).unwrap();
/// assert_eq!(n.mean(), 2.0);
/// assert!(prec::almost_eq(n.pdf(0.0), 0.353553390593274, 1e-15)); // TODO:
/// Adjust values
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Pareto {
    scale: f64,
    shape: f64,
}

impl Pareto {
    /// Constructs a new Pareto distribution with scale `scale`, and `shape`
    /// shape.
    ///
    /// # Errors
    ///
    /// Returns an error if any of `scale` or `shape` are `NaN`.
    /// Returns an error if `scale <= 0.0` or `shape <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Pareto;
    ///
    /// let mut result = Pareto::new(1.0, 2.0);
    /// assert!(result.is_ok());
    ///
    /// result = Pareto::new(0.0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(scale: f64, shape: f64) -> Result<Pareto> {
        let is_nan = scale.is_nan() || shape.is_nan();
        if is_nan || scale <= 0.0 || shape <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Pareto {
                scale: scale,
                shape: shape,
            })
        }
    }

    /// Returns the scale of the Pareto distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Pareto;
    ///
    /// let n = Pareto::new(1.0, 2.0).unwrap();
    /// assert_eq!(n.scale(), 1.0);
    /// ```
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns the shape of the Pareto distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Pareto;
    ///
    /// let n = Pareto::new(1.0, 2.0).unwrap();
    /// assert_eq!(n.shape(), 2.0);
    /// ```
    pub fn shape(&self) -> f64 {
        self.shape
    }
}

// TODO: CHECK
impl Sample<f64> for Pareto {
    /// Generate a random sample from a Pareto distribution
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

// TODO: CHECK
impl IndependentSample<f64> for Pareto {
    /// Generate a random independent sample from a Pareto distribution
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution<f64> for Pareto {
    /// Generate a random sample from a Pareto distribution using
    /// `r` as the source of randomness. The implementation uses inverse
    /// transform sampling.
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{Pareto, Distribution};
    ///
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = Pareto::new(1.0, 2.0).unwrap();
    /// print!("{}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        // Draw a sample from (0, 1]
        // next_f64() samples from [0, 1), so we have to subtract it from 1
        let u = 1.0 - r.next_f64();
        self.scale * u.powf(-1.0 / self.shape)
    }
}

impl Univariate<f64, f64> for Pareto {
    /// Calculates the cumulative distribution function for the Pareto
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if x < x_m {
    ///     0
    /// } else {
    ///     1 - (x_m/x)^α
    /// }
    /// ```
    ///
    /// where `x_m` is the scale and `α` is the shape
    fn cdf(&self, x: f64) -> f64 {
        1.0 - (self.scale / x).powf(self.shape)
    }
}

impl Min<f64> for Pareto {
    /// Returns the minimum value in the domain of the Pareto distribution
    /// representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// x_m
    /// ```
    ///
    /// where `x_m` is the scale
    fn min(&self) -> f64 {
        self.scale
    }
}

impl Max<f64> for Pareto {
    /// Returns the maximum value in the domain of the Pareto distribution
    /// representable by a double precision float
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

impl Mean<f64> for Pareto {
    /// Returns the mean of the Pareto distribution
    ///
    /// # Formula
    ///
    /// if α <= 1 {
    ///     INF
    /// } else {
    ///     (α * x_m)/(α - 1)
    /// }
    /// ```
    ///
    /// where `x_m` is the scale and `α` is the shape
    fn mean(&self) -> f64 {
        if self.shape <= 1.0 {
            f64::INFINITY
        } else {
            (self.shape * self.scale) / (self.shape - 1.0)
        }
    }
}

impl Variance<f64> for Pareto {
    /// Returns the variance of the Pareto distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if α <= 2 {
    ///     INF
    /// } else {
    ///     (x_m/(α - 1))^2 * (α/(α - 2))
    /// }
    /// ```
    ///
    /// where `x_m` is the scale and `α` is the shape
    fn variance(&self) -> f64 {
        if self.shape <= 2.0 {
            f64::INFINITY
        } else {
            let a = self.scale / (self.shape - 1.0); // just a temporary variable
            a * a * self.shape / (self.shape - 2.0)
        }
    }

    /// Returns the standard deviation of the Pareto distribution
    ///
    /// # Panics
    ///
    /// If `shape <= 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// let variance = if v == INF {
    ///     σ^2
    /// } else if shape > 2.0 {
    ///     v * σ^2 / (v - 2)
    /// } else {
    ///     INF
    /// }
    /// sqrt(variance)
    /// ```
    ///
    /// where `σ` is the scale and `v` is the shape
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

// TODO: Check
impl Entropy<f64> for Pareto {
    /// Returns the entropy for the Pareto distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(α/x_m) - 1/α - 1
    /// ```
    ///
    /// where `x_m` is the scale and `α` is the shape
    fn entropy(&self) -> f64 {
        self.shape.ln() - self.scale.ln() - (1.0 / self.shape) - 1.0
    }
}

impl Skewness<f64> for Pareto {
    /// Returns the skewness of the Pareto distribution
    ///
    /// # Panics
    ///
    /// If `α <= 3.0`
    ///
    /// where `α` is the shape
    ///
    /// # Formula
    ///
    /// ```ignore
    ///     (2*(α + 1)/(α - 3))*sqrt((α - 2)/α)
    /// ```
    ///
    /// where `α` is the shape
    fn skewness(&self) -> f64 {
        assert!(
            self.shape > 3.0,
            format!("{}", StatsError::ArgGt("shape", 3.0))
        );
        (2.0 * (self.shape + 1.0) / (self.shape - 3.0)) * ((self.shape - 2.0) / self.shape).sqrt()
    }
}

impl Median<f64> for Pareto {
    /// Returns the median of the Pareto distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// x_m*2^(1/α)
    /// ```
    ///
    /// where `x_m` is the scale and `α` is the shape
    fn median(&self) -> f64 {
        self.scale * (2.0_f64.powf(1.0 / self.shape))
    }
}

impl Mode<f64> for Pareto {
    /// Returns the mode of the Pareto distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// x_m
    /// ```
    ///
    /// where `x_m` is the scale
    fn mode(&self) -> f64 {
        self.scale
    }
}

impl Continuous<f64, f64> for Pareto {
    /// Calculates the probability density function for the Pareto distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if x < x_m {
    ///     0
    /// } else {
    ///     (α * x_m^α)/(x^(α + 1))
    /// }
    /// ```
    ///
    /// where `x_m` is the scale and `α` is the shape
    fn pdf(&self, x: f64) -> f64 {
        if x < self.scale {
            0.0
        } else {
            (self.shape * self.scale.powf(self.shape)) / x.powf(self.shape + 1.0)
        }
    }

    /// Calculates the log probability density function for the Pareto
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// if x < x_m {
    ///     -INF
    /// } else {
    ///     ln(α) + α*ln(x_m) - (α + 1)*ln(x)
    /// }
    /// ```
    ///
    /// where `x_m` is the scale and `α` is the shape
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < self.scale {
            f64::NEG_INFINITY
        } else {
            self.shape.ln() + self.shape * self.scale.ln() - (self.shape + 1.0) * x.ln()
        }
    }
}
