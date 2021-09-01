use crate::distribution::{Continuous, ContinuousCDF};
use crate::function::logistic;
use crate::statistics::*;
use crate::{consts, Result, StatsError};
use rand::Rng;
use std::f64;

/// Implements the [Logistic](https://en.wikipedia.org/wiki/Logistic_distribution)
/// distribution.
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Logistic, Continuous};
/// use statrs::statistics::Mode;
///
/// let n = Logistic::new(0.0, 1.0).unwrap();
/// assert_eq!(n.mode().unwrap(), 0.0);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Logistic {
    location: f64,
    scale: f64,
}

impl Logistic {
    /// Constructs a new logistic distribution with the given
    /// location and scale.
    ///
    /// # Errors
    ///
    /// Returns an error if location or scale are `NaN` or `scale <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Logistic;
    ///
    /// let mut result = Logistic::new(0.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = Logistic::new(0.0, -1.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(location: f64, scale: f64) -> Result<Logistic> {
        if location.is_infinite()
            || location.is_nan()
            || scale.is_infinite()
            || scale.is_nan()
            || scale <= 0.0
        {
            Err(StatsError::BadParams)
        } else {
            Ok(Logistic { location, scale })
        }
    }

    /// Returns the location of the logistic distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Logistic;
    ///
    /// let n = Logistic::new(0.0, 1.0).unwrap();
    /// assert_eq!(n.location(), 0.0);
    /// ```
    pub fn location(&self) -> f64 {
        self.location
    }

    /// Returns the scale of the logistic distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Logistic;
    ///
    /// let n = Logistic::new(0.0, 1.0).unwrap();
    /// assert_eq!(n.scale(), 1.0);
    /// ```
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl ::rand::distributions::Distribution<f64> for Logistic {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let x: f64 = rng.gen_range(0.0..1.0);
        self.location + self.scale * logistic::logit(x)
    }
}

impl ContinuousCDF<f64, f64> for Logistic {
    /// Calculates the cumulative distribution function for the
    /// logistic distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 / (1 + exp(-(x - μ) / σ))
    /// ```
    ///
    /// where `μ` is the location, `σ` is the scale
    fn cdf(&self, x: f64) -> f64 {
        logistic::logistic((x - self.location()) / self.scale)
    }
    /// Calculates the inverse cumulative distribution function for the
    /// logistic distribution at `p`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// μ + σ * ln(p / (1 - p))
    /// ```
    ///
    /// where `μ` is the location, `σ` is the scale
    fn inverse_cdf(&self, p: f64) -> f64 {
        self.location + self.scale * logistic::logit(p)
    }
}

impl Min<f64> for Logistic {
    /// Returns the minimum value in the domain of the logistic
    /// distribution representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// NEG_INF
    /// ```
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
}

impl Max<f64> for Logistic {
    /// Returns the maximum value in the domain of the logistic
    /// distribution representable by a double precision float
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

impl Distribution<f64> for Logistic {
    /// Returns the mode of the logistic distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// μ
    /// ```
    ///
    /// where `μ` is the location
    fn mean(&self) -> Option<f64> {
        Some(self.location)
    }
    /// Returns the variance of the logistic distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// σ * π^2 / 3
    /// ```
    ///
    /// where `σ` is the scale
    fn variance(&self) -> Option<f64> {
        Some(self.scale.powi(2) * consts::PI_SQ / 3.0)
    }
    /// Returns the entropy of the logistic distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(σ) + 2
    /// ```
    ///
    /// where `σ` is the scale
    fn entropy(&self) -> Option<f64> {
        Some(self.scale.ln() + 2.0)
    }
    /// Returns the skewness of the logistic distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 0
    /// ```
    fn skewness(&self) -> Option<f64> {
        Some(0.0)
    }
}

impl Median<f64> for Logistic {
    /// Returns the median of the logistic distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// μ
    /// ```
    ///
    /// where `μ` is the location
    fn median(&self) -> f64 {
        self.location
    }
}

impl Mode<Option<f64>> for Logistic {
    /// Returns the mode of the logistic distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// μ
    /// ```
    ///
    /// where `μ` is the location
    fn mode(&self) -> Option<f64> {
        Some(self.location)
    }
}

impl Continuous<f64, f64> for Logistic {
    /// Calculates the probability density function for the logistic
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// exp(-(x - μ) / σ) / (1 + exp(-(x - μ) / σ))^2 / σ
    /// ```
    /// where `μ` is the location and `σ` is the scale
    fn pdf(&self, x: f64) -> f64 {
        (-(x - self.location) / self.scale).exp()
            / (1.0 + (-(x - self.location) / self.scale).exp()).powi(2)
            / self.scale
    }

    /// Calculates the log probability density function for the logistic
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// -(x - μ) / σ - 2 * ln(1 + exp(-(x - μ) / σ)) - ln(σ)
    /// ```
    ///
    /// where `μ` is the location and `b` is the scale
    fn ln_pdf(&self, x: f64) -> f64 {
        ((-(x - self.location) / self.scale).exp()
            / (1.0 + (-(x - self.location) / self.scale).exp()).powi(2)
            / self.scale)
            .ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::INFINITY as INF;

    fn try_create(location: f64, scale: f64) -> Logistic {
        let n = Logistic::new(location, scale);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn bad_create_case(location: f64, scale: f64) {
        let n = Logistic::new(location, scale);
        assert!(n.is_err());
    }

    fn test_case<F>(location: f64, scale: f64, expected: f64, eval: F)
    where
        F: Fn(Logistic) -> f64,
    {
        let n = try_create(location, scale);
        let x = eval(n);
        assert_eq!(expected, x);
    }

    #[test]
    fn test_create() {
        try_create(1.0, 2.0);
        try_create(-5.0 - 1.0, 1.0);
        try_create(0.0, 5.0);
        try_create(1.0, 7.0);
        try_create(5.0, 10.0);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(0.0, -1.0);
        bad_create_case(0.0, f64::NAN);
        bad_create_case(0.0, f64::INFINITY);
        bad_create_case(0.0, f64::NEG_INFINITY);
        bad_create_case(f64::NAN, 1.0);
        bad_create_case(f64::INFINITY, 1.0);
        bad_create_case(f64::NEG_INFINITY, 1.0);
    }

    #[test]
    fn test_mean() {
        let mean = |x: Logistic| x.mean().unwrap();
        test_case(0.0, 0.1, 0.0, mean);
        test_case(-5.0 - 1.0, 1.0, -6.0, mean);
        test_case(0.0, 5.0, 0.0, mean);
        test_case(1.0, 10.0, 1.0, mean);
    }

    #[test]
    fn test_median() {
        let median = |x: Logistic| x.median();
        test_case(0.0, 0.1, 0.0, median);
        test_case(-5.0 - 1.0, 1.0, -6.0, median);
        test_case(0.0, 5.0, 0.0, median);
        test_case(1.0, 10.0, 1.0, median);
    }

    #[test]
    fn test_mode() {
        let mode = |x: Logistic| x.mode().unwrap();
        test_case(0.0, 0.1, 0.0, mode);
        test_case(-5.0 - 1.0, 1.0, -6.0, mode);
        test_case(0.0, 5.0, 0.0, mode);
        test_case(1.0, 10.0, 1.0, mode);
    }

    #[test]
    fn test_variance() {
        let variance = |x: Logistic| x.variance().unwrap();
        test_case(0.0, 1.0, 3.289868133696453, variance);
        test_case(-1.0, 1.0, 3.289868133696453, variance);
        test_case(0.0, 5.0, 82.24670334241132, variance);
        test_case(1.0, 10.0, 328.9868133696453, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Logistic| x.entropy().unwrap();
        test_case(0.0, 1.0, 2.0, entropy);
        test_case(-1.0, 1.0, 2.0, entropy);
        test_case(0.0, 5.0, 3.6094379124341005, entropy);
        test_case(1.0, 10.0, 4.302585092994046, entropy);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: Logistic| x.skewness().unwrap();
        test_case(0.0, 1.0, 0.0, skewness);
        test_case(-1.0, 1.0, 0.0, skewness);
        test_case(0.0, 5.0, 0.0, skewness);
        test_case(1.0, 10.0, 0.0, skewness);
    }

    #[test]
    fn test_cdf_limits() {
        let n = try_create(0.0, 1.0);
        assert_eq!(n.cdf(-INF), 0.0);
        assert_eq!(n.cdf(INF), 1.0);
    }

    #[test]
    fn test_cdf_pointwise() {
        let n = try_create(0.0, 1.0);
        assert_almost_eq!(n.cdf(-5.0), 0.0066928509242848554, 1e-5);
        assert_almost_eq!(n.cdf(-4.0), 0.01798620996209156, 1e-5);
        assert_almost_eq!(n.cdf(-3.0), 0.04742587317756678, 1e-5);
        assert_almost_eq!(n.cdf(-2.0), 0.11920292202211755, 1e-5);
        assert_almost_eq!(n.cdf(-1.0), 0.2689414213699951, 1e-5);
        assert_almost_eq!(n.cdf(0.0), 0.5, 1e-5);
        assert_almost_eq!(n.cdf(1.0), 0.7310585786300049, 1e-5);
        assert_almost_eq!(n.cdf(2.0), 0.8807970779778823, 1e-5);
        assert_almost_eq!(n.cdf(3.0), 0.9525741268224334, 1e-5);
        assert_almost_eq!(n.cdf(4.0), 0.9820137900379085, 1e-5);
        assert_almost_eq!(n.cdf(5.0), 0.9933071490757153, 1e-5);
    }

    #[test]
    fn test_pdf_pointwise() {
        let n = try_create(0.0, 1.0);
        assert_almost_eq!(n.pdf(-5.0), 0.006648056670790152, 1e-5);
        assert_almost_eq!(n.pdf(-4.0), 0.01766270621329111, 1e-5);
        assert_almost_eq!(n.pdf(-3.0), 0.04517665973091213, 1e-5);
        assert_almost_eq!(n.pdf(-2.0), 0.10499358540350652, 1e-5);
        assert_almost_eq!(n.pdf(-1.0), 0.19661193324148185, 1e-5);
        assert_almost_eq!(n.pdf(0.0), 0.25, 1e-5);
        assert_almost_eq!(n.pdf(1.0), 0.19661193324148185, 1e-5);
        assert_almost_eq!(n.pdf(2.0), 0.10499358540350652, 1e-5);
        assert_almost_eq!(n.pdf(3.0), 0.04517665973091213, 1e-5);
        assert_almost_eq!(n.pdf(4.0), 0.01766270621329111, 1e-5);
        assert_almost_eq!(n.pdf(5.0), 0.006648056670790152, 1e-5);
    }

    #[test]
    fn test_cdf_then_inverse_is_identity() {
        let n = try_create(0.0, 1.0);
        for x in -5..5 {
            assert_almost_eq!(n.inverse_cdf(n.cdf(x as f64)), x as f64, 1e-5);
        }
    }

    #[test]
    fn test_exp_ln_pdf_is_pdf() {
        let n = try_create(0.0, 1.0);
        for x in -5..5 {
            assert_almost_eq!(n.ln_pdf(x as f64).exp(), n.pdf(x as f64), 1e-5);
        }
    }

    #[test]
    fn test_sample() {
        use rand::distributions::Distribution;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let n = try_create(0.0, 1.0);
        let first_quartile = -1.0986122886681098;
        let second_quartile = 0.0;
        let third_quartile = 1.0986122886681098;
        let trials = 10_000;
        let prop_increment = 1.0 / trials as f64;

        for seed in 0..10 {
            let mut sample: f64;
            let mut first_quartile_prop = 0.0;
            let mut second_quartile_prop = 0.0;
            let mut third_quartile_prop = 0.0;
            let mut r: StdRng = SeedableRng::seed_from_u64(seed);

            for _ in 0..trials {
                sample = n.sample(&mut r);
                if sample < first_quartile {
                    first_quartile_prop += prop_increment;
                }
                if sample < second_quartile {
                    second_quartile_prop += prop_increment;
                }
                if sample < third_quartile {
                    third_quartile_prop += prop_increment;
                }
            }
            assert_almost_eq!(first_quartile_prop, 0.25, 0.02);
            assert_almost_eq!(second_quartile_prop, 0.5, 0.02);
            assert_almost_eq!(third_quartile_prop, 0.75, 0.02);
        }
    }
}
