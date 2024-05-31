use crate::distribution::{Continuous, ContinuousCDF};
use crate::statistics::{Distribution, Max, Median, Min, Mode};
use crate::{Result, StatsError};
use rand::Rng;
use std::f64;

/// Implements the [Laplace](https://en.wikipedia.org/wiki/Laplace_distribution)
/// distribution.
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Laplace, Continuous};
/// use statrs::statistics::Mode;
///
/// let n = Laplace::new(0.0, 1.0).unwrap();
/// assert_eq!(n.mode().unwrap(), 0.0);
/// assert_eq!(n.pdf(1.0), 0.18393972058572117);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Laplace {
    location: f64,
    scale: f64,
}

impl Laplace {
    /// Constructs a new laplace distribution with the given
    /// location and scale.
    ///
    /// # Errors
    ///
    /// Returns an error if location or scale are `NaN` or `scale <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Laplace;
    ///
    /// let mut result = Laplace::new(0.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = Laplace::new(0.0, -1.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(location: f64, scale: f64) -> Result<Laplace> {
        if location.is_nan() || scale.is_nan() || scale <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Laplace { location, scale })
        }
    }

    /// Returns the location of the laplace distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Laplace;
    ///
    /// let n = Laplace::new(0.0, 1.0).unwrap();
    /// assert_eq!(n.location(), 0.0);
    /// ```
    pub fn location(&self) -> f64 {
        self.location
    }

    /// Returns the scale of the laplace distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Laplace;
    ///
    /// let n = Laplace::new(0.0, 1.0).unwrap();
    /// assert_eq!(n.scale(), 1.0);
    /// ```
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl ::rand::distributions::Distribution<f64> for Laplace {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let x: f64 = rng.gen_range(-0.5..0.5);
        self.location - self.scale * x.signum() * (1. - 2. * x.abs()).ln()
    }
}

impl ContinuousCDF<f64, f64> for Laplace {
    /// Calculates the cumulative distribution function for the
    /// laplace distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (1 / 2) * (1 + signum(x - μ)) - signum(x - μ) * exp(-|x - μ| / b)
    /// ```
    ///
    /// where `μ` is the location, `b` is the scale
    fn cdf(&self, x: f64) -> f64 {
        let y = (-(x - self.location).abs() / self.scale).exp() / 2.;
        if x >= self.location {
            1. - y
        } else {
            y
        }
    }

    /// Calculates the survival function for the
    /// laplace distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// 1 - [(1 / 2) * (1 + signum(x - μ)) - signum(x - μ) * exp(-|x - μ| / b)]
    /// ```
    ///
    /// where `μ` is the location, `b` is the scale
    fn sf(&self, x: f64) -> f64 {
        let y = (-(x - self.location).abs() / self.scale).exp() / 2.;
        if x >= self.location {
            y
        } else {
            1. - y
        }
    }

    /// Calculates the inverse cumulative distribution function for the
    /// laplace distribution at `p`
    ///
    /// # Formula
    ///
    /// if p <= 1/2
    /// ```text
    /// μ + b * ln(2p)
    /// ```
    /// if p >= 1/2
    /// ```text
    /// μ - b * ln(2 - 2p)
    /// ```
    ///
    /// where `μ` is the location, `b` is the scale
    fn inverse_cdf(&self, p: f64) -> f64 {
        if p <= 0. || 1. <= p {
            panic!("p must be in [0, 1]");
        };
        if p <= 0.5 {
            self.location + self.scale * (2. * p).ln()
        } else {
            self.location - self.scale * (2. - 2. * p).ln()
        }
    }
}

impl Min<f64> for Laplace {
    /// Returns the minimum value in the domain of the laplace
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

impl Max<f64> for Laplace {
    /// Returns the maximum value in the domain of the laplace
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

impl Distribution<f64> for Laplace {
    /// Returns the mode of the laplace distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// μ
    /// ```
    ///
    /// where `μ` is the location
    fn mean(&self) -> Option<f64> {
        Some(self.location)
    }

    /// Returns the variance of the laplace distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// 2*b^2
    /// ```
    ///
    /// where `b` is the scale
    fn variance(&self) -> Option<f64> {
        Some(2. * self.scale * self.scale)
    }

    /// Returns the entropy of the laplace distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// ln(2be)
    /// ```
    ///
    /// where `b` is the scale
    fn entropy(&self) -> Option<f64> {
        Some((2. * self.scale).ln() + 1.)
    }

    /// Returns the skewness of the laplace distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// 0
    /// ```
    fn skewness(&self) -> Option<f64> {
        Some(0.)
    }
}

impl Median<f64> for Laplace {
    /// Returns the median of the laplace distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// μ
    /// ```
    ///
    /// where `μ` is the location
    fn median(&self) -> f64 {
        self.location
    }
}

impl Mode<Option<f64>> for Laplace {
    /// Returns the mode of the laplace distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// μ
    /// ```
    ///
    /// where `μ` is the location
    fn mode(&self) -> Option<f64> {
        Some(self.location)
    }
}

impl Continuous<f64, f64> for Laplace {
    /// Calculates the probability density function for the laplace
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (1 / 2b) * exp(-|x - μ| / b)
    /// ```
    /// where `μ` is the location and `b` is the scale
    fn pdf(&self, x: f64) -> f64 {
        (-(x - self.location).abs() / self.scale).exp() / (2. * self.scale)
    }

    /// Calculates the log probability density function for the laplace
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// ln((1 / 2b) * exp(-|x - μ| / b))
    /// ```
    ///
    /// where `μ` is the location and `b` is the scale
    fn ln_pdf(&self, x: f64) -> f64 {
        ((-(x - self.location).abs() / self.scale).exp() / (2. * self.scale)).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    fn try_create(location: f64, scale: f64) -> Laplace {
        let n = Laplace::new(location, scale);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn bad_create_case(location: f64, scale: f64) {
        let n = Laplace::new(location, scale);
        assert!(n.is_err());
    }

    fn test_case<F>(location: f64, scale: f64, expected: f64, eval: F)
    where
        F: Fn(Laplace) -> f64,
    {
        let n = try_create(location, scale);
        let x = eval(n);
        assert_eq!(expected, x);
    }

    fn test_is_nan<F>(location: f64, scale: f64, eval: F)
    where
        F: Fn(Laplace) -> f64,
    {
        let n = try_create(location, scale);
        let x = eval(n);
        assert!(x.is_nan());
    }

    fn test_almost<F>(location: f64, scale: f64, expected: f64, acc: f64, eval: F)
    where
        F: Fn(Laplace) -> f64,
    {
        let n = try_create(location, scale);
        let x = eval(n);
        assert_almost_eq!(expected, x, acc);
    }

    // A wrapper for the `assert_relative_eq!` macro from the approx crate.
    //
    // `rtol` is the accepable relative error.  This function is for testing
    // relative tolerance *only*.  It should not be used with `expected = 0`.
    //
    fn test_rel_close<F>(location: f64, scale: f64, expected: f64, rtol: f64, eval: F)
    where
        F: Fn(Laplace) -> f64,
    {
        let n = try_create(location, scale);
        let x = eval(n);
        assert_relative_eq!(expected, x, epsilon = 0.0, max_relative = rtol);
    }

    #[test]
    fn test_create() {
        try_create(1.0, 2.0);
        try_create(f64::NEG_INFINITY, 0.1);
        try_create(-5.0 - 1.0, 1.0);
        try_create(0.0, 5.0);
        try_create(1.0, 7.0);
        try_create(5.0, 10.0);
        try_create(f64::INFINITY, f64::INFINITY);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(2.0, -1.0);
        bad_create_case(f64::NAN, 1.0);
        bad_create_case(f64::NAN, -1.0);
    }

    #[test]
    fn test_mean() {
        let mean = |x: Laplace| x.mean().unwrap();
        test_case(f64::NEG_INFINITY, 0.1, f64::NEG_INFINITY, mean);
        test_case(-5.0 - 1.0, 1.0, -6.0, mean);
        test_case(0.0, 5.0, 0.0, mean);
        test_case(1.0, 10.0, 1.0, mean);
        test_case(f64::INFINITY, f64::INFINITY, f64::INFINITY, mean);
    }

    #[test]
    fn test_variance() {
        let variance = |x: Laplace| x.variance().unwrap();
        test_almost(f64::NEG_INFINITY, 0.1, 0.02, 1E-12, variance);
        test_almost(-5.0 - 1.0, 1.0, 2.0, 1E-12, variance);
        test_almost(0.0, 5.0, 50.0, 1E-12, variance);
        test_almost(1.0, 7.0, 98.0, 1E-12, variance);
        test_almost(5.0, 10.0, 200.0, 1E-12, variance);
        test_almost(f64::INFINITY, f64::INFINITY, f64::INFINITY, 1E-12, variance);
    }
    #[test]
    fn test_entropy() {
        let entropy = |x: Laplace| x.entropy().unwrap();
        test_almost(
            f64::NEG_INFINITY,
            0.1,
            (2.0 * f64::consts::E * 0.1).ln(),
            1E-12,
            entropy,
        );
        test_almost(-6.0, 1.0, (2.0 * f64::consts::E).ln(), 1E-12, entropy);
        test_almost(1.0, 7.0, (2.0 * f64::consts::E * 7.0).ln(), 1E-12, entropy);
        test_almost(5., 10., (2. * f64::consts::E * 10.).ln(), 1E-12, entropy);
        test_almost(f64::INFINITY, f64::INFINITY, f64::INFINITY, 1E-12, entropy);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: Laplace| x.skewness().unwrap();
        test_case(f64::NEG_INFINITY, 0.1, 0.0, skewness);
        test_case(-6.0, 1.0, 0.0, skewness);
        test_case(1.0, 7.0, 0.0, skewness);
        test_case(5.0, 10.0, 0.0, skewness);
        test_case(f64::INFINITY, f64::INFINITY, 0.0, skewness);
    }

    #[test]
    fn test_mode() {
        let mode = |x: Laplace| x.mode().unwrap();
        test_case(f64::NEG_INFINITY, 0.1, f64::NEG_INFINITY, mode);
        test_case(-6.0, 1.0, -6.0, mode);
        test_case(1.0, 7.0, 1.0, mode);
        test_case(5.0, 10.0, 5.0, mode);
        test_case(f64::INFINITY, f64::INFINITY, f64::INFINITY, mode);
    }

    #[test]
    fn test_median() {
        let median = |x: Laplace| x.median();
        test_case(f64::NEG_INFINITY, 0.1, f64::NEG_INFINITY, median);
        test_case(-6.0, 1.0, -6.0, median);
        test_case(1.0, 7.0, 1.0, median);
        test_case(5.0, 10.0, 5.0, median);
        test_case(f64::INFINITY, f64::INFINITY, f64::INFINITY, median);
    }

    #[test]
    fn test_min() {
        test_case(0.0, 1.0, f64::NEG_INFINITY, |l| l.min());
    }

    #[test]
    fn test_max() {
        test_case(0.0, 1.0, f64::INFINITY, |l| l.max());
    }

    #[test]
    fn test_density() {
        let pdf = |arg: f64| move |x: Laplace| x.pdf(arg);
        test_almost(0.0, 0.1, 1.529511602509129e-06, 1E-12, pdf(1.5));
        test_almost(1.0, 0.1, 7.614989872356341e-08, 1E-12, pdf(2.8));
        test_almost(-1.0, 0.1, 3.8905661205668983e-19, 1E-12, pdf(-5.4));
        test_almost(5.0, 0.1, 5.056107463052243e-43, 1E-12, pdf(-4.9));
        test_almost(-5.0, 0.1, 1.9877248679543235e-30, 1E-12, pdf(2.0));
        test_almost(f64::INFINITY, 0.1, 0.0, 1E-12, pdf(5.5));
        test_almost(f64::NEG_INFINITY, 0.1, 0.0, 1E-12, pdf(-0.0));
        test_almost(0.0, 1.0, 0.0, 1E-12, pdf(f64::INFINITY));
        test_almost(1.0, 1.0, 0.00915781944436709, 1E-12, pdf(5.0));
        test_almost(-1.0, 1.0, 0.5, 1E-12, pdf(-1.0));
        test_almost(5.0, 1.0, 0.0012393760883331792, 1E-12, pdf(-1.0));
        test_almost(-5.0, 1.0, 0.0002765421850739168, 1E-12, pdf(2.5));
        test_almost(f64::INFINITY, 0.1, 0.0, 1E-12, pdf(2.0));
        test_almost(f64::NEG_INFINITY, 0.1, 0.0, 1E-12, pdf(15.0));
        test_almost(0.0, f64::INFINITY, 0.0, 1E-12, pdf(89.3));
        test_almost(1.0, f64::INFINITY, 0.0, 1E-12, pdf(-0.1));
        test_almost(-1.0, f64::INFINITY, 0.0, 1E-12, pdf(0.1));
        test_almost(5.0, f64::INFINITY, 0.0, 1E-12, pdf(-6.1));
        test_almost(-5.0, f64::INFINITY, 0.0, 1E-12, pdf(-10.0));
        test_is_nan(f64::INFINITY, f64::INFINITY, pdf(2.0));
        test_is_nan(f64::NEG_INFINITY, f64::INFINITY, pdf(-5.1));
    }

    #[test]
    fn test_ln_density() {
        let ln_pdf = |arg: f64| move |x: Laplace| x.ln_pdf(arg);
        test_almost(0.0, 0.1, -13.3905620875659, 1E-12, ln_pdf(1.5));
        test_almost(1.0, 0.1, -16.390562087565897, 1E-12, ln_pdf(2.8));
        test_almost(-1.0, 0.1, -42.39056208756591, 1E-12, ln_pdf(-5.4));
        test_almost(5.0, 0.1, -97.3905620875659, 1E-12, ln_pdf(-4.9));
        test_almost(-5.0, 0.1, -68.3905620875659, 1E-12, ln_pdf(2.0));
        test_case(f64::INFINITY, 0.1, f64::NEG_INFINITY, ln_pdf(5.5));
        test_case(f64::NEG_INFINITY, 0.1, f64::NEG_INFINITY, ln_pdf(-0.0));
        test_case(0.0, 1.0, f64::NEG_INFINITY, ln_pdf(f64::INFINITY));
        test_almost(1.0, 1.0, -4.693147180559945, 1E-12, ln_pdf(5.0));
        test_almost(-1.0, 1.0, -f64::consts::LN_2, 1E-12, ln_pdf(-1.0));
        test_almost(5.0, 1.0, -6.693147180559945, 1E-12, ln_pdf(-1.0));
        test_almost(-5.0, 1.0, -8.193147180559945, 1E-12, ln_pdf(2.5));
        test_case(f64::INFINITY, 0.1, f64::NEG_INFINITY, ln_pdf(2.0));
        test_case(f64::NEG_INFINITY, 0.1, f64::NEG_INFINITY, ln_pdf(15.0));
        test_case(0.0, f64::INFINITY, f64::NEG_INFINITY, ln_pdf(89.3));
        test_case(1.0, f64::INFINITY, f64::NEG_INFINITY, ln_pdf(-0.1));
        test_case(-1.0, f64::INFINITY, f64::NEG_INFINITY, ln_pdf(0.1));
        test_case(5.0, f64::INFINITY, f64::NEG_INFINITY, ln_pdf(-6.1));
        test_case(-5.0, f64::INFINITY, f64::NEG_INFINITY, ln_pdf(-10.0));
        test_is_nan(f64::INFINITY, f64::INFINITY, ln_pdf(2.0));
        test_is_nan(f64::NEG_INFINITY, f64::INFINITY, ln_pdf(-5.1));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: Laplace| x.cdf(arg);
        let loc = 0.0f64;
        let scale = 1.0f64;
        let reltol = 1e-15f64;

        // Expected value from Wolfram Alpha: CDF[LaplaceDistribution[0, 1], 1/2].
        let expected = 0.69673467014368328819810023250440977f64;
        test_rel_close(loc, scale, expected, reltol, cdf(0.5));

        // Wolfram Alpha: CDF[LaplaceDistribution[0, 1], -1/2]
        let expected = 0.30326532985631671180189976749559023f64;
        test_rel_close(loc, scale, expected, reltol, cdf(-0.5));

        // Wolfram Alpha: CDF[LaplaceDistribution[0, 1], -100]
        let expected = 1.8600379880104179814798479019315592e-44f64;
        test_rel_close(loc, scale, expected, reltol, cdf(-100.0));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: Laplace| x.sf(arg);
        let loc = 0.0f64;
        let scale = 1.0f64;
        let reltol = 1e-15f64;

        // Expected value from Wolfram Alpha: SurvivalFunction[LaplaceDistribution[0, 1], 1/2].
        let expected = 0.30326532985631671180189976749559022f64;
        test_rel_close(loc, scale, expected, reltol, sf(0.5));

        // Wolfram Alpha: SurvivalFunction[LaplaceDistribution[0, 1], -1/2]
        let expected = 0.69673467014368328819810023250440977f64;
        test_rel_close(loc, scale, expected, reltol, sf(-0.5));

        // Wolfram Alpha: SurvivalFunction[LaplaceDistribution[0, 1], 100]
        let expected = 1.86003798801041798147984790193155916e-44;
        test_rel_close(loc, scale, expected, reltol, sf(100.0));
    }

    #[test]
    fn test_inverse_cdf() {
        let inverse_cdf = |arg: f64| move |x: Laplace| x.inverse_cdf(arg);
        let loc = 0.0f64;
        let scale = 1.0f64;
        let reltol = 1e-15f64;

        // Wolfram Alpha: Inverse CDF[LaplaceDistribution[0, 1], 1/10000000000]
        let expected = -22.3327037493805115307626824253854655f64;
        test_rel_close(loc, scale, expected, reltol, inverse_cdf(1e-10));

        // Wolfram Alpha: Inverse CDF[LaplaceDistribution[0, 1], 1/1000].
        let expected = -6.2146080984221917426367422425949161f64;
        test_rel_close(loc, scale, expected, reltol, inverse_cdf(0.001));

        // Wolfram Alpha: Inverse CDF[LaplaceDistribution[0, 1], 95/100]
        let expected = 2.3025850929940456840179914546843642f64;
        test_rel_close(loc, scale, expected, reltol, inverse_cdf(0.95));
    }

    #[test]
    fn test_sample() {
        use ::rand::distributions::Distribution;
        let l = try_create(0.1, 0.5);
        l.sample(&mut thread_rng());
    }

    #[test]
    fn test_sample_distribution() {
        use ::rand::rngs::StdRng;
        use ::rand::SeedableRng;
        use rand::distributions::Distribution;

        // sanity check sampling
        let location = 0.0;
        let scale = 1.0;
        let n = try_create(location, scale);
        let trials = 10_000;
        let tolerance = 250;

        for seed in 0..10 {
            let mut r: StdRng = SeedableRng::seed_from_u64(seed);

            let result = (0..trials).map(|_| n.sample(&mut r)).fold(0, |sum, val| {
                if val > 0.0 {
                    sum + 1
                } else if val < 0.0 {
                    sum - 1
                } else {
                    0
                }
            });
            assert!(
                result > -tolerance && result < tolerance,
                "Balance is {} for seed {}",
                result,
                seed
            );
        }
    }
}
