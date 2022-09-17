use crate::distribution::{ziggurat, Continuous, ContinuousCDF};
use crate::function::erf;
use crate::statistics::*;
use crate::{consts, Result, StatsError};
use rand::Rng;
use std::f64;

/// Implements the [Normal](https://en.wikipedia.org/wiki/Normal_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Normal, Continuous};
/// use statrs::statistics::Distribution;
///
/// let n = Normal::new(0.0, 1.0).unwrap();
/// assert_eq!(n.mean().unwrap(), 0.0);
/// assert_eq!(n.pdf(1.0), 0.2419707245191433497978);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Normal {
    mean: f64,
    std_dev: f64,
}

impl Normal {
    ///  Constructs a new normal distribution with a mean of `mean`
    /// and a standard deviation of `std_dev`
    ///
    /// # Errors
    ///
    /// Returns an error if `mean` or `std_dev` are `NaN` or if
    /// `std_dev <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Normal;
    ///
    /// let mut result = Normal::new(0.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = Normal::new(0.0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(mean: f64, std_dev: f64) -> Result<Normal> {
        if mean.is_nan() || std_dev.is_nan() || std_dev <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Normal { mean, std_dev })
        }
    }
}

impl ::rand::distributions::Distribution<f64> for Normal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        sample_unchecked(rng, self.mean, self.std_dev)
    }
}

impl ContinuousCDF<f64, f64> for Normal {
    /// Calculates the cumulative distribution function for the
    /// normal distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / 2) * (1 + erf((x - μ) / (σ * sqrt(2))))
    /// ```
    ///
    /// where `μ` is the mean, `σ` is the standard deviation, and
    /// `erf` is the error function
    fn cdf(&self, x: f64) -> f64 {
        cdf_unchecked(x, self.mean, self.std_dev)
    }

    /// Calculates the survival function for the
    /// normal distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / 2) * (1 + erf(-(x - μ) / (σ * sqrt(2))))
    /// ```
    ///
    /// where `μ` is the mean, `σ` is the standard deviation, and
    /// `erf` is the error function
    ///
    /// note that this calculates the complement due to flipping
    /// the sign of the argument error function with respect to the cdf.
    ///
    /// the normal cdf Φ (and internal error function) as the following property:
    /// ```ignore
    ///  Φ(-x) + Φ(x) = 1
    ///  Φ(-x)        = 1 - Φ(x) 
    /// ```
    fn sf(&self, x: f64) -> f64 {
        sf_unchecked(x, self.mean, self.std_dev)
    }

    /// Calculates the inverse cumulative distribution function for the
    /// normal distribution at `x`
    ///
    /// # Panics
    ///
    /// If `x < 0.0` or `x > 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// μ - sqrt(2) * σ * erfc_inv(2x)
    /// ```
    ///
    /// where `μ` is the mean, `σ` is the standard deviation and `erfc_inv` is
    /// the inverse of the complementary error function
    fn inverse_cdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) {
            panic!("x must be in [0, 1]");
        } else {
            self.mean - (self.std_dev * f64::consts::SQRT_2 * erf::erfc_inv(2.0 * x))
        }
    }
}

impl Min<f64> for Normal {
    /// Returns the minimum value in the domain of the
    /// normal distribution representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// -INF
    /// ```
    fn min(&self) -> f64 {
        f64::NEG_INFINITY
    }
}

impl Max<f64> for Normal {
    /// Returns the maximum value in the domain of the
    /// normal distribution representable by a double precision float
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

impl Distribution<f64> for Normal {
    /// Returns the mean of the normal distribution
    ///
    /// # Remarks
    ///
    /// This is the same mean used to construct the distribution
    fn mean(&self) -> Option<f64> {
        Some(self.mean)
    }
    /// Returns the variance of the normal distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// σ^2
    /// ```
    ///
    /// where `σ` is the standard deviation
    fn variance(&self) -> Option<f64> {
        Some(self.std_dev * self.std_dev)
    }
    /// Returns the entropy of the normal distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / 2) * ln(2σ^2 * π * e)
    /// ```
    ///
    /// where `σ` is the standard deviation
    fn entropy(&self) -> Option<f64> {
        Some(self.std_dev.ln() + consts::LN_SQRT_2PIE)
    }
    /// Returns the skewness of the normal distribution
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

impl Median<f64> for Normal {
    /// Returns the median of the normal distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// μ
    /// ```
    ///
    /// where `μ` is the mean
    fn median(&self) -> f64 {
        self.mean
    }
}

impl Mode<Option<f64>> for Normal {
    /// Returns the mode of the normal distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// μ
    /// ```
    ///
    /// where `μ` is the mean
    fn mode(&self) -> Option<f64> {
        Some(self.mean)
    }
}

impl Continuous<f64, f64> for Normal {
    /// Calculates the probability density function for the normal distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / sqrt(2σ^2 * π)) * e^(-(x - μ)^2 / 2σ^2)
    /// ```
    ///
    /// where `μ` is the mean and `σ` is the standard deviation
    fn pdf(&self, x: f64) -> f64 {
        pdf_unchecked(x, self.mean, self.std_dev)
    }

    /// Calculates the log probability density function for the normal
    /// distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln((1 / sqrt(2σ^2 * π)) * e^(-(x - μ)^2 / 2σ^2))
    /// ```
    ///
    /// where `μ` is the mean and `σ` is the standard deviation
    fn ln_pdf(&self, x: f64) -> f64 {
        ln_pdf_unchecked(x, self.mean, self.std_dev)
    }
}

/// performs an unchecked cdf calculation for a normal distribution
/// with the given mean and standard deviation at x
pub fn cdf_unchecked(x: f64, mean: f64, std_dev: f64) -> f64 {
    0.5 * erf::erfc((mean - x) / (std_dev * f64::consts::SQRT_2))
}

/// performs an unchecked sf calculation for a normal distribution
/// with the given mean and standard deviation at x
pub fn sf_unchecked(x: f64, mean: f64, std_dev: f64) -> f64 {
    0.5 * erf::erfc((x - mean) / (std_dev * f64::consts::SQRT_2))
}

/// performs an unchecked pdf calculation for a normal distribution
/// with the given mean and standard deviation at x
pub fn pdf_unchecked(x: f64, mean: f64, std_dev: f64) -> f64 {
    let d = (x - mean) / std_dev;
    (-0.5 * d * d).exp() / (consts::SQRT_2PI * std_dev)
}

/// performs an unchecked log(pdf) calculation for a normal distribution
/// with the given mean and standard deviation at x
pub fn ln_pdf_unchecked(x: f64, mean: f64, std_dev: f64) -> f64 {
    let d = (x - mean) / std_dev;
    (-0.5 * d * d) - consts::LN_SQRT_2PI - std_dev.ln()
}

/// draws a sample from a normal distribution using the Box-Muller algorithm
pub fn sample_unchecked<R: Rng + ?Sized>(rng: &mut R, mean: f64, std_dev: f64) -> f64 {
    mean + std_dev * ziggurat::sample_std_normal(rng)
}

#[rustfmt::skip]
#[cfg(all(test, feature = "nightly"))]
mod tests {
    use crate::statistics::*;
    use crate::distribution::{ContinuousCDF, Continuous, Normal};
    use crate::distribution::internal::*;
    use crate::consts::ACC;

    fn try_create(mean: f64, std_dev: f64) -> Normal {
        let n = Normal::new(mean, std_dev);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(mean: f64, std_dev: f64) {
        let n = try_create(mean, std_dev);
        assert_eq!(mean, n.mean().unwrap());
        assert_eq!(std_dev, n.std_dev().unwrap());
    }

    fn bad_create_case(mean: f64, std_dev: f64) {
        let n = Normal::new(mean, std_dev);
        assert!(n.is_err());
    }

    fn test_case<F>(mean: f64, std_dev: f64, expected: f64, eval: F)
        where F: Fn(Normal) -> f64
    {
        let n = try_create(mean, std_dev);
        let x = eval(n);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(mean: f64, std_dev: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Normal) -> f64
    {
        let n = try_create(mean, std_dev);
        let x = eval(n);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_create() {
        create_case(10.0, 0.1);
        create_case(-5.0, 1.0);
        create_case(0.0, 10.0);
        create_case(10.0, 100.0);
        create_case(-5.0, f64::INFINITY);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(0.0, 0.0);
        bad_create_case(f64::NAN, 1.0);
        bad_create_case(1.0, f64::NAN);
        bad_create_case(f64::NAN, f64::NAN);
        bad_create_case(1.0, -1.0);
    }

    #[test]
    fn test_variance() {
        let variance = |x: Normal| x.variance().unwrap();
        test_case(0.0, 0.1, 0.1 * 0.1, variance);
        test_case(0.0, 1.0, 1.0, variance);
        test_case(0.0, 10.0, 100.0, variance);
        test_case(0.0, f64::INFINITY, f64::INFINITY, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: Normal| x.entropy().unwrap();
        test_almost(0.0, 0.1, -0.8836465597893729422377, 1e-15, entropy);
        test_case(0.0, 1.0, 1.41893853320467274178, entropy);
        test_case(0.0, 10.0, 3.721523626198718425798, entropy);
        test_case(0.0, f64::INFINITY, f64::INFINITY, entropy);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: Normal| x.skewness().unwrap();
        test_case(0.0, 0.1, 0.0, skewness);
        test_case(4.0, 1.0, 0.0, skewness);
        test_case(0.3, 10.0, 0.0, skewness);
        test_case(0.0, f64::INFINITY, 0.0, skewness);
    }

    #[test]
    fn test_mode() {
        let mode = |x: Normal| x.mode().unwrap();
        test_case(-0.0, 1.0, 0.0, mode);
        test_case(0.0, 1.0, 0.0, mode);
        test_case(0.1, 1.0, 0.1, mode);
        test_case(1.0, 1.0, 1.0, mode);
        test_case(-10.0, 1.0, -10.0, mode);
        test_case(f64::INFINITY, 1.0, f64::INFINITY, mode);
    }

    #[test]
    fn test_median() {
        let median = |x: Normal| x.median();
        test_case(-0.0, 1.0, 0.0, median);
        test_case(0.0, 1.0, 0.0, median);
        test_case(0.1, 1.0, 0.1, median);
        test_case(1.0, 1.0, 1.0, median);
        test_case(-0.0, 1.0, -0.0, median);
        test_case(f64::INFINITY, 1.0, f64::INFINITY, median);
    }

    #[test]
    fn test_min_max() {
        let min = |x: Normal| x.min();
        let max = |x: Normal| x.max();
        test_case(0.0, 0.1, f64::NEG_INFINITY, min);
        test_case(-3.0, 10.0, f64::NEG_INFINITY, min);
        test_case(0.0, 0.1, f64::INFINITY, max);
        test_case(-3.0, 10.0, f64::INFINITY, max);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: f64| move |x: Normal| x.pdf(arg);
        test_almost(10.0, 0.1, 5.530709549844416159162E-49, 1e-64, pdf(8.5));
        test_almost(10.0, 0.1, 0.5399096651318805195056, 1e-14, pdf(9.8));
        test_almost(10.0, 0.1, 3.989422804014326779399, 1e-15, pdf(10.0));
        test_almost(10.0, 0.1, 0.5399096651318805195056, 1e-14, pdf(10.2));
        test_almost(10.0, 0.1, 5.530709549844416159162E-49, 1e-64, pdf(11.5));
        test_case(-5.0, 1.0, 1.486719514734297707908E-6, pdf(-10.0));
        test_case(-5.0, 1.0, 0.01752830049356853736216, pdf(-7.5));
        test_almost(-5.0, 1.0, 0.3989422804014326779399, 1e-16, pdf(-5.0));
        test_case(-5.0, 1.0, 0.01752830049356853736216, pdf(-2.5));
        test_case(-5.0, 1.0, 1.486719514734297707908E-6, pdf(0.0));
        test_case(0.0, 10.0, 0.03520653267642994777747, pdf(-5.0));
        test_almost(0.0, 10.0, 0.03866681168028492069412, 1e-17, pdf(-2.5));
        test_almost(0.0, 10.0, 0.03989422804014326779399, 1e-17, pdf(0.0));
        test_almost(0.0, 10.0, 0.03866681168028492069412, 1e-17, pdf(2.5));
        test_case(0.0, 10.0, 0.03520653267642994777747, pdf(5.0));
        test_almost(10.0, 100.0, 4.398359598042719404845E-4, 1e-19, pdf(-200.0));
        test_case(10.0, 100.0, 0.002178521770325505313831, pdf(-100.0));
        test_case(10.0, 100.0, 0.003969525474770117655105, pdf(0.0));
        test_almost(10.0, 100.0, 0.002660852498987548218204, 1e-18, pdf(100.0));
        test_case(10.0, 100.0, 6.561581477467659126534E-4, pdf(200.0));
        test_case(-5.0, f64::INFINITY, 0.0, pdf(-5.0));
        test_case(-5.0, f64::INFINITY, 0.0, pdf(0.0));
        test_case(-5.0, f64::INFINITY, 0.0, pdf(100.0));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: Normal| x.ln_pdf(arg);
        test_almost(10.0, 0.1, (5.530709549844416159162E-49f64).ln(), 1e-13, ln_pdf(8.5));
        test_almost(10.0, 0.1, (0.5399096651318805195056f64).ln(), 1e-13, ln_pdf(9.8));
        test_almost(10.0, 0.1, (3.989422804014326779399f64).ln(), 1e-15, ln_pdf(10.0));
        test_almost(10.0, 0.1, (0.5399096651318805195056f64).ln(), 1e-13, ln_pdf(10.2));
        test_almost(10.0, 0.1, (5.530709549844416159162E-49f64).ln(), 1e-13, ln_pdf(11.5));
        test_case(-5.0, 1.0, (1.486719514734297707908E-6f64).ln(), ln_pdf(-10.0));
        test_case(-5.0, 1.0, (0.01752830049356853736216f64).ln(), ln_pdf(-7.5));
        test_almost(-5.0, 1.0, (0.3989422804014326779399f64).ln(), 1e-15, ln_pdf(-5.0));
        test_case(-5.0, 1.0, (0.01752830049356853736216f64).ln(), ln_pdf(-2.5));
        test_case(-5.0, 1.0, (1.486719514734297707908E-6f64).ln(), ln_pdf(0.0));
        test_case(0.0, 10.0, (0.03520653267642994777747f64).ln(), ln_pdf(-5.0));
        test_case(0.0, 10.0, (0.03866681168028492069412f64).ln(), ln_pdf(-2.5));
        test_case(0.0, 10.0, (0.03989422804014326779399f64).ln(), ln_pdf(0.0));
        test_case(0.0, 10.0, (0.03866681168028492069412f64).ln(), ln_pdf(2.5));
        test_case(0.0, 10.0, (0.03520653267642994777747f64).ln(), ln_pdf(5.0));
        test_case(10.0, 100.0, (4.398359598042719404845E-4f64).ln(), ln_pdf(-200.0));
        test_case(10.0, 100.0, (0.002178521770325505313831f64).ln(), ln_pdf(-100.0));
        test_almost(10.0, 100.0, (0.003969525474770117655105f64).ln(),1e-15, ln_pdf(0.0));
        test_almost(10.0, 100.0, (0.002660852498987548218204f64).ln(), 1e-15, ln_pdf(100.0));
        test_almost(10.0, 100.0, (6.561581477467659126534E-4f64).ln(), 1e-15, ln_pdf(200.0));
        test_case(-5.0, f64::INFINITY, f64::NEG_INFINITY, ln_pdf(-5.0));
        test_case(-5.0, f64::INFINITY, f64::NEG_INFINITY, ln_pdf(0.0));
        test_case(-5.0, f64::INFINITY, f64::NEG_INFINITY, ln_pdf(100.0));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: Normal| x.cdf(arg);
        test_case(5.0, 2.0, 0.0, cdf(f64::NEG_INFINITY));
        test_almost(5.0, 2.0, 0.0000002866515718, 1e-16, cdf(-5.0));
        test_almost(5.0, 2.0, 0.0002326290790, 1e-13, cdf(-2.0));
        test_almost(5.0, 2.0, 0.006209665325, 1e-12, cdf(0.0));
        test_case(5.0, 2.0, 0.30853753872598689636229538939166226011639782444542207, cdf(4.0));
        test_case(5.0, 2.0, 0.5, cdf(5.0));
        test_case(5.0, 2.0, 0.69146246127401310363770461060833773988360217555457859, cdf(6.0));
        test_almost(5.0, 2.0, 0.993790334674, 1e-12, cdf(10.0));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: Normal| x.sf(arg);
        test_case(5.0, 2.0, 1.0, sf(f64::NEG_INFINITY));
        test_almost(5.0, 2.0, 0.9999997133484281, 1e-16, sf(-5.0));
        test_almost(5.0, 2.0, 0.9997673709209455, 1e-13, sf(-2.0));
        test_almost(5.0, 2.0, 0.9937903346744879, 1e-12, sf(0.0));
        test_case(5.0, 2.0, 0.6914624612740131, sf(4.0));
        test_case(5.0, 2.0, 0.5, sf(5.0));
        test_case(5.0, 2.0, 0.3085375387259869, sf(6.0));
        test_almost(5.0, 2.0, 0.006209665325512148, 1e-12, sf(10.0));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&try_create(0.0, 1.0), -10.0, 10.0);
        test::check_continuous_distribution(&try_create(20.0, 0.5), 10.0, 30.0);
    }

    #[test]
    fn test_inverse_cdf() {
        let inverse_cdf = |arg: f64| move |x: Normal| x.inverse_cdf(arg);
        test_case(5.0, 2.0, f64::NEG_INFINITY, inverse_cdf( 0.0));
        test_almost(5.0, 2.0, -5.0, 1e-14, inverse_cdf(0.00000028665157187919391167375233287464535385442301361187883));
        test_almost(5.0, 2.0, -2.0, 1e-14, inverse_cdf(0.0002326290790355250363499258867279847735487493358890356));
        test_almost(5.0, 2.0, -0.0, 1e-14, inverse_cdf(0.0062096653257761351669781045741922211278977469230927036));
        test_almost(5.0, 2.0, 0.0, 1e-14, inverse_cdf(0.0062096653257761351669781045741922211278977469230927036));
        test_almost(5.0, 2.0, 4.0, 1e-14, inverse_cdf(0.30853753872598689636229538939166226011639782444542207));
        test_almost(5.0, 2.0, 5.0, 1e-14, inverse_cdf(0.5));
        test_almost(5.0, 2.0, 6.0, 1e-14, inverse_cdf(0.69146246127401310363770461060833773988360217555457859));
        test_almost(5.0, 2.0, 10.0, 1e-14, inverse_cdf(0.9937903346742238648330218954258077788721022530769078));
        test_case(5.0, 2.0, f64::INFINITY, inverse_cdf(1.0));
    }
}
