use Result;
use distribution::{Continuous, Distribution, Gamma, Univariate};
use rand::Rng;
use rand::distributions::{IndependentSample, Sample};
use statistics::*;
use std::f64;

/// Implements the
/// [Chi-squared](https://en.wikipedia.org/wiki/Chi-squared_distribution)
/// distribution which is a special case of the
/// [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) distribution
/// (referenced [Here](./struct.Gamma.html))
///
/// # Examples
///
/// ```
/// use statrs::distribution::{ChiSquared, Continuous};
/// use statrs::statistics::Mean;
/// use statrs::prec;
///
/// let n = ChiSquared::new(3.0).unwrap();
/// assert_eq!(n.mean(), 3.0);
/// assert!(prec::almost_eq(n.pdf(4.0), 0.107981933026376103901, 1e-15));
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ChiSquared {
    freedom: f64,
    g: Gamma,
}

impl ChiSquared {
    /// Constructs a new chi-squared distribution with `freedom`
    /// degrees of freedom. This is equivalent to a Gamma distribution
    /// with a shape of `freedom / 2.0` and a rate of `0.5`.
    ///
    /// # Errors
    ///
    /// Returns an error if `freedom` is `NaN` or less than
    /// or equal to `0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::ChiSquared;
    ///
    /// let mut result = ChiSquared::new(3.0);
    /// assert!(result.is_ok());
    ///
    /// result = ChiSquared::new(0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(freedom: f64) -> Result<ChiSquared> {
        Gamma::new(freedom / 2.0, 0.5).map(|g| {
            ChiSquared {
                freedom: freedom,
                g: g,
            }
        })
    }

    /// Returns the degrees of freedom of the chi-squared
    /// distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::ChiSquared;
    ///
    /// let n = ChiSquared::new(3.0).unwrap();
    /// assert_eq!(n.freedom(), 3.0);
    /// ```
    pub fn freedom(&self) -> f64 {
        self.freedom
    }

    /// Returns the shape of the underlying Gamma distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::ChiSquared;
    ///
    /// let n = ChiSquared::new(3.0).unwrap();
    /// assert_eq!(n.shape(), 3.0 / 2.0);
    /// ```
    pub fn shape(&self) -> f64 {
        self.g.shape()
    }

    /// Returns the rate of the underlying Gamma distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::ChiSquared;
    ///
    /// let n = ChiSquared::new(3.0).unwrap();
    /// assert_eq!(n.rate(), 0.5);
    /// ```
    pub fn rate(&self) -> f64 {
        self.g.rate()
    }
}

impl Sample<f64> for ChiSquared {
    /// Generate a random sample from a chi-squared
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn sample<R: Rng>(&mut self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl IndependentSample<f64> for ChiSquared {
    /// Generate a random independent sample from a Chi
    /// distribution using `r` as the source of randomness.
    /// Refer [here](#method.sample-1) for implementation details
    fn ind_sample<R: Rng>(&self, r: &mut R) -> f64 {
        super::Distribution::sample(self, r)
    }
}

impl Distribution<f64> for ChiSquared {
    /// Generate a random sample from the chi-squared distribution
    /// using `r` as the source of randomness
    ///
    /// # Examples
    ///
    /// ```
    /// # extern crate rand;
    /// # extern crate statrs;
    /// use rand::StdRng;
    /// use statrs::distribution::{ChiSquared, Distribution};
    ///
    /// # fn main() {
    /// let mut r = rand::StdRng::new().unwrap();
    /// let n = ChiSquared::new(3.0).unwrap();
    /// print!("{}", n.sample::<StdRng>(&mut r));
    /// # }
    /// ```
    fn sample<R: Rng>(&self, r: &mut R) -> f64 {
        self.g.sample(r)
    }
}

impl Univariate<f64, f64> for ChiSquared {
    /// Calculates the cumulative distribution function for the
    /// chi-squared distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / Γ(k / 2)) * γ(k / 2, x / 2)
    /// ```
    ///
    /// where `k` is the degrees of freedom, `Γ` is the gamma function,
    /// and `γ` is the lower incomplete gamma function
    fn cdf(&self, x: f64) -> f64 {
        self.g.cdf(x)
    }
}

impl Min<f64> for ChiSquared {
    /// Returns the minimum value in the domain of the
    /// chi-squared distribution representable by a double precision
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

impl Max<f64> for ChiSquared {
    /// Returns the maximum value in the domain of the
    /// chi-squared distribution representable by a double precision
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

impl Mean<f64> for ChiSquared {
    /// Returns the mean of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// k
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn mean(&self) -> f64 {
        self.g.mean()
    }
}

impl Variance<f64> for ChiSquared {
    /// Returns the variance of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 2k
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn variance(&self) -> f64 {
        self.g.variance()
    }

    /// Returns the standard deviation of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(2k)
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn std_dev(&self) -> f64 {
        self.g.std_dev()
    }
}

impl Entropy<f64> for ChiSquared {
    /// Returns the entropy of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (k / 2) + ln(2 * Γ(k / 2)) + (1 - (k / 2)) * ψ(k / 2)
    /// ```
    ///
    /// where `k` is the degrees of freedom, `Γ` is the gamma function,
    /// and `ψ` is the digamma function
    fn entropy(&self) -> f64 {
        self.g.entropy()
    }
}

impl Skewness<f64> for ChiSquared {
    /// Returns the skewness of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(8 / k)
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn skewness(&self) -> f64 {
        self.g.skewness()
    }
}

impl Median<f64> for ChiSquared {
    /// Returns the median  of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// k * (1 - (2 / 9k))^3
    /// ```
    fn median(&self) -> f64 {
        if self.freedom < 1.0 {
            // if k is small, calculate using expansion of formula
            self.freedom - 2.0 / 3.0 + 12.0 / (81.0 * self.freedom) - 8.0 / (729.0 * self.freedom * self.freedom)
        } else {
            // if k is large enough, median heads toward k - 2/3
            self.freedom - 2.0 / 3.0
        }
    }
}

impl Mode<f64> for ChiSquared {
    /// Returns the mode of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// max(k - 2, 0)
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn mode(&self) -> f64 {
        self.g.mode().max(0.0)
    }
}

impl Continuous<f64, f64> for ChiSquared {
    /// Calculates the probability density function for the chi-squared
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 / (2^(k / 2) * Γ(k / 2)) * x^((k / 2) - 1) * e^(-x / 2)
    /// ```
    ///
    /// where `k` is the degrees of freedom and `Γ` is the gamma function
    fn pdf(&self, x: f64) -> f64 {
        if x > 0.0 {
            self.g.pdf(x)
        } else {
            0.0
        }
    }

    /// Calculates the log probability density function for the chi-squared
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(1 / (2^(k / 2) * Γ(k / 2)) * x^((k / 2) - 1) * e^(-x / 2))
    /// ```
    fn ln_pdf(&self, x: f64) -> f64 {
        if x > 0.0 {
            self.g.ln_pdf(x)
        } else {
            -f64::INFINITY
        }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use statistics::*;
    use distribution::{ChiSquared, Continuous};
    use distribution::internal::*;

    fn try_create(freedom: f64) -> ChiSquared {
        let n = ChiSquared::new(freedom);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn test_case<F>(freedom: f64, expected: f64, eval: F)
        where F: Fn(ChiSquared) -> f64
    {
        let n = try_create(freedom);
        let x = eval(n);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(freedom: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(ChiSquared) -> f64
    {
        let n = try_create(freedom);
        let x = eval(n);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_mean() {
        test_case(1.0, 1.0, |x| x.mean());
        test_case(2.1, 2.1, |x| x.mean());
        test_case(3.0, 3.0, |x| x.mean());
        test_case(4.5, 4.5, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_case(1.0, 2.0, |x| x.variance());
        test_case(2.1, 4.2, |x| x.variance());
        test_case(3.0, 6.0, |x| x.variance());
        test_case(4.5, 9.0, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_case(1.0, 2f64.sqrt(), |x| x.std_dev());
        test_case(2.1, 4.2f64.sqrt(), |x| x.std_dev());
        test_case(3.0, 6f64.sqrt(), |x| x.std_dev());
        test_case(4.5, 3.0, |x| x.std_dev());
    }

    #[test]
    fn test_skewness() {
        test_almost(1.0, 8f64.sqrt(), 1e-15, |x| x.skewness());
        test_almost(2.1, (8f64/2.1).sqrt(), 1e-15, |x| x.skewness());
        test_almost(3.0, (8f64/3.0).sqrt(),  1e-15, |x| x.skewness());
        test_almost(4.5, (8f64/4.5).sqrt(),  1e-15, |x| x.skewness());
    }

    #[test]
    fn test_mode() {
        test_case(1.0, 0.0, |x| x.mode());
        test_case(2.0, 0.0, |x| x.mode());
        test_case(3.0, 1.0, |x| x.mode());
        test_case(4.5, 2.5, |x| x.mode());
        test_case(10.0, 8.0, |x| x.mode());
    }

    #[test]
    fn test_median() {
        test_almost(0.5, 0.0857338820301783264746, 1e-16, |x| x.median());
        test_case(1.0, 1.0 - 2.0 / 3.0, |x| x.median());
        test_case(2.0, 2.0 - 2.0 / 3.0, |x| x.median());
        test_case(2.5, 2.5 - 2.0 / 3.0, |x| x.median());
        test_case(3.0, 3.0 - 2.0 / 3.0, |x| x.median());
    }

    #[test]
    fn test_min_max() {
        test_case(1.0, 0.0, |x| x.min());
        test_case(1.0, f64::INFINITY, |x| x.max());
    }

    #[test]
    fn test_entropy() {
        test_almost(0.1, -15.760926360123200, 1e-13, |x| x.entropy());
        test_almost(1.0, 0.783757110473934, 1e-15, |x| x.entropy());
        test_almost(2.0, 1.693147180559945, 1e-15, |x| x.entropy());
        test_almost(4.0, 2.270362845461478, 1e-13, |x| x.entropy());
        test_almost(16.0, 3.108818195936093, 1e-13, |x| x.entropy());
        test_almost(100.0, 4.061397128938097, 1e-13, |x| x.entropy());
    }

    #[test]
    fn test_pdf() {
        test_case(1.0, 0.0, |x| x.pdf(0.0));
        test_almost(1.0, 1.2000389484301359798, 1e-15, |x| x.pdf(0.1));
        test_almost(1.0, 0.24197072451914334980, 1e-15, |x| x.pdf(1.0));
        test_almost(1.0, 0.010874740337283141714, 1e-15, |x| x.pdf(5.5));
        test_almost(1.0, 4.7000792147504127122e-26, 1e-38, |x| x.pdf(110.1));
        test_case(1.0, 0.0, |x| x.pdf(f64::INFINITY));
        test_case(2.0, 0.0, |x| x.pdf(0.0));
        test_almost(2.0, 0.47561471225035700455, 1e-15, |x| x.pdf(0.1));
        test_almost(2.0, 0.30326532985631671180, 1e-15, |x| x.pdf(1.0));
        test_almost(2.0, 0.031963930603353786351, 1e-15, |x| x.pdf(5.5));
        test_almost(2.0, 6.1810004550085248492e-25, 1e-37, |x| x.pdf(110.1));
        test_case(2.0, 0.0, |x| x.pdf(f64::INFINITY));
        test_case(2.5, 0.0, |x| x.pdf(0.0));
        test_almost(2.5, 0.24812852712543073541, 1e-15, |x| x.pdf(0.1));
        test_almost(2.5, 0.28134822576318228131, 1e-15, |x| x.pdf(1.0));
        test_almost(2.5, 0.045412171451573920401, 1e-15, |x| x.pdf(5.5));
        test_almost(2.5, 1.8574923023527248767e-24, 1e-36, |x| x.pdf(110.1));
        test_case(2.5, 0.0, |x| x.pdf(f64::INFINITY));
        // test_case(f64::INFINITY, 0.0, |x| x.pdf(0.0));
        // test_case(f64::INFINITY, 0.0, |x| x.pdf(0.1));
        // test_case(f64::INFINITY, 0.0, |x| x.pdf(1.0));
        // test_case(f64::INFINITY, 0.0, |x| x.pdf(5.5));
        // test_case(f64::INFINITY, 0.0, |x| x.pdf(110.1));
        // test_case(f64::INFINITY, 0.0, |x| x.pdf(f64::INFINITY));
    }

    #[test]
    fn test_ln_pdf() {
        test_case(1.0, -f64::INFINITY, |x| x.ln_pdf(0.0));
        test_almost(1.0, 0.18235401329235010023, 1e-15, |x| x.ln_pdf(0.1));
        test_almost(1.0, -1.4189385332046727418, 1e-15, |x| x.ln_pdf(1.0));
        test_almost(1.0, -4.5213125793238853591, 1e-15, |x| x.ln_pdf(5.5));
        test_almost(1.0, -58.319633055068989881, 1e-13, |x| x.ln_pdf(110.1));
        test_case(1.0, -f64::INFINITY, |x| x.ln_pdf(f64::INFINITY));
        test_case(2.0, -f64::INFINITY, |x| x.ln_pdf(0.0));
        test_almost(2.0, -0.74314718055994530942, 1e-15, |x| x.ln_pdf(0.1));
        test_almost(2.0, -1.1931471805599453094, 1e-15, |x| x.ln_pdf(1.0));
        test_almost(2.0, -3.4431471805599453094, 1e-15, |x| x.ln_pdf(5.5));
        test_almost(2.0, -55.743147180559945309, 1e-13, |x| x.ln_pdf(110.1));
        test_case(2.0, -f64::INFINITY, |x| x.ln_pdf(f64::INFINITY));
        test_case(2.5, -f64::INFINITY, |x| x.ln_pdf(0.0));
        test_almost(2.5, -1.3938084125266298963, 1e-15, |x| x.ln_pdf(0.1));
        test_almost(2.5, -1.2681621392781184753, 1e-15, |x| x.ln_pdf(1.0));
        test_almost(2.5, -3.0919751162185121666, 1e-15, |x| x.ln_pdf(5.5));
        test_almost(2.5, -54.642814878345959906, 1e-13, |x| x.ln_pdf(110.1));
        test_case(2.5, -f64::INFINITY, |x| x.ln_pdf(f64::INFINITY));
        // test_case(f64::INFINITY, -f64::INFINITY, |x| x.ln_pdf(0.0));
        // test_case(f64::INFINITY, -f64::INFINITY, |x| x.ln_pdf(0.1));
        // test_case(f64::INFINITY, -f64::INFINITY, |x| x.ln_pdf(1.0));
        // test_case(f64::INFINITY, -f64::INFINITY, |x| x.ln_pdf(5.5));
        // test_case(f64::INFINITY, -f64::INFINITY, |x| x.ln_pdf(110.1));
        // test_case(f64::INFINITY, -f64::INFINITY, |x| x.ln_pdf(f64::INFINITY));
    }

    #[test]
    fn test_continuous() {
        // Cannot test for `freedom == 1.0`. The pdf is very steep near 0.0,
        // leading to problems with numerical integration
        test::check_continuous_distribution(&try_create(1.5), 0.0, 10.0);
        test::check_continuous_distribution(&try_create(2.0), 0.0, 10.0);
        test::check_continuous_distribution(&try_create(5.0), 0.0, 50.0);
    }
}
