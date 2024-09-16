use crate::distribution::{Continuous, ContinuousCDF, Gamma, GammaError};
use crate::statistics::*;
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
/// use statrs::statistics::Distribution;
/// use statrs::prec;
///
/// let n = ChiSquared::new(3.0).unwrap();
/// assert_eq!(n.mean().unwrap(), 3.0);
/// assert!(prec::almost_eq(n.pdf(4.0), 0.107981933026376103901, 1e-15));
/// ```
#[derive(Copy, Clone, PartialEq, Debug)]
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
    pub fn new(freedom: f64) -> Result<ChiSquared, GammaError> {
        Gamma::new(freedom / 2.0, 0.5).map(|g| ChiSquared { freedom, g })
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

impl std::fmt::Display for ChiSquared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "χ^2_{}", self.freedom)
    }
}

#[cfg(feature = "rand")]
#[cfg_attr(docsrs, doc(cfg(feature = "rand")))]
impl ::rand::distributions::Distribution<f64> for ChiSquared {
    fn sample<R: ::rand::Rng + ?Sized>(&self, r: &mut R) -> f64 {
        ::rand::distributions::Distribution::sample(&self.g, r)
    }
}

impl ContinuousCDF<f64, f64> for ChiSquared {
    /// Calculates the cumulative distribution function for the
    /// chi-squared distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (1 / Γ(k / 2)) * γ(k / 2, x / 2)
    /// ```
    ///
    /// where `k` is the degrees of freedom, `Γ` is the gamma function,
    /// and `γ` is the lower incomplete gamma function
    fn cdf(&self, x: f64) -> f64 {
        self.g.cdf(x)
    }

    /// Calculates the cumulative distribution function for the
    /// chi-squared distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// (1 / Γ(k / 2)) * γ(k / 2, x / 2)
    /// ```
    ///
    /// where `k` is the degrees of freedom, `Γ` is the gamma function,
    /// and `γ` is the upper incomplete gamma function
    fn sf(&self, x: f64) -> f64 {
        self.g.sf(x)
    }

    /// Calculates the inverse cumulative distribution function for the
    /// chi-squared distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// γ^{-1}(k / 2, x * Γ(k / 2) / 2)
    /// ```
    ///
    /// where `k` is the degrees of freedom, `Γ` is the gamma function,
    /// and `γ` is the lower incomplete gamma function
    fn inverse_cdf(&self, p: f64) -> f64 {
        self.g.inverse_cdf(p)
    }
}

impl Min<f64> for ChiSquared {
    /// Returns the minimum value in the domain of the
    /// chi-squared distribution representable by a double precision
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

impl Max<f64> for ChiSquared {
    /// Returns the maximum value in the domain of the
    /// chi-squared distribution representable by a double precision
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

impl Distribution<f64> for ChiSquared {
    /// Returns the mean of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// k
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn mean(&self) -> Option<f64> {
        self.g.mean()
    }

    /// Returns the variance of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// 2k
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn variance(&self) -> Option<f64> {
        self.g.variance()
    }

    /// Returns the entropy of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// (k / 2) + ln(2 * Γ(k / 2)) + (1 - (k / 2)) * ψ(k / 2)
    /// ```
    ///
    /// where `k` is the degrees of freedom, `Γ` is the gamma function,
    /// and `ψ` is the digamma function
    fn entropy(&self) -> Option<f64> {
        self.g.entropy()
    }

    /// Returns the skewness of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// sqrt(8 / k)
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn skewness(&self) -> Option<f64> {
        self.g.skewness()
    }
}

impl Median<f64> for ChiSquared {
    /// Returns the median  of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// k * (1 - (2 / 9k))^3
    /// ```
    fn median(&self) -> f64 {
        if self.freedom < 1.0 {
            // if k is small, calculate using expansion of formula
            self.freedom - 2.0 / 3.0 + 12.0 / (81.0 * self.freedom)
                - 8.0 / (729.0 * self.freedom * self.freedom)
        } else {
            // if k is large enough, median heads toward k - 2/3
            self.freedom - 2.0 / 3.0
        }
    }
}

impl Mode<Option<f64>> for ChiSquared {
    /// Returns the mode of the chi-squared distribution
    ///
    /// # Formula
    ///
    /// ```text
    /// k - 2
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn mode(&self) -> Option<f64> {
        self.g.mode()
    }
}

impl Continuous<f64, f64> for ChiSquared {
    /// Calculates the probability density function for the chi-squared
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// 1 / (2^(k / 2) * Γ(k / 2)) * x^((k / 2) - 1) * e^(-x / 2)
    /// ```
    ///
    /// where `k` is the degrees of freedom and `Γ` is the gamma function
    fn pdf(&self, x: f64) -> f64 {
        self.g.pdf(x)
    }

    /// Calculates the log probability density function for the chi-squared
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```text
    /// ln(1 / (2^(k / 2) * Γ(k / 2)) * x^((k / 2) - 1) * e^(-x / 2))
    /// ```
    fn ln_pdf(&self, x: f64) -> f64 {
        self.g.ln_pdf(x)
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::internal::*;
    use crate::testing_boiler;

    testing_boiler!(freedom: f64; ChiSquared; GammaError);

    #[test]
    fn test_median() {
        let median = |x: ChiSquared| x.median();
        test_absolute(0.5, 0.0857338820301783264746, 1e-16, median);
        test_exact(1.0, 1.0 - 2.0 / 3.0, median);
        test_exact(2.0, 2.0 - 2.0 / 3.0, median);
        test_exact(2.5, 2.5 - 2.0 / 3.0, median);
        test_exact(3.0, 3.0 - 2.0 / 3.0, median);
    }

    #[test]
    fn test_continuous() {
        // TODO: figure out why this test fails:
        //test::check_continuous_distribution(&create_ok(1.0), 0.0, 10.0);
        test::check_continuous_distribution(&create_ok(2.0), 0.0, 10.0);
        test::check_continuous_distribution(&create_ok(5.0), 0.0, 50.0);
    }
}
