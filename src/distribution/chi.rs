use distribution::{Continuous, Univariate};
use function::gamma;
use rand::distributions::Distribution;
use rand::Rng;
use statistics::*;
use std::f64;
use {Result, StatsError};

/// Implements the [Chi](https://en.wikipedia.org/wiki/Chi_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Chi, Continuous};
/// use statrs::statistics::Mean;
/// use statrs::prec;
///
/// let n = Chi::new(2.0).unwrap();
/// assert!(prec::almost_eq(n.mean(), 1.25331413731550025121, 1e-14));
/// assert!(prec::almost_eq(n.pdf(1.0), 0.60653065971263342360, 1e-15));
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Chi {
    freedom: f64,
}

impl Chi {
    /// Constructs a new chi distribution
    /// with `freedom` degrees of freedom
    ///
    /// # Errors
    ///
    /// Returns an error if `freedom` is `NaN` or
    /// less than or equal to `0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Chi;
    ///
    /// let mut result = Chi::new(2.0);
    /// assert!(result.is_ok());
    ///
    /// result = Chi::new(0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(freedom: f64) -> Result<Chi> {
        if freedom.is_nan() || freedom <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Chi { freedom: freedom })
        }
    }

    /// Returns the degrees of freedom of
    /// the chi distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Chi;
    ///
    /// let n = Chi::new(2.0).unwrap();
    /// assert_eq!(n.freedom(), 2.0);
    /// ```
    pub fn freedom(&self) -> f64 {
        self.freedom
    }
}

impl Distribution<f64> for Chi {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> f64 {
        (0..self.freedom as i64)
            .fold(0.0, |acc, _| {
                acc + super::normal::sample_unchecked(r, 0.0, 1.0).powf(2.0)
            }).sqrt()
    }
}

impl Univariate<f64, f64> for Chi {
    /// Calculates the cumulative distribution function for the chi
    /// distribution at `x`.
    ///
    /// # Formula
    ///
    /// ```ignore
    /// P(k / 2, x^2 / 2)
    /// ```
    ///
    /// where `k` is the degrees of freedom and `P` is
    /// the regularized Gamma function
    fn cdf(&self, x: f64) -> f64 {
        if self.freedom == f64::INFINITY || x == f64::INFINITY {
            1.0
        } else if x <= 0.0 {
            0.0
        } else {
            gamma::gamma_lr(self.freedom / 2.0, x * x / 2.0)
        }
    }
}

impl Min<f64> for Chi {
    /// Returns the minimum value in the domain of the chi distribution
    /// representable by a double precision float
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

impl Max<f64> for Chi {
    /// Returns the maximum value in the domain of the chi distribution
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

impl Mean<f64> for Chi {
    /// Returns the mean of the chi distribution
    ///
    /// # Remarks
    ///
    /// Returns `NaN` if `freedom` is `INF`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt2 * Γ((k + 1) / 2) / Γ(k / 2)
    /// ```
    ///
    /// where `k` is degrees of freedom and `Γ` is the gamma function
    fn mean(&self) -> f64 {
        f64::consts::SQRT_2 * gamma::gamma((self.freedom + 1.0) / 2.0)
            / gamma::gamma(self.freedom / 2.0)
    }
}

impl Variance<f64> for Chi {
    /// Returns the variance of the chi distribution
    ///
    /// # Remarks
    ///
    /// Returns `NaN` if `freedom` is `INF`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// k - μ^2
    /// ```
    ///
    /// where `k` is degrees of freedom and `μ` is the mean
    /// of the distribution
    fn variance(&self) -> f64 {
        self.freedom - self.mean() * self.mean()
    }

    /// Returns the standard deviation of the chi distribution
    ///
    /// # Remarks
    ///
    /// Returns `NaN` if `freedom` is `INF`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(k - μ^2)
    /// ```
    ///
    /// where `k` is degrees of freedom and `μ` is the mean
    /// of the distribution
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

impl Entropy<f64> for Chi {
    /// Returns the entropy of the chi distribution
    ///
    /// # Remarks
    ///
    /// Returns `NaN` if `freedom` is `INF`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(Γ(k / 2)) + 0.5 * (k - ln2 - (k - 1) * ψ(k / 2))
    /// ```
    ///
    /// where `k` is degrees of freedom, `Γ` is the gamma function,
    /// and `ψ` is the digamma function
    fn entropy(&self) -> f64 {
        gamma::ln_gamma(self.freedom / 2.0)
            + (self.freedom
                - (2.0f64).ln()
                - (self.freedom - 1.0) * gamma::digamma(self.freedom / 2.0))
                / 2.0
    }
}

impl Skewness<f64> for Chi {
    /// Returns the skewness of the chi distribution
    ///
    /// # Remarks
    ///
    /// Returns `NaN` if `freedom` is `INF`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (μ / σ^3) * (1 - 2σ^2)
    /// ```
    /// where `μ` is the mean and `σ` the standard deviation
    /// of the distribution
    fn skewness(&self) -> f64 {
        let sigma = self.std_dev();
        self.mean() * (1.0 - 2.0 * sigma * sigma) / (sigma * sigma * sigma)
    }
}

impl Mode<f64> for Chi {
    /// Returns the mode for the chi distribution
    ///
    /// # Panics
    ///
    /// If `freedom < 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(k - 1)
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn mode(&self) -> f64 {
        self.checked_mode().unwrap()
    }
}

impl CheckedMode<f64> for Chi {
    /// Returns the mode for the chi distribution
    ///
    /// # Errors
    ///
    /// If `freedom < 1.0`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// sqrt(k - 1)
    /// ```
    ///
    /// where `k` is the degrees of freedom
    fn checked_mode(&self) -> Result<f64> {
        if self.freedom < 1.0 {
            Err(StatsError::ArgGte("freedom", 1.0))
        } else {
            Ok((self.freedom - 1.0).sqrt())
        }
    }
}

impl Continuous<f64, f64> for Chi {
    /// Calculates the probability density function for the chi
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (2^(1 - (k / 2)) * x^(k - 1) * e^(-x^2 / 2)) / Γ(k / 2)
    /// ```
    ///
    /// where `k` is the degrees of freedom and `Γ` is the gamma function
    fn pdf(&self, x: f64) -> f64 {
        if self.freedom == f64::INFINITY || x == f64::INFINITY || x <= 0.0 {
            0.0
        } else if self.freedom > 160.0 {
            self.ln_pdf(x).exp()
        } else {
            (2.0f64).powf(1.0 - self.freedom / 2.0)
                * x.powf(self.freedom - 1.0)
                * (-x * x / 2.0).exp()
                / gamma::gamma(self.freedom / 2.0)
        }
    }

    /// Calculates the log probability density function for the chi distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln((2^(1 - (k / 2)) * x^(k - 1) * e^(-x^2 / 2)) / Γ(k / 2))
    /// ```
    fn ln_pdf(&self, x: f64) -> f64 {
        if self.freedom == f64::INFINITY || x == f64::INFINITY || x <= 0.0 {
            f64::NEG_INFINITY
        } else {
            (1.0 - self.freedom / 2.0) * (2.0f64).ln() + ((self.freedom - 1.0) * x.ln())
                - x * x / 2.0
                - gamma::ln_gamma(self.freedom / 2.0)
        }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use statistics::*;
    use distribution::{Univariate, Continuous, Chi};
    use distribution::internal::*;

    fn try_create(freedom: f64) -> Chi {
        let n = Chi::new(freedom);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(freedom: f64) {
        let n = try_create(freedom);
        assert_eq!(freedom, n.freedom());
    }

    fn bad_create_case(freedom: f64) {
        let n = Chi::new(freedom);
        assert!(n.is_err());
    }

    fn get_value<F>(freedom: f64, eval: F) -> f64
        where F: Fn(Chi) -> f64
    {
        let n = try_create(freedom);
        eval(n)
    }

    fn test_case<F>(freedom: f64, expected: f64, eval: F)
        where F: Fn(Chi) -> f64
    {
        let x = get_value(freedom, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(freedom: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Chi) -> f64
    {
        let x = get_value(freedom, eval);
        assert_almost_eq!(expected, x, acc);
    }

    fn test_is_nan<F>(freedom: f64, eval: F)
        where F : Fn(Chi) -> f64
    {
        let x = get_value(freedom, eval);
        assert!(x.is_nan());
    }

    #[test]
    fn test_create() {
        create_case(1.0);
        create_case(3.0);
        create_case(f64::INFINITY);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(0.0);
        bad_create_case(-1.0);
        bad_create_case(-100.0);
        bad_create_case(f64::NEG_INFINITY);
        bad_create_case(f64::NAN);
    }

    #[test]
    fn test_mean() {
        test_almost(1.0, 0.7978845608028653558799, 1e-15, |x| x.mean());
        test_almost(2.0, 1.25331413731550025121, 1e-14, |x| x.mean());
        test_almost(2.5, 1.43396639245837498609, 1e-14, |x| x.mean());
        test_almost(5.0, 2.12769216214097428235, 1e-14, |x| x.mean());
        test_is_nan(f64::INFINITY, |x| x.mean());
    }

    #[test]
    fn test_variance() {
        test_almost(1.0, 0.3633802276324186569245, 1e-15, |x| x.variance());
        test_almost(2.0, 0.42920367320510338077, 1e-14, |x| x.variance());
        test_almost(2.5, 0.44374038529991368581, 1e-13, |x| x.variance());
        test_almost(3.0, 0.4535209105296746277, 1e-14, |x| x.variance());
        test_is_nan(f64::INFINITY, |x| x.variance());
    }

    #[test]
    fn test_std_dev() {
        test_almost(1.0, 0.6028102749890869742759, 1e-15, |x| x.std_dev());
        test_almost(2.0, 0.65513637756203355309, 1e-14, |x| x.std_dev());
        test_almost(2.5, 0.66613841301933165564, 1e-14, |x| x.std_dev());
        test_almost(3.0, 0.67343961164285148374, 1e-14, |x| x.std_dev());
        test_is_nan(f64::INFINITY, |x| x.std_dev());
    }

    #[test]
    fn test_entropy() {
        test_almost(1.0, 0.7257913526447274323631, 1e-15, |x| x.entropy());
        test_almost(2.0, 0.9420342421707937755946, 1e-15, |x| x.entropy());
        test_almost(2.5, 0.97574472333041323989, 1e-14, |x| x.entropy());
        test_almost(3.0, 0.99615419810620560239, 1e-14, |x| x.entropy());
        test_is_nan(f64::INFINITY, |x| x.entropy());
    }

    #[test]
    fn test_skewness() {
        test_almost(1.0, 0.995271746431156042444, 1e-14, |x| x.skewness());
        test_almost(2.0, 0.6311106578189371382, 1e-13, |x| x.skewness());
        test_almost(2.5, 0.5458487096285153216, 1e-12, |x| x.skewness());
        test_almost(3.0, 0.485692828049590809, 1e-12, |x| x.skewness());
        test_is_nan(f64::INFINITY, |x| x.skewness());
    }

    #[test]
    fn test_mode() {
        test_case(1.0, 0.0, |x| x.mode());
        test_case(2.0, 1.0, |x| x.mode());
        test_case(2.5, 1.224744871391589049099, |x| x.mode());
        test_case(3.0, f64::consts::SQRT_2, |x| x.mode());
        test_case(f64::INFINITY, f64::INFINITY, |x| x.mode());
    }

    #[test]
    #[should_panic]
    fn test_mode_freedom_lt_1() {
        get_value(0.5, |x| x.mode());
    }

    #[test]
    fn test_checked_mode_freedom_lt_1() {
        let n = try_create(0.5);
        assert!(n.checked_mode().is_err());
    }

    #[test]
    fn test_min_max() {
        test_case(1.0, 0.0, |x| x.min());
        test_case(2.0, 0.0, |x| x.min());
        test_case(2.5, 0.0, |x| x.min());
        test_case(3.0, 0.0, |x| x.min());
        test_case(f64::INFINITY, 0.0, |x| x.min());
        test_case(1.0, f64::INFINITY, |x| x.max());
        test_case(2.0, f64::INFINITY, |x| x.max());
        test_case(2.5, f64::INFINITY, |x| x.max());
        test_case(3.0, f64::INFINITY, |x| x.max());
        test_case(f64::INFINITY, f64::INFINITY, |x| x.max());
    }

    #[test]
    fn test_pdf() {
        test_case(1.0, 0.0, |x| x.pdf(0.0));
        test_almost(1.0, 0.79390509495402353102, 1e-15, |x| x.pdf(0.1));
        test_almost(1.0, 0.48394144903828669960, 1e-15, |x| x.pdf(1.0));
        test_almost(1.0, 2.1539520085086552718e-7, 1e-22, |x| x.pdf(5.5));
        test_case(1.0, 0.0, |x| x.pdf(f64::INFINITY));
        test_case(2.0, 0.0, |x| x.pdf(0.0));
        test_almost(2.0, 0.099501247919268231335, 1e-16, |x| x.pdf(0.1));
        test_almost(2.0, 0.60653065971263342360, 1e-15, |x| x.pdf(1.0));
        test_almost(2.0, 1.4847681768496578863e-6, 1e-21, |x| x.pdf(5.5));
        test_case(2.0, 0.0, |x| x.pdf(f64::INFINITY));
        test_case(2.5, 0.0, |x| x.pdf(0.0));
        test_almost(2.5, 0.029191065334961657461, 1e-16, |x| x.pdf(0.1));
        test_almost(2.5, 0.56269645152636456261, 1e-15, |x| x.pdf(1.0));
        test_almost(2.5, 3.2304380188895211768e-6, 1e-20, |x| x.pdf(5.5));
        test_case(2.5, 0.0, |x| x.pdf(f64::INFINITY));
        test_case(f64::INFINITY, 0.0, |x| x.pdf(0.0));
        test_case(f64::INFINITY, 0.0, |x| x.pdf(0.1));
        test_case(f64::INFINITY, 0.0, |x| x.pdf(1.0));
        test_case(f64::INFINITY, 0.0, |x| x.pdf(5.5));
        test_case(f64::INFINITY, 0.0, |x| x.pdf(f64::INFINITY));
        test_almost(170.0, 0.5644678498668440878, 1e-13, |x| x.pdf(13.0));
    }

    #[test]
    fn test_neg_pdf() {
        test_case(1.0, 0.0, |x| x.pdf(-1.0));
    }

    #[test]
    fn test_ln_pdf() {
        test_case(1.0, f64::NEG_INFINITY, |x| x.ln_pdf(0.0));
        test_almost(1.0, -0.23079135264472743236, 1e-15, |x| x.ln_pdf(0.1));
        test_almost(1.0, -0.72579135264472743236, 1e-15, |x| x.ln_pdf(1.0));
        test_almost(1.0, -15.350791352644727432, 1e-14, |x| x.ln_pdf(5.5));
        test_case(1.0, f64::NEG_INFINITY, |x| x.ln_pdf(f64::INFINITY));
        test_case(2.0, f64::NEG_INFINITY, |x| x.ln_pdf(0.0));
        test_almost(2.0, -2.3075850929940456840, 1e-15, |x| x.ln_pdf(0.1));
        test_almost(2.0, -0.5, 1e-15, |x| x.ln_pdf(1.0));
        test_almost(2.0, -13.420251907761574765, 1e-15, |x| x.ln_pdf(5.5));
        test_case(2.0, f64::NEG_INFINITY, |x| x.ln_pdf(f64::INFINITY));
        test_case(2.5, f64::NEG_INFINITY, |x| x.ln_pdf(0.0));
        test_almost(2.5, -3.5338925982092416919, 1e-15, |x| x.ln_pdf(0.1));
        test_almost(2.5, -0.57501495871817316589, 1e-15, |x| x.ln_pdf(1.0));
        test_almost(2.5, -12.642892820360535314, 1e-16, |x| x.ln_pdf(5.5));
        test_case(2.5, f64::NEG_INFINITY, |x| x.ln_pdf(f64::INFINITY));
        test_case(f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(0.0));
        test_case(f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(0.1));
        test_case(f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(1.0));
        test_case(f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(5.5));
        test_case(f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(f64::INFINITY));
        test_almost(170.0, -0.57187185030600516424237, 1e-13, |x| x.ln_pdf(13.0));
    }

    #[test]
    fn test_neg_ln_pdf() {
        test_case(1.0, f64::NEG_INFINITY, |x| x.ln_pdf(-1.0));
    }

    #[test]
    fn test_cdf() {
        test_case(1.0, 0.0, |x| x.cdf(0.0));
        test_almost(1.0, 0.079655674554057962931, 1e-16, |x| x.cdf(0.1));
        test_almost(1.0, 0.68268949213708589717, 1e-15, |x| x.cdf(1.0));
        test_case(1.0, 0.99999996202087506822, |x| x.cdf(5.5));
        test_case(1.0, 1.0, |x| x.cdf(f64::INFINITY));
// test_case(2.0, 0.0, |x| x.cdf(0.0));
// test_almost(2.0, 0.0049875208073176866474, 1e-17, |x| x.cdf(0.1));
// test_almost(2.0, 0.39346934028736657640, 1e-15, |x| x.cdf(1.0));
// test_case(2.0, 0.99999973004214966370, |x| x.cdf(5.5));
// test_case(2.0, 1.0, |x| x.cdf(f64::INFINITY));
// test_case(2.5, 0.0, |x| x.cdf(0.0));
// test_almost(2.5, 0.0011702413714030096290, 1e-18, |x| x.cdf(0.1));
// test_almost(2.5, 0.28378995266531297417, 1e-16, |x| x.cdf(1.0));
// test_case(2.5, 0.99999940337322804750, |x| x.cdf(5.5));
// test_case(2.5, 1.0, |x| x.cdf(f64::INFINITY));
// test_case(f64::INFINITY, 0.0, |x| x.cdf(0.0));
// test_case(f64::INFINITY, 0.0, |x| x.cdf(0.1));
// test_case(f64::INFINITY, 0.0, |x| x.cdf(1.0));
// test_case(f64::INFINITY, 0.0, |x| x.cdf(5.5));
// test_case(f64::INFINITY, 1.0, |x| x.cdf(f64::INFINITY));
    }

    #[test]
    fn test_neg_cdf() {
        test_case(1.0, 0.0, |x| x.cdf(-1.0));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&try_create(1.0), 0.0, 10.0);
        test::check_continuous_distribution(&try_create(2.0), 0.0, 10.0);
        test::check_continuous_distribution(&try_create(5.0), 0.0, 10.0);
    }
}
