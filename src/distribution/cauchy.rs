use distribution::{Continuous, Univariate};
use rand::distributions::Distribution;
use rand::Rng;
use statistics::*;
use std::f64;
use {Result, StatsError};

/// Implements the [Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution)
/// distribution, also known as the Lorentz distribution.
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Cauchy, Continuous};
/// use statrs::statistics::Mode;
///
/// let n = Cauchy::new(0.0, 1.0).unwrap();
/// assert_eq!(n.mode(), 0.0);
/// assert_eq!(n.pdf(1.0), 0.1591549430918953357689);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Cauchy {
    location: f64,
    scale: f64,
}

impl Cauchy {
    /// Constructs a new cauchy distribution with the given
    /// location and scale.
    ///
    /// # Errors
    ///
    /// Returns an error if location or scale are `NaN` or `scale <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Cauchy;
    ///
    /// let mut result = Cauchy::new(0.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = Cauchy::new(0.0, -1.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(location: f64, scale: f64) -> Result<Cauchy> {
        if location.is_nan() || scale.is_nan() || scale <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(Cauchy {
                location: location,
                scale: scale,
            })
        }
    }

    /// Returns the location of the cauchy distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Cauchy;
    ///
    /// let n = Cauchy::new(0.0, 1.0).unwrap();
    /// assert_eq!(n.location(), 0.0);
    /// ```
    pub fn location(&self) -> f64 {
        self.location
    }

    /// Returns the scale of the cauchy distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Cauchy;
    ///
    /// let n = Cauchy::new(0.0, 1.0).unwrap();
    /// assert_eq!(n.scale(), 1.0);
    /// ```
    pub fn scale(&self) -> f64 {
        self.scale
    }
}

impl Distribution<f64> for Cauchy {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> f64 {
        self.location + self.scale * (f64::consts::PI * (r.gen::<f64>() - 0.5)).tan()
    }
}

impl Univariate<f64, f64> for Cauchy {
    /// Calculates the cumulative distribution function for the
    /// cauchy distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / π) * arctan((x - x_0) / γ) + 0.5
    /// ```
    ///
    /// where `x_0` is the location and `γ` is the scale
    fn cdf(&self, x: f64) -> f64 {
        (1.0 / f64::consts::PI) * ((x - self.location) / self.scale).atan() + 0.5
    }
}

impl Min<f64> for Cauchy {
    /// Returns the minimum value in the domain of the cauchy
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

impl Max<f64> for Cauchy {
    /// Returns the maximum value in the domain of the cauchy
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

impl Entropy<f64> for Cauchy {
    /// Returns the entropy of the cauchy distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(γ) + ln(4π)
    /// ```
    ///
    /// where `γ` is the scale
    fn entropy(&self) -> f64 {
        (4.0 * f64::consts::PI * self.scale).ln()
    }
}

impl Median<f64> for Cauchy {
    /// Returns the median of the cauchy distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// x_0
    /// ```
    ///
    /// where `x_0` is the location
    fn median(&self) -> f64 {
        self.location
    }
}

impl Mode<f64> for Cauchy {
    /// Returns the mode of the cauchy distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// x_0
    /// ```
    ///
    /// where `x_0` is the location
    fn mode(&self) -> f64 {
        self.location
    }
}

impl Continuous<f64, f64> for Cauchy {
    /// Calculates the probability density function for the cauchy
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 / (πγ * (1 + ((x - x_0) / γ)^2))
    /// ```
    ///
    /// where `x_0` is the location and `γ` is the scale
    fn pdf(&self, x: f64) -> f64 {
        1.0 / (f64::consts::PI
            * self.scale
            * (1.0 + ((x - self.location) / self.scale) * ((x - self.location) / self.scale)))
    }

    /// Calculates the log probability density function for the cauchy
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(1 / (πγ * (1 + ((x - x_0) / γ)^2)))
    /// ```
    ///
    /// where `x_0` is the location and `γ` is the scale
    fn ln_pdf(&self, x: f64) -> f64 {
        -(f64::consts::PI
            * self.scale
            * (1.0 + ((x - self.location) / self.scale) * ((x - self.location) / self.scale)))
            .ln()
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;
    use statistics::*;
    use distribution::{Univariate, Continuous, Cauchy};
    use distribution::internal::*;

    fn try_create(location: f64, scale: f64) -> Cauchy {
        let n = Cauchy::new(location, scale);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(location: f64, scale: f64) {
        let n = try_create(location, scale);
        assert_eq!(location, n.location());
        assert_eq!(scale, n.scale());
    }

    fn bad_create_case(location: f64, scale: f64) {
        let n = Cauchy::new(location, scale);
        assert!(n.is_err());
    }

    fn test_case<F>(location: f64, scale: f64, expected: f64, eval: F)
        where F: Fn(Cauchy) -> f64
    {
        let n = try_create(location, scale);
        let x = eval(n);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(location: f64, scale: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(Cauchy) -> f64
    {
        let n = try_create(location, scale);
        let x = eval(n);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_create() {
        create_case(0.0, 0.1);
        create_case(0.0, 1.0);
        create_case(0.0, 10.0);
        create_case(10.0, 11.0);
        create_case(-5.0, 100.0);
        create_case(0.0, f64::INFINITY);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(f64::NAN, 1.0);
        bad_create_case(1.0, f64::NAN);
        bad_create_case(f64::NAN, f64::NAN);
        bad_create_case(1.0, 0.0);
    }

    #[test]
    fn test_entropy() {
        test_case(0.0, 2.0, 3.224171427529236102395, |x| x.entropy());
        test_case(0.1, 4.0, 3.917318608089181411812, |x| x.entropy());
        test_case(1.0, 10.0, 4.833609339963336476996, |x| x.entropy());
        test_case(10.0, 11.0, 4.92891951976766133704, |x| x.entropy());
    }

    #[test]
    fn test_mode() {
        test_case(0.0, 2.0, 0.0, |x| x.mode());
        test_case(0.1, 4.0, 0.1, |x| x.mode());
        test_case(1.0, 10.0, 1.0, |x| x.mode());
        test_case(10.0, 11.0, 10.0, |x| x.mode());
        test_case(0.0, f64::INFINITY, 0.0, |x| x.mode());
    }

    #[test]
    fn test_median() {
        test_case(0.0, 2.0, 0.0, |x| x.median());
        test_case(0.1, 4.0, 0.1, |x| x.median());
        test_case(1.0, 10.0, 1.0, |x| x.median());
        test_case(10.0, 11.0, 10.0, |x| x.median());
        test_case(0.0, f64::INFINITY, 0.0, |x| x.median());
    }

    #[test]
    fn test_min_max() {
        test_case(0.0, 1.0, f64::NEG_INFINITY, |x| x.min());
        test_case(0.0, 1.0, f64::INFINITY, |x| x.max());
    }

    #[test]
    fn test_pdf() {
        test_case(0.0, 0.1, 0.001272730452554141029739, |x| x.pdf(-5.0));
        test_case(0.0, 0.1, 0.03151583031522679916216, |x| x.pdf(-1.0));
        test_almost(0.0, 0.1, 3.183098861837906715378, 1e-14, |x| x.pdf(0.0));
        test_case(0.0, 0.1, 0.03151583031522679916216, |x| x.pdf(1.0));
        test_case(0.0, 0.1, 0.001272730452554141029739, |x| x.pdf(5.0));
        test_almost(0.0, 1.0, 0.01224268793014579505914, 1e-17, |x| x.pdf(-5.0));
        test_case(0.0, 1.0, 0.1591549430918953357689, |x| x.pdf(-1.0));
        test_case(0.0, 1.0, 0.3183098861837906715378, |x| x.pdf(0.0));
        test_case(0.0, 1.0, 0.1591549430918953357689, |x| x.pdf(1.0));
        test_almost(0.0, 1.0, 0.01224268793014579505914, 1e-17, |x| x.pdf(5.0));
        test_case(0.0, 10.0, 0.02546479089470325372302, |x| x.pdf(-5.0));
        test_case(0.0, 10.0, 0.03151583031522679916216, |x| x.pdf(-1.0));
        test_case(0.0, 10.0, 0.03183098861837906715378, |x| x.pdf(0.0));
        test_case(0.0, 10.0, 0.03151583031522679916216, |x| x.pdf(1.0));
        test_case(0.0, 10.0, 0.02546479089470325372302, |x| x.pdf(5.0));
        test_case(-5.0, 100.0, 0.003183098861837906715378, |x| x.pdf(-5.0));
        test_almost(-5.0, 100.0, 0.003178014039374906864395, 1e-17, |x| x.pdf(-1.0));
        test_case(-5.0, 100.0, 0.003175160959439308444267, |x| x.pdf(0.0));
        test_case(-5.0, 100.0, 0.003171680810918599756255, |x| x.pdf(1.0));
        test_almost(-5.0, 100.0, 0.003151583031522679916216, 1e-17, |x| x.pdf(5.0));
        test_case(0.0, f64::INFINITY, 0.0, |x| x.pdf(-5.0));
        test_case(0.0, f64::INFINITY, 0.0, |x| x.pdf(-1.0));
        test_case(0.0, f64::INFINITY, 0.0, |x| x.pdf(0.0));
        test_case(0.0, f64::INFINITY, 0.0, |x| x.pdf(1.0));
        test_case(0.0, f64::INFINITY, 0.0, |x| x.pdf(5.0));
        test_case(f64::INFINITY, 1.0, 0.0, |x| x.pdf(-5.0));
        test_case(f64::INFINITY, 1.0, 0.0, |x| x.pdf(-1.0));
        test_case(f64::INFINITY, 1.0, 0.0, |x| x.pdf(0.0));
        test_case(f64::INFINITY, 1.0, 0.0, |x| x.pdf(1.0));
        test_case(f64::INFINITY, 1.0, 0.0, |x| x.pdf(5.0));
    }

    #[test]
    fn test_ln_pdf() {
        test_case(0.0, 0.1, -6.666590723732973542744, |x| x.ln_pdf(-5.0));
        test_almost(0.0, 0.1, -3.457265309696613941009, 1e-14, |x| x.ln_pdf(-1.0));
        test_case(0.0, 0.1, 1.157855207144645509875, |x| x.ln_pdf(0.0));
        test_almost(0.0, 0.1, -3.457265309696613941009, 1e-14, |x| x.ln_pdf(1.0));
        test_case(0.0, 0.1, -6.666590723732973542744, |x| x.ln_pdf(5.0));
        test_case(0.0, 1.0, -4.402826423870882219615, |x| x.ln_pdf(-5.0));
        test_almost(0.0, 1.0, -1.837877066409345483561, 1e-15, |x| x.ln_pdf(-1.0));
        test_case(0.0, 1.0, -1.144729885849400174143, |x| x.ln_pdf(0.0));
        test_almost(0.0, 1.0, -1.837877066409345483561, 1e-15, |x| x.ln_pdf(1.0));
        test_case(0.0, 1.0, -4.402826423870882219615, |x| x.ln_pdf(5.0));
        test_case(0.0, 10.0, -3.670458530157655613928, |x| x.ln_pdf(-5.0));
        test_almost(0.0, 10.0, -3.457265309696613941009, 1e-14, |x| x.ln_pdf(-1.0));
        test_case(0.0, 10.0, -3.447314978843445858161, |x| x.ln_pdf(0.0));
        test_almost(0.0, 10.0, -3.457265309696613941009, 1e-14, |x| x.ln_pdf(1.0));
        test_case(0.0, 10.0, -3.670458530157655613928, |x| x.ln_pdf(5.0));
        test_case(-5.0, 100.0, -5.749900071837491542179, |x| x.ln_pdf(-5.0));
        test_case(-5.0, 100.0, -5.751498793201188569872, |x| x.ln_pdf(-1.0));
        test_case(-5.0, 100.0, -5.75239695203607874116, |x| x.ln_pdf(0.0));
        test_case(-5.0, 100.0, -5.75349360734762171285, |x| x.ln_pdf(1.0));
        test_case(-5.0, 100.0, -5.759850402690659625027, |x| x.ln_pdf(5.0));
        test_case(0.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(-5.0));
        test_case(0.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(-1.0));
        test_case(0.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(0.0));
        test_case(0.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(1.0));
        test_case(0.0, f64::INFINITY, f64::NEG_INFINITY, |x| x.ln_pdf(5.0));
        test_case(f64::INFINITY, 1.0, f64::NEG_INFINITY, |x| x.ln_pdf(-5.0));
        test_case(f64::INFINITY, 1.0, f64::NEG_INFINITY, |x| x.ln_pdf(-1.0));
        test_case(f64::INFINITY, 1.0, f64::NEG_INFINITY, |x| x.ln_pdf(0.0));
        test_case(f64::INFINITY, 1.0, f64::NEG_INFINITY, |x| x.ln_pdf(1.0));
        test_case(f64::INFINITY, 1.0, f64::NEG_INFINITY, |x| x.ln_pdf(5.0));
    }

    #[test]
    fn test_cdf() {
        test_almost(0.0, 0.1, 0.006365349100972796679298, 1e-16, |x| x.cdf(-5.0));
        test_almost(0.0, 0.1, 0.03172551743055356951498, 1e-16, |x| x.cdf(-1.0));
        test_case(0.0, 0.1, 0.5, |x| x.cdf(0.0));
        test_case(0.0, 0.1, 0.968274482569446430485, |x| x.cdf(1.0));
        test_case(0.0, 0.1, 0.9936346508990272033207, |x| x.cdf(5.0));
        test_almost(0.0, 1.0, 0.06283295818900118381375, 1e-16, |x| x.cdf(-5.0));
        test_case(0.0, 1.0, 0.25, |x| x.cdf(-1.0));
        test_case(0.0, 1.0, 0.5, |x| x.cdf(0.0));
        test_case(0.0, 1.0, 0.75, |x| x.cdf(1.0));
        test_case(0.0, 1.0, 0.9371670418109988161863, |x| x.cdf(5.0));
        test_case(0.0, 10.0, 0.3524163823495667258246, |x| x.cdf(-5.0));
        test_case(0.0, 10.0, 0.468274482569446430485, |x| x.cdf(-1.0));
        test_case(0.0, 10.0, 0.5, |x| x.cdf(0.0));
        test_case(0.0, 10.0, 0.531725517430553569515, |x| x.cdf(1.0));
        test_case(0.0, 10.0, 0.6475836176504332741754, |x| x.cdf(5.0));
        test_case(-5.0, 100.0, 0.5, |x| x.cdf(-5.0));
        test_case(-5.0, 100.0, 0.5127256113479918307809, |x| x.cdf(-1.0));
        test_case(-5.0, 100.0, 0.5159022512561763751816, |x| x.cdf(0.0));
        test_case(-5.0, 100.0, 0.5190757242358362337495, |x| x.cdf(1.0));
        test_case(-5.0, 100.0, 0.531725517430553569515, |x| x.cdf(5.0));
        test_case(0.0, f64::INFINITY, 0.5, |x| x.cdf(-5.0));
        test_case(0.0, f64::INFINITY, 0.5, |x| x.cdf(-1.0));
        test_case(0.0, f64::INFINITY, 0.5, |x| x.cdf(0.0));
        test_case(0.0, f64::INFINITY, 0.5, |x| x.cdf(1.0));
        test_case(0.0, f64::INFINITY, 0.5, |x| x.cdf(5.0));
        test_case(f64::INFINITY, 1.0, 0.0, |x| x.cdf(-5.0));
        test_case(f64::INFINITY, 1.0, 0.0, |x| x.cdf(-1.0));
        test_case(f64::INFINITY, 1.0, 0.0, |x| x.cdf(0.0));
        test_case(f64::INFINITY, 1.0, 0.0, |x| x.cdf(1.0));
        test_case(f64::INFINITY, 1.0, 0.0, |x| x.cdf(5.0));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&try_create(-1.2, 3.4), -1500.0, 1500.0);
        test::check_continuous_distribution(&try_create(-4.5, 6.7), -5000.0, 5000.0);
    }
}
