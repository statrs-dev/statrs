use crate::distribution::{Continuous, ContinuousCDF};
use crate::function::erf;
use crate::statistics::*;
use crate::{consts, Result, StatsError};
use rand::Rng;
use std::f64;

/// Implements the
/// [Log-normal](https://en.wikipedia.org/wiki/Log-normal_distribution)
/// distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{LogNormal, Continuous};
/// use statrs::statistics::Distribution;
/// use statrs::prec;
///
/// let n = LogNormal::new(0.0, 1.0).unwrap();
/// assert_eq!(n.mean().unwrap(), (0.5f64).exp());
/// assert!(prec::almost_eq(n.pdf(1.0), 0.3989422804014326779399, 1e-16));
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct LogNormal {
    location: f64,
    scale: f64,
}

impl LogNormal {
    /// Constructs a new log-normal distribution with a location of `location`
    /// and a scale of `scale`
    ///
    /// # Errors
    ///
    /// Returns an error if `location` or `scale` are `NaN`.
    /// Returns an error if `scale <= 0.0`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::LogNormal;
    ///
    /// let mut result = LogNormal::new(0.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = LogNormal::new(0.0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(location: f64, scale: f64) -> Result<LogNormal> {
        if location.is_nan() || scale.is_nan() || scale <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(LogNormal { location, scale })
        }
    }
}

impl ::rand::distributions::Distribution<f64> for LogNormal {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        super::normal::sample_unchecked(rng, self.location, self.scale).exp()
    }
}

impl ContinuousCDF<f64, f64> for LogNormal {
    /// Calculates the cumulative distribution function for the log-normal
    /// distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / 2) + (1 / 2) * erf((ln(x) - μ) / sqrt(2) * σ)
    /// ```
    ///
    /// where `μ` is the location, `σ` is the scale, and `erf` is the
    /// error function
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else if x.is_infinite() {
            1.0
        } else {
            0.5 * erf::erfc((self.location - x.ln()) / (self.scale * f64::consts::SQRT_2))
        }
    }

    /// Calculates the survival function for the log-normal
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / 2) + (1 / 2) * erf(-(ln(x) - μ) / sqrt(2) * σ)
    /// ```
    ///
    /// where `μ` is the location, `σ` is the scale, and `erf` is the
    /// error function
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
        if x <= 0.0 {
            1.0
        } else if x.is_infinite() {
            0.0
        } else {
            0.5 * erf::erfc((x.ln() - self.location) / (self.scale * f64::consts::SQRT_2))
        }
    }
}

impl Min<f64> for LogNormal {
    /// Returns the minimum value in the domain of the log-normal
    /// distribution representable by a double precision float
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

impl Max<f64> for LogNormal {
    /// Returns the maximum value in the domain of the log-normal
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

impl Distribution<f64> for LogNormal {
    /// Returns the mean of the log-normal distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// e^(μ + σ^2 / 2)
    /// ```
    ///
    /// where `μ` is the location and `σ` is the scale
    fn mean(&self) -> Option<f64> {
        Some((self.location + self.scale * self.scale / 2.0).exp())
    }
    /// Returns the variance of the log-normal distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (e^(σ^2) - 1) * e^(2μ + σ^2)
    /// ```
    ///
    /// where `μ` is the location and `σ` is the scale
    fn variance(&self) -> Option<f64> {
        let sigma2 = self.scale * self.scale;
        Some((sigma2.exp() - 1.0) * (self.location + self.location + sigma2).exp())
    }
    /// Returns the entropy of the log-normal distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(σe^(μ + 1 / 2) * sqrt(2π))
    /// ```
    ///
    /// where `μ` is the location and `σ` is the scale
    fn entropy(&self) -> Option<f64> {
        Some(0.5 + self.scale.ln() + self.location + consts::LN_SQRT_2PI)
    }
    /// Returns the skewness of the log-normal distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (e^(σ^2) + 2) * sqrt(e^(σ^2) - 1)
    /// ```
    ///
    /// where `μ` is the location and `σ` is the scale
    fn skewness(&self) -> Option<f64> {
        let expsigma2 = (self.scale * self.scale).exp();
        Some((expsigma2 + 2.0) * (expsigma2 - 1.0).sqrt())
    }
}

impl Median<f64> for LogNormal {
    /// Returns the median of the log-normal distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// e^μ
    /// ```
    ///
    /// where `μ` is the location
    fn median(&self) -> f64 {
        self.location.exp()
    }
}

impl Mode<Option<f64>> for LogNormal {
    /// Returns the mode of the log-normal distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// e^(μ - σ^2)
    /// ```
    ///
    /// where `μ` is the location and `σ` is the scale
    fn mode(&self) -> Option<f64> {
        Some((self.location - self.scale * self.scale).exp())
    }
}

impl Continuous<f64, f64> for LogNormal {
    /// Calculates the probability density function for the log-normal
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / xσ * sqrt(2π)) * e^(-((ln(x) - μ)^2) / 2σ^2)
    /// ```
    ///
    /// where `μ` is the location and `σ` is the scale
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x.is_infinite() {
            0.0
        } else {
            let d = (x.ln() - self.location) / self.scale;
            (-0.5 * d * d).exp() / (x * consts::SQRT_2PI * self.scale)
        }
    }

    /// Calculates the log probability density function for the log-normal
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln((1 / xσ * sqrt(2π)) * e^(-((ln(x) - μ)^2) / 2σ^2))
    /// ```
    ///
    /// where `μ` is the location and `σ` is the scale
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x.is_infinite() {
            f64::NEG_INFINITY
        } else {
            let d = (x.ln() - self.location) / self.scale;
            (-0.5 * d * d) - consts::LN_SQRT_2PI - (x * self.scale).ln()
        }
    }
}

#[rustfmt::skip]
#[cfg(all(test, feature = "nightly"))]
mod tests {
    use crate::statistics::*;
    use crate::distribution::{ContinuousCDF, Continuous, LogNormal};
    use crate::distribution::internal::*;
    use crate::consts::ACC;

    fn try_create(mean: f64, std_dev: f64) -> LogNormal {
        let n = LogNormal::new(mean, std_dev);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn bad_create_case(mean: f64, std_dev: f64) {
        let n = LogNormal::new(mean, std_dev);
        assert!(n.is_err());
    }

    fn get_value<F>(mean: f64, std_dev: f64, eval: F) -> f64
        where F: Fn(LogNormal) -> f64
    {
        let n = try_create(mean, std_dev);
        eval(n)
    }

    fn test_case<F>(mean: f64, std_dev: f64, expected: f64, eval: F)
        where F: Fn(LogNormal) -> f64
    {
        let x = get_value(mean, std_dev, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(mean: f64, std_dev: f64, expected: f64, acc: f64, eval: F)
        where F: Fn(LogNormal) -> f64
    {
        let x = get_value(mean, std_dev, eval);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_create() {
        try_create(10.0, 0.1);
        try_create(-5.0, 1.0);
        try_create(0.0, 10.0);
        try_create(10.0, 100.0);
        try_create(-5.0, f64::INFINITY);
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
    fn test_mean() {
        let mean = |x: LogNormal| x.mean().unwrap();
        test_case(-1.0, 0.1, 0.369723444544058982601, mean);
        test_case(-1.0, 1.5, 1.133148453066826316829, mean);
        test_case(-1.0, 2.5, 8.372897488127264663205, mean);
        test_case(-1.0, 5.5, 1362729.18425285481771, mean);
        test_case(-0.1, 0.1, 0.9093729344682314204933, mean);
        test_case(-0.1, 1.5, 2.787095460565850768514, mean);
        test_case(-0.1, 2.5, 20.59400471119602917533, mean);
        test_almost(-0.1, 5.5, 3351772.941252693807591, 1e-9, mean);
        test_case(0.1, 0.1, 1.110710610355705232259, mean);
        test_case(0.1, 1.5, 3.40416608279081898632, mean);
        test_almost(0.1, 2.5, 25.15357415581836182776, 1e-14, mean);
        test_almost(0.1, 5.5, 4093864.715172665106863, 1e-8, mean);
        test_almost(1.5, 0.1, 4.50415363028848413209, 1e-15, mean);
        test_case(1.5, 1.5, 13.80457418606709491926, mean);
        test_case(1.5, 2.5, 102.0027730826996844534, mean);
        test_case(1.5, 5.5, 16601440.05723477471392, mean);
        test_almost(2.5, 0.1, 12.24355896580102707724, 1e-14, mean);
        test_almost(2.5, 1.5, 37.52472315960099891407, 1e-11, mean);
        test_case(2.5, 2.5, 277.2722845231339804081, mean);
        test_case(2.5, 5.5, 45127392.83383337999291, mean);
        test_almost(5.5, 0.1, 245.9184556788219446833, 1e-13, mean);
        test_case(5.5, 1.5, 753.7042125545612656606, mean);
        test_case(5.5, 2.5, 5569.162708566004074422, mean);
        test_case(5.5, 5.5, 906407915.0111549133446, mean);
    }

    #[test]
    fn test_variance() {
        let variance = |x: LogNormal| x.variance().unwrap();
        test_almost(-1.0, 0.1, 0.001373811865368952608715, 1e-16, variance);
        test_case(-1.0, 1.5, 10.898468544015731954, variance);
        test_case(-1.0, 2.5, 36245.39726189994988081, variance);
        test_almost(-1.0, 5.5, 2.5481629178024539E+25, 1e10, variance);
        test_almost(-0.1, 0.1, 0.008311077467909703803238, 1e-16, variance);
        test_case(-0.1, 1.5, 65.93189259328902509552, variance);
        test_almost(-0.1, 2.5, 219271.8756420929704707, 1e-10, variance);
        test_almost(-0.1, 5.5, 1.541548733459471E+26, 1e12, variance);
        test_almost(0.1, 0.1, 0.01239867063063756838894, 1e-15, variance);
        test_almost(0.1, 1.5, 98.35882573290010981464, 1e-13, variance);
        test_almost(0.1, 2.5, 327115.1995809995715014, 1e-10, variance);
        test_almost(0.1, 5.5, 2.299720473192458E+26, 1e12, variance);
        test_almost(1.5, 0.1, 0.2038917589520099120699, 1e-14, variance);
        test_almost(1.5, 1.5, 1617.476145997433210727, 1e-12, variance);
        test_almost(1.5, 2.5, 5379293.910566451644527, 1e-9, variance);
        test_almost(1.5, 5.5, 3.7818090853910142E+27, 1e12, variance);
        test_almost(2.5, 0.1, 1.506567645006046841936, 1e-13, variance);
        test_almost(2.5, 1.5, 11951.62198145717670088, 1e-11, variance);
        test_case(2.5, 2.5, 39747904.47781154725843, variance);
        test_almost(2.5, 5.5, 2.7943999487399818E+28, 1e13, variance);
        test_almost(5.5, 0.1, 607.7927673399807484235, 1e-11, variance);
        test_case(5.5, 1.5, 4821628.436260521100027, variance);
        test_case(5.5, 2.5, 16035449147.34799637823, variance);
        test_case(5.5, 5.5, 1.127341399856331737823E+31, variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: LogNormal| x.entropy().unwrap();
        test_case(-1.0, 0.1, -1.8836465597893728867265104870209210873020761202386, entropy);
        test_case(-1.0, 1.5, 0.82440364131283712375834285186996677643338789710028, entropy);
        test_case(-1.0, 2.5, 1.335229265078827806963856948173628711311498693546, entropy);
        test_case(-1.0, 5.5, 2.1236866254430979764250411929125703716076041932149, entropy);
        test_almost(-0.1, 0.1, -0.9836465597893728922776256101467037894202344606927, 1e-15, entropy);
        test_case(-0.1, 1.5, 1.7244036413128371182072277287441840743152295566462, entropy);
        test_case(-0.1, 2.5, 2.2352292650788278014127418250478460091933403530919, entropy);
        test_case(-0.1, 5.5, 3.0236866254430979708739260697867876694894458527608, entropy);
        test_almost(0.1, 0.1, -0.7836465597893728811753953638951383851839177797845, 1e-15, entropy);
        test_almost(0.1, 1.5, 1.9244036413128371293094579749957494785515462375544, 1e-15, entropy);
        test_case(0.1, 2.5, 2.4352292650788278125149720712994114134296570340001, entropy);
        test_case(0.1, 5.5, 3.223686625443097981976156316038353073725762533669, entropy);
        test_almost(1.5, 0.1, 0.6163534402106271132734895129790789126979238797614, 1e-15, entropy);
        test_case(1.5, 1.5, 3.3244036413128371237583428518699667764333878971003, entropy);
        test_case(1.5, 2.5, 3.835229265078827806963856948173628711311498693546, entropy);
        test_case(1.5, 5.5, 4.6236866254430979764250411929125703716076041932149, entropy);
        test_case(2.5, 0.1, 1.6163534402106271132734895129790789126979238797614, entropy);
        test_almost(2.5, 1.5, 4.3244036413128371237583428518699667764333878971003, 1e-15, entropy);
        test_case(2.5, 2.5, 4.835229265078827806963856948173628711311498693546, entropy);
        test_case(2.5, 5.5, 5.6236866254430979764250411929125703716076041932149, entropy);
        test_case(5.5, 0.1, 4.6163534402106271132734895129790789126979238797614, entropy);
        test_almost(5.5, 1.5, 7.3244036413128371237583428518699667764333878971003, 1e-15, entropy);
        test_case(5.5, 2.5, 7.835229265078827806963856948173628711311498693546, entropy);
        test_case(5.5, 5.5, 8.6236866254430979764250411929125703716076041932149, entropy);
    }

    #[test]
    fn test_skewness() {
        let skewness = |x: LogNormal| x.skewness().unwrap();
        test_almost(-1.0, 0.1, 0.30175909933883402945387113824982918009810212213629, 1e-14, skewness);
        test_case(-1.0, 1.5, 33.46804679732172529147579024311650645764144530123, skewness);
        test_almost(-1.0, 2.5, 11824.007933610287521341659465200553739278936344799, 1e-11, skewness);
        test_almost(-1.0, 5.5, 50829064464591483629.132631635472412625371367420496, 1e4, skewness);
        test_almost(-0.1, 0.1, 0.30175909933883402945387113824982918009810212213629, 1e-14, skewness);
        test_case(-0.1, 1.5, 33.46804679732172529147579024311650645764144530123, skewness);
        test_almost(-0.1, 2.5, 11824.007933610287521341659465200553739278936344799, 1e-11, skewness);
        test_almost(-0.1, 5.5, 50829064464591483629.132631635472412625371367420496, 1e4, skewness);
        test_almost(0.1, 0.1, 0.30175909933883402945387113824982918009810212213629, 1e-14, skewness);
        test_case(0.1, 1.5, 33.46804679732172529147579024311650645764144530123, skewness);
        test_almost(0.1, 2.5, 11824.007933610287521341659465200553739278936344799, 1e-11, skewness);
        test_almost(0.1, 5.5, 50829064464591483629.132631635472412625371367420496, 1e4, skewness);
        test_almost(1.5, 0.1, 0.30175909933883402945387113824982918009810212213629, 1e-14, skewness);
        test_case(1.5, 1.5, 33.46804679732172529147579024311650645764144530123, skewness);
        test_almost(1.5, 2.5, 11824.007933610287521341659465200553739278936344799, 1e-11, skewness);
        test_almost(1.5, 5.5, 50829064464591483629.132631635472412625371367420496, 1e4, skewness);
        test_almost(2.5, 0.1, 0.30175909933883402945387113824982918009810212213629, 1e-14, skewness);
        test_case(2.5, 1.5, 33.46804679732172529147579024311650645764144530123, skewness);
        test_almost(2.5, 2.5, 11824.007933610287521341659465200553739278936344799, 1e-11, skewness);
        test_almost(2.5, 5.5, 50829064464591483629.132631635472412625371367420496, 1e4, skewness);
        test_almost(5.5, 0.1, 0.30175909933883402945387113824982918009810212213629, 1e-14, skewness);
        test_case(5.5, 1.5, 33.46804679732172529147579024311650645764144530123, skewness);
        test_almost(5.5, 2.5, 11824.007933610287521341659465200553739278936344799, 1e-11, skewness);
        test_almost(5.5, 5.5, 50829064464591483629.132631635472412625371367420496, 1e4, skewness);
    }

    #[test]
    fn test_mode() {
        let mode = |x: LogNormal| x.mode().unwrap();
        test_case(-1.0, 0.1, 0.36421897957152331652213191863106773137983085909534, mode);
        test_case(-1.0, 1.5, 0.03877420783172200988689983526759614326014406193602, mode);
        test_case(-1.0, 2.5, 0.0007101743888425490635846003705775444086763023873619, mode);
        test_case(-1.0, 5.5, 0.000000000000026810038677818032221548731163905979029274677187036, mode);
        test_case(-0.1, 0.1, 0.89583413529652823774737070060865897390995185639633, mode);
        test_case(-0.1, 1.5, 0.095369162215549610417813418326627245539514227574881, mode);
        test_case(-0.1, 2.5, 0.0017467471362611196181003627521060283221112106850165, mode);
        test_case(-0.1, 5.5, 0.00000000000006594205454219929159167575814655534255162059017114, mode);
        test_case(0.1, 0.1, 1.0941742837052103542285651753780976842292770841345, mode);
        test_case(0.1, 1.5, 0.11648415777349696821514223131929465848700730137808, mode);
        test_case(0.1, 2.5, 0.0021334817700377079925027678518795817076296484352472, mode);
        test_case(0.1, 5.5, 0.000000000000080541807296590798973741710866097756565304960216803, mode);
        test_case(1.5, 0.1, 4.4370955190036645692996309927420381428715912422597, mode);
        test_case(1.5, 1.5, 0.47236655274101470713804655094326791297020357913648, mode);
        test_case(1.5, 2.5, 0.008651695203120634177071503957250390848166331197708, mode);
        test_case(1.5, 5.5, 0.00000000000032661313427874471360158184468030186601222739665225, mode);
        test_case(2.5, 0.1, 12.061276120444720299113038763305617245808510584994, mode);
        test_case(2.5, 1.5, 1.2840254166877414840734205680624364583362808652815, mode);
        test_case(2.5, 2.5, 0.023517745856009108236151185100432939470067655273072, mode);
        test_case(2.5, 5.5, 0.00000000000088782654784596584473099190326928541185172970391855, mode);
        test_case(5.5, 0.1, 242.2572068579541371904816252345031593584721473492, mode);
        test_case(5.5, 1.5, 25.790339917193062089080107669377221876655268848954, mode);
        test_case(5.5, 2.5, 0.47236655274101470713804655094326791297020357913648, mode);
        test_case(5.5, 5.5, 0.000000000017832472908146389493511850431527026413424899198327, mode);
    }

    #[test]
    fn test_median() {
        let median = |x: LogNormal| x.median();
        test_case(-1.0, 0.1, 0.36787944117144232159552377016146086744581113103177, median);
        test_case(-1.0, 1.5, 0.36787944117144232159552377016146086744581113103177, median);
        test_case(-1.0, 2.5, 0.36787944117144232159552377016146086744581113103177, median);
        test_case(-1.0, 5.5, 0.36787944117144232159552377016146086744581113103177, median);
        test_case(-0.1, 0.1, 0.90483741803595956814139238421693559530906465375738, median);
        test_case(-0.1, 1.5, 0.90483741803595956814139238421693559530906465375738, median);
        test_case(-0.1, 2.5, 0.90483741803595956814139238421693559530906465375738, median);
        test_case(-0.1, 5.5, 0.90483741803595956814139238421693559530906465375738, median);
        test_case(0.1, 0.1, 1.1051709180756476309466388234587796577416634163742, median);
        test_case(0.1, 1.5, 1.1051709180756476309466388234587796577416634163742, median);
        test_case(0.1, 2.5, 1.1051709180756476309466388234587796577416634163742, median);
        test_case(0.1, 5.5, 1.1051709180756476309466388234587796577416634163742, median);
        test_case(1.5, 0.1, 4.4816890703380648226020554601192758190057498683697, median);
        test_case(1.5, 1.5, 4.4816890703380648226020554601192758190057498683697, median);
        test_case(1.5, 2.5, 4.4816890703380648226020554601192758190057498683697, median);
        test_case(1.5, 5.5, 4.4816890703380648226020554601192758190057498683697, median);
        test_case(2.5, 0.1, 12.182493960703473438070175951167966183182767790063, median);
        test_case(2.5, 1.5, 12.182493960703473438070175951167966183182767790063, median);
        test_case(2.5, 2.5, 12.182493960703473438070175951167966183182767790063, median);
        test_case(2.5, 5.5, 12.182493960703473438070175951167966183182767790063, median);
        test_case(5.5, 0.1, 244.6919322642203879151889495118393501842287101075, median);
        test_case(5.5, 1.5, 244.6919322642203879151889495118393501842287101075, median);
        test_case(5.5, 2.5, 244.6919322642203879151889495118393501842287101075, median);
        test_case(5.5, 5.5, 244.6919322642203879151889495118393501842287101075, median);
    }

    #[test]
    fn test_min_max() {
        let min = |x: LogNormal| x.min();
        let max = |x: LogNormal| x.max();
        test_case(0.0, 0.1, 0.0, min);
        test_case(-3.0, 10.0, 0.0, min);
        test_case(0.0, 0.1, f64::INFINITY, max);
        test_case(-3.0, 10.0, f64::INFINITY, max);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: f64| move |x: LogNormal| x.pdf(arg);
        test_almost(-0.1, 0.1, 1.7968349035073582236359415565799753846986440127816e-104, 1e-118, pdf(0.1));
        test_almost(-0.1, 0.1, 0.00000018288923328441197822391757965928083462391836798722, 1e-21, pdf(0.5));
        test_case(-0.1, 0.1, 2.3363114904470413709866234247494393485647978367885, pdf(0.8));
        test_almost(-0.1, 1.5, 0.90492497850024368541682348133921492204585092983646, 1e-15, pdf(0.1));
        test_almost(-0.1, 1.5, 0.49191985207660942803818797602364034466489243416574, 1e-16, pdf(0.5));
        test_case(-0.1, 1.5, 0.33133347214343229148978298237579567194870525187207, pdf(0.8));
        test_case(-0.1, 2.5, 1.0824698632626565182080576574958317806389057196768, pdf(0.1));
        test_almost(-0.1, 2.5, 0.31029619474753883558901295436486123689563749784867, 1e-16, pdf(0.5));
        test_almost(-0.1, 2.5, 0.19922929916156673799861939824205622734205083805245, 1e-16, pdf(0.8));

// Test removed because it was causing compiler issues (see issue 31407 for rust)
// test_almost(1.5, 0.1, 4.1070141770545881694056265342787422035256248474059e-313, 1e-322, pdf(0.1));
//

        test_almost(1.5, 0.1, 2.8602688726477103843476657332784045661507239533567e-104, 1e-116, pdf(0.5));
        test_case(1.5, 0.1, 1.6670425710002183246335601541889400558525870482613e-64, pdf(0.8));
        test_almost(1.5, 1.5, 0.10698412103361841220076392503406214751353235895732, 1e-16, pdf(0.1));
        test_almost(1.5, 1.5, 0.18266125308224685664142384493330155315630876975024, 1e-16, pdf(0.5));
        test_almost(1.5, 1.5, 0.17185785323404088913982425377565512294017306418953, 1e-16, pdf(0.8));
        test_almost(1.5, 2.5, 0.50186885259059181992025035649158160252576845315332, 1e-15, pdf(0.1));
        test_almost(1.5, 2.5, 0.21721369314437986034957451699565540205404697589349, 1e-16, pdf(0.5));
        test_case(1.5, 2.5, 0.15729636000661278918949298391170443742675565300598, pdf(0.8));
        test_case(2.5, 0.1, 5.6836826548848916385760779034504046896805825555997e-500, pdf(0.1));
        test_almost(2.5, 0.1, 3.1225608678589488061206338085285607881363155340377e-221, 1e-233, pdf(0.5));
        test_almost(2.5, 0.1, 4.6994713794671660918554320071312374073172560048297e-161, 1e-173, pdf(0.8));
        test_almost(2.5, 1.5, 0.015806486291412916772431170442330946677601577502353, 1e-16, pdf(0.1));
        test_almost(2.5, 1.5, 0.055184331257528847223852028950484131834529030116388, 1e-16, pdf(0.5));
        test_case(2.5, 1.5, 0.063982134749859504449658286955049840393511776984362, pdf(0.8));
        test_almost(2.5, 2.5, 0.25212505662402617595900822552548977822542300480086, 1e-15, pdf(0.1));
        test_almost(2.5, 2.5, 0.14117186955911792460646517002386088579088567275401, 1e-16, pdf(0.5));
        test_almost(2.5, 2.5, 0.11021452580363707866161369621432656293405065561317, 1e-16, pdf(0.8));
    }

    #[test]
    fn test_neg_pdf() {
        let pdf = |arg: f64| move |x: LogNormal| x.pdf(arg);
        test_case(0.0, 1.0, 0.0, pdf(0.0));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: LogNormal| x.ln_pdf(arg);
        test_case(-0.1, 0.1, -238.88282294119596467794686179588610665317241097599, ln_pdf(0.1));
        test_almost(-0.1, 0.1, -15.514385149961296196003163062199569075052113039686, 1e-14, ln_pdf(0.5));
        test_case(-0.1, 0.1, 0.84857339958981283964373051826407417105725729082041, ln_pdf(0.8));
        test_almost(-0.1, 1.5, -0.099903235403144611051953094864849327288457482212211, 1e-15, ln_pdf(0.1));
        test_almost(-0.1, 1.5, -0.70943947804316122682964396008813828577195771418027, 1e-15, ln_pdf(0.5));
        test_almost(-0.1, 1.5, -1.1046299420497998262946038709903250420774183529995, 1e-15, ln_pdf(0.8));
        test_almost(-0.1, 2.5, 0.07924534056485078867266307735371665927517517183681, 1e-16, ln_pdf(0.1));
        test_case(-0.1, 2.5, -1.1702279707433794860424967893989374511050637417043, ln_pdf(0.5));
        test_case(-0.1, 2.5, -1.6132988605030400828957768752511536087538109996183, ln_pdf(0.8));
        test_case(1.5, 0.1, -719.29643782024317312262673764204041218720576249741, ln_pdf(0.1));
        test_almost(1.5, 0.1, -238.41793403955250272430898754048547661932857086122, 1e-13, ln_pdf(0.5));
        test_case(1.5, 0.1, -146.85439481068371057247137024006716189469284256628, ln_pdf(0.8));
        test_almost(1.5, 1.5, -2.2350748570877992856465076624973458117562108140674, 1e-15, ln_pdf(0.1));
        test_almost(1.5, 1.5, -1.7001219175524556705452882616787223585705662860012, 1e-15, ln_pdf(0.5));
        test_almost(1.5, 1.5, -1.7610875785399045023354101841009649273236721172008, 1e-15, ln_pdf(0.8));
        test_almost(1.5, 2.5, -0.68941644324162489418137656699398207513321602763104, 1e-15, ln_pdf(0.1));
        test_case(1.5, 2.5, -1.5268736489667254857801287379715477173125628275598, ln_pdf(0.5));
        test_case(1.5, 2.5, -1.8496236096394777662704671479709839674424623547308, ln_pdf(0.8));
        test_almost(2.5, 0.1, -1149.5549471196476523788026360929146688367845019398, 1e-12, ln_pdf(0.1));
        test_almost(2.5, 0.1, -507.73265209554698134113704985174959301922196605736, 1e-12, ln_pdf(0.5));
        test_almost(2.5, 0.1, -369.16874994210463740474549611573497379941224077335, 1e-13, ln_pdf(0.8));
        test_almost(2.5, 1.5, -4.1473348984184862316495477617980296904955324113457, 1e-15, ln_pdf(0.1));
        test_almost(2.5, 1.5, -2.8970762200235424747307247601045786110485663457169, 1e-15, ln_pdf(0.5));
        test_case(2.5, 1.5, -2.7491513791239977024488074547907467152956602019989, ln_pdf(0.8));
        test_almost(2.5, 2.5, -1.3778300581206721947424710027422282714793718026513, 1e-15, ln_pdf(0.1));
        test_case(2.5, 2.5, -1.9577771978563167352868858774048559682046428490575, ln_pdf(0.5));
        test_case(2.5, 2.5, -2.2053265778497513183112901654193054111123780652581, ln_pdf(0.8));
    }

    #[test]
    fn test_neg_ln_pdf() {
        let ln_pdf = |arg: f64| move |x: LogNormal| x.ln_pdf(arg);
        test_case(0.0, 1.0, f64::NEG_INFINITY, ln_pdf(0.0));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: f64| move |x: LogNormal| x.cdf(arg);
        test_almost(-0.1, 0.1, 0.0, 1e-107, cdf(0.1));
        test_almost(-0.1, 0.1, 0.0000000015011556178148777579869633555518882664666520593658, 1e-19, cdf(0.5));
        test_almost(-0.1, 0.1, 0.10908001076375810900224507908874442583171381706127, 1e-11, cdf(0.8));
        test_almost(-0.1, 1.5, 0.070999149762464508991968731574953594549291668468349, 1e-11, cdf(0.1));
        test_case(-0.1, 1.5, 0.34626224992888089297789445771047690175505847991946, cdf(0.5));
        test_case(-0.1, 1.5, 0.46728530589487698517090261668589508746353129242404, cdf(0.8));
        test_almost(-0.1, 2.5, 0.18914969879695093477606645992572208111152994999076, 1e-10, cdf(0.1));
        test_case(-0.1, 2.5, 0.40622798321378106125020505907901206714868922279347, cdf(0.5));
        test_case(-0.1, 2.5, 0.48035707589956665425068652807400957345208517749893, cdf(0.8));
        test_almost(1.5, 0.1, 0.0, 1e-315, cdf(0.1));
        test_almost(1.5, 0.1, 0.0, 1e-106, cdf(0.5));
        test_almost(1.5, 0.1, 0.0, 1e-66, cdf(0.8));
        test_almost(1.5, 1.5, 0.005621455876973168709588070988239748831823850202953, 1e-12, cdf(0.1));
        test_almost(1.5, 1.5, 0.07185716187918271235246980951571040808235628115265, 1e-11, cdf(0.5));
        test_almost(1.5, 1.5, 0.12532699044614938400496547188720940854423187977236, 1e-11, cdf(0.8));
        test_almost(1.5, 2.5, 0.064125647996943514411570834861724406903677144126117, 1e-11, cdf(0.1));
        test_almost(1.5, 2.5, 0.19017302281590810871719754032332631806011441356498, 1e-10, cdf(0.5));
        test_almost(1.5, 2.5, 0.24533064397555500690927047163085419096928289095201, 1e-16, cdf(0.8));
        test_case(2.5, 0.1, 0.0, cdf(0.1));
        test_almost(2.5, 0.1, 0.0, 1e-223, cdf(0.5));
        test_almost(2.5, 0.1, 0.0, 1e-162, cdf(0.8));
        test_almost(2.5, 1.5, 0.00068304052220788502001572635016579586444611070077399, 1e-13, cdf(0.1));
        test_almost(2.5, 1.5, 0.016636862816580533038130583128179878924863968664206, 1e-12, cdf(0.5));
        test_almost(2.5, 1.5, 0.034729001282904174941366974418836262996834852343018, 1e-11, cdf(0.8));
        test_almost(2.5, 2.5, 0.027363708266690978870139978537188410215717307180775, 1e-11, cdf(0.1));
        test_almost(2.5, 2.5, 0.10075543423327634536450625420610429181921642201567, 1e-11, cdf(0.5));
        test_almost(2.5, 2.5, 0.13802019192453118732001307556787218421918336849121, 1e-11, cdf(0.8));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: f64| move |x: LogNormal| x.sf(arg);

        // Wolfram Alpha:: SurvivalFunction[ LogNormalDistribution(-0.1, 0.1), 0.1]
        test_almost(-0.1, 0.1, 1.0, 1e-107, sf(0.1));

        // Wolfram Alpha:: SurvivalFunction[ LogNormalDistribution(-0.1, 0.1), 0.8]
        test_almost(-0.1, 0.1, 0.890919989231123, 1e-14, sf(0.8));

        // Wolfram Alpha:: SurvivalFunction[LogNormalDistribution[1.5, 1], 0.8]
        test_almost(1.5, 1.0, 0.957568715612642, 1e-14, sf(0.8));

        // Wolfram Alpha:: SurvivalFunction[ LogNormalDistribution(2.5, 1.5), 0.1]
        test_almost(2.5, 1.5, 0.9993169594777358, 1e-14, sf(0.1));
    }

    #[test]
    fn test_neg_cdf() {
        let cdf = |arg: f64| move |x: LogNormal| x.cdf(arg);
        test_case(0.0, 1.0, 0.0, cdf(0.0));
    }


    #[test]
    fn test_neg_sf() {
        let sf = |arg: f64| move |x: LogNormal| x.sf(arg);
        test_case(0.0, 1.0, 1.0, sf(0.0));
    }

    #[test]
    fn test_continuous() {
        test::check_continuous_distribution(&try_create(0.0, 0.25), 0.0, 10.0);
        test::check_continuous_distribution(&try_create(0.0, 0.5), 0.0, 10.0);
    }
}
