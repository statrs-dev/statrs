use crate::distribution::{Continuous, ContinuousMultivariateCDF};
use crate::statistics::*;
use crate::{Result, StatsError};
use nalgebra::DVector;
use rand::distributions::Uniform as RandUniform;
use rand::Rng;
use std::f64;

/// Implements a Continuous Multivariate Uniform distribution
///
/// # Examples
///
/// ```
/// use statrs::distribution::{MultivariateUniform, Continuous};
/// use statrs::statistics::Distribution;
///
/// let n = MultivariateUniform::new(vec![-1., 0.], vec![0., 1.]).unwrap();
// /// assert_eq!(n.mean().unwrap(), 0.5);
// /// assert_eq!(n.pdf(0.5), 1.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateUniform {
    dim: usize,
    min: DVector<f64>,
    max: DVector<f64>,
    sample_limits: Option<DVector<RandUniform<f64>>>,
}

impl MultivariateUniform {
    /// Constructs a new uniform distribution with a min in each dimension 
    /// of `min` and a max in each dimension of `max`
    ///
    /// # Errors
    ///
    /// Returns an error if any elements of `min` or `max` are `NaN`
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::MultivariateUniform;
    /// use std::f64;
    ///
    /// let mut result = Uniform::new(vec![-1., 0.], vec![0., 1.,]);
    /// assert!(result.is_ok());
    ///
    /// result = Uniform::new(f64::NAN, f64::NAN);
    /// result = Uniform::new(vec![0., f64::NAN], vec![f64::NAN, 1.]);
    /// assert!(result.is_err());
    /// ```
    pub fn new(min: Vec<f64>, max: Vec<f64>) -> Result<MultivariateUniform> {
        let min = DVector::from_vec(min);
        let max = DVector::from_vec(max);
        if min.iter().any(|f| f.is_nan())
            || max.iter().any(|f| f.is_nan()) 
            || min.len() != max.len()
            || min.iter().zip(max.iter()).any(|(f,g)| f > g)
        {
            Err(StatsError::BadParams)
        } else {
            let dim = min.len();
            let mut sample_limits_unchecked: Vec<Option<RandUniform<f64>>> = vec![None; min.len()];
            // If we have infinite values as min or max we can't sample
            let sample_limits: Option<DVector<RandUniform<f64>>>;
            if min.iter().any(|f| f == &f64::NEG_INFINITY)
                || max.iter().any(|f| f == &f64::INFINITY)
            {
                sample_limits = None;
            } else {
                for i in 0..dim {
                    sample_limits_unchecked[i] = Some(RandUniform::new_inclusive(min[i],max[i]))
                }
                sample_limits = Some(DVector::from_vec(
                    sample_limits_unchecked.into_iter().flatten().collect::<Vec<RandUniform<f64>>>()
                ));
            }
            Ok(MultivariateUniform { dim, min, max, sample_limits})
        }
    }
}

/// Returns an Option of sampled point in `self.dim` dimensions if the boundaries
/// are not positive or negative infinite, else return None
impl ::rand::distributions::Distribution<Option<DVector<f64>>> for MultivariateUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<DVector<f64>> {
        match &self.sample_limits {
            Some(sample_limits) => {
                let mut samples: DVector<f64> = DVector::zeros(self.dim);
                for i in 0..self.dim {
                    samples[i] = rng.sample(sample_limits[i]);
                }
                Some(samples)
            },
            None => None
        }
    }
}

impl ContinuousMultivariateCDF<f64, f64> for MultivariateUniform {
    /// Calculates the cumulative distribution function for the uniform
    /// distribution
    /// at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (x - min) / (max - min)
    /// ```
    fn cdf(&self, x: DVector<f64>) -> f64 {
        if x <= self.min {
            0.
        } else if x >= self.max {
            1.
        } else {
            (x - &self.min).iter().product::<f64>() / self.min.iter().zip(self.max.iter()).map(|(f,g)| g - f).product::<f64>()
        }
    }

    /// Calculates the survival function for the uniform
    /// distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (max - x) / (max - min)
    /// ```
    fn sf(&self, x: DVector<f64>) -> f64 {
        if x <= self.min {
            1.0
        } else if x >= self.max {
            0.0
        } else {
            (&self.max - x).iter().product::<f64>() / self.min.iter().zip(self.max.iter()).map(|(f,g)| g - f).product::<f64>()
        }
    }
}

impl Min<DVector<f64>> for MultivariateUniform {
    fn min(&self) -> DVector<f64> {
        self.min.clone()
    }
}

impl Max<DVector<f64>> for MultivariateUniform {
    fn max(&self) -> DVector<f64> {
        self.max.clone()
    }
}

impl MeanN<DVector<f64>> for MultivariateUniform {
    /// Returns the mean of the multivariate uniform distribution
    fn mean(&self) -> Option<DVector<f64>> {
        Some((&self.min + &self.max) / 2.)
    }
}

// impl VarianceN<DMatrix<f64>> for MultivariateUniform {
//     /// Returns the covariance matrix of the multivariate uniform distribution
//     fn variance(&self) -> Option<DMatrix<f64>> {
//         Some(...)
//     }
// }

impl Median<DVector<f64>> for MultivariateUniform {
    /// Returns the median for the continuous uniform distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (min + max) / 2
    /// ```
    fn median(&self) -> DVector<f64> {
        (&self.min + &self.max) / 2.0
    }
}

impl Mode<Option<DVector<f64>>> for MultivariateUniform {
    /// Returns the mode for the continuous uniform distribution
    ///
    /// # Remarks
    ///
    /// Since every element has an equal probability, mode simply
    /// returns the middle element
    ///
    /// # Formula
    ///
    /// ```ignore
    /// N/A // (max + min) / 2 for the middle element
    /// ```
    fn mode(&self) -> Option<DVector<f64>> {
        Some((&self.min + &self.max) / 2.0)
    }
}

impl<'a> Continuous<&'a DVector<f64>, f64> for MultivariateUniform {
    /// Calculates the probability density function for the continuous uniform
    /// distribution at `x`
    ///
    /// # Remarks
    ///
    /// Returns `0.0` if `x` is not in `[min, max]`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// 1 / ∏(max - min)
    /// ```
    fn pdf(&self, x: &'a DVector<f64>) -> f64 {
        if x < &self.min || x > &self.max {
            0.0
        } else {
            1. / self.min.iter().zip(self.max.iter()).map(|(f,g)| g - f).product::<f64>()
        }
    }

    /// Calculates the log probability density function for the continuous
    /// uniform
    /// distribution at `x`
    ///
    /// # Remarks
    ///
    /// Returns `f64::NEG_INFINITY` if `x` is not in `[min, max]`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln(1 / ∏(max - min)) = -ln(∑(max -min))
    /// ```
    fn ln_pdf(&self, x: &'a DVector<f64>) -> f64 {
        if x < &self.min || x > &self.max {
            f64::NEG_INFINITY
        } else {
            -self.min.iter().zip(self.max.iter()).map(|(f,g)| g - f).sum::<f64>().ln()
        }
    }
}

#[rustfmt::skip]
// #[cfg(all(test, feature = "nightly"))]
mod tests {
    use crate::statistics::*;
    use crate::distribution::{ContinuousCDF, Continuous, MultivariateUniform};
    use crate::distribution::internal::*;
    use crate::consts::ACC;
    use nalgebra::DVector;

    fn try_create(min: Vec<f64>, max: Vec<f64>) -> MultivariateUniform {
        let n = MultivariateUniform::new(min, max);
        assert!(n.is_ok());
        n.unwrap()
    }

    fn create_case(min: Vec<f64>, max: Vec<f64>) {
        let n = try_create(min.clone(), max.clone());
        assert_eq!(n.min(), DVector::from_vec(min));
        assert_eq!(n.max(), DVector::from_vec(max));
    }

    fn bad_create_case(min: Vec<f64>, max: Vec<f64>) {
        let n = MultivariateUniform::new(min, max);
        assert!(n.is_err());
    }

    fn get_value<F>(min: Vec<f64>, max: Vec<f64>, eval: F) -> DVector<f64>
        where F: Fn(MultivariateUniform) -> DVector<f64>
    {
        let n = try_create(min, max);
        eval(n)
    }

    fn test_case<F>(min: Vec<f64>, max: Vec<f64>, expected: DVector<f64>, eval: F)
        where F: Fn(MultivariateUniform) -> DVector<f64>
    {
        let x = get_value(min, max, eval);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(min: Vec<f64>, max: Vec<f64>, expected: DVector<f64>, acc: f64, eval: F)
        where F: Fn(MultivariateUniform) -> DVector<f64>
    {

        let x = get_value(min, max, eval);
        // assert_almost_eq!(expected, x, acc);
    }

    macro_rules! dvec {
        ($($x:expr),*) => (DVector::from_vec(vec![$($x),*]));
    }

    #[test]
    fn test_create() {
        create_case(vec![0., 2.], vec![1., 3.,]);
        create_case(vec![0.1, 0.], vec![0.2, 0.1,]);
        create_case(vec![-5., 5., 10.], vec![-4., 6., 11.,]);
        create_case(vec![0.], vec![1.]);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(vec![f64::NAN, 5.], vec![0., 6.]);
        bad_create_case(vec![1., 1.,], vec![0., 0.]);
        bad_create_case(vec![0., 0.], vec![-1., 1.]);
        bad_create_case(vec![0., 0.,], vec![f64::NAN, 3.]);
        bad_create_case(vec![0.], vec![1., 2.]);
    }

    // #[test]
    // fn test_variance() {
    //     let variance = |x: MultivariateUniform| x.variance().unwrap();
    //     test_case(-0.0, 2.0, 1.0 / 3.0, variance);
    //     test_case(0.0, 2.0, 1.0 / 3.0, variance);
    //     test_almost(0.1, 4.0, 1.2675, 1e-15, variance);
    //     test_case(10.0, 11.0, 1.0 / 12.0, variance);
    //     test_case(0.0, f64::INFINITY, f64::INFINITY, variance);
    // }

    // #[test]
    // fn test_entropy() {
    //     let entropy = |x: MultivariateUniform| x.entropy().unwrap();
    //     test_case(-0.0, 2.0, 0.6931471805599453094172, entropy);
    //     test_case(0.0, 2.0, 0.6931471805599453094172, entropy);
    //     test_almost(0.1, 4.0, 1.360976553135600743431, 1e-15, entropy);
    //     test_case(1.0, 10.0, 2.19722457733621938279, entropy);
    //     test_case(10.0, 11.0, 0.0, entropy);
    //     test_case(0.0, f64::INFINITY, f64::INFINITY, entropy);
    // }

    // #[test]
    // fn test_skewness() {
    //     let skewness = |x: MultivariateUniform| x.skewness().unwrap();
    //     test_case(-0.0, 2.0, 0.0, skewness);
    //     test_case(0.0, 2.0, 0.0, skewness);
    //     test_case(0.1, 4.0, 0.0, skewness);
    //     test_case(1.0, 10.0, 0.0, skewness);
    //     test_case(10.0, 11.0, 0.0, skewness);
    //     test_case(0.0, f64::INFINITY, 0.0, skewness);
    // }

    #[test]
    fn test_mode() {
        let mode = |x: MultivariateUniform| x.mode().unwrap();
        test_case(vec![0., 0.,], vec![1., 1.,], dvec![0.5, 0.5], mode);
        test_case(vec![-1., 0.], vec![1., 2.], dvec![0., 1.], mode);
        test_case(vec![0., 0., f64::NEG_INFINITY], vec![1., 1., 0.], dvec![0.5, 0.5, f64::NEG_INFINITY], mode);
        test_case(vec![0.], vec![f64::INFINITY], dvec![f64::INFINITY], mode);
    }

    // #[test]
    // fn test_median() {
    //     let median = |x: MultivariateUniform| x.median();
    //     test_case(-0.0, 2.0, 1.0, median);
    //     test_case(0.0, 2.0, 1.0, median);
    //     test_case(0.1, 4.0, 2.05, median);
    //     test_case(1.0, 10.0, 5.5, median);
    //     test_case(10.0, 11.0, 10.5, median);
    //     test_case(0.0, f64::INFINITY, f64::INFINITY, median);
    // }

    // #[test]
    // fn test_pdf() {
    //     let pdf = |arg: f64| move |x: MultivariateUniform| x.pdf(arg);
    //     test_case(0.0, 0.0, 0.0, pdf(-5.0));
    //     test_case(0.0, 0.0, f64::INFINITY, pdf(0.0));
    //     test_case(0.0, 0.0, 0.0, pdf(5.0));
    //     test_case(0.0, 0.1, 0.0, pdf(-5.0));
    //     test_case(0.0, 0.1, 10.0, pdf(0.05));
    //     test_case(0.0, 0.1, 0.0, pdf(5.0));
    //     test_case(0.0, 1.0, 0.0, pdf(-5.0));
    //     test_case(0.0, 1.0, 1.0, pdf(0.5));
    //     test_case(0.0, 0.1, 0.0, pdf(5.0));
    //     test_case(0.0, 10.0, 0.0, pdf(-5.0));
    //     test_case(0.0, 10.0, 0.1, pdf(1.0));
    //     test_case(0.0, 10.0, 0.1, pdf(5.0));
    //     test_case(0.0, 10.0, 0.0, pdf(11.0));
    //     test_case(-5.0, 100.0, 0.0, pdf(-10.0));
    //     test_case(-5.0, 100.0, 0.009523809523809523809524, pdf(-5.0));
    //     test_case(-5.0, 100.0, 0.009523809523809523809524, pdf(0.0));
    //     test_case(-5.0, 100.0, 0.0, pdf(101.0));
    //     test_case(0.0, f64::INFINITY, 0.0, pdf(-5.0));
    //     test_case(0.0, f64::INFINITY, 0.0, pdf(10.0));
    //     test_case(0.0, f64::INFINITY, 0.0, pdf(f64::INFINITY));
    // }

    // #[test]
    // fn test_ln_pdf() {
    //     let ln_pdf = |arg: f64| move |x: MultivariateUniform| x.ln_pdf(arg);
    //     test_case(0.0, 0.0, f64::NEG_INFINITY, ln_pdf(-5.0));
    //     test_case(0.0, 0.0, f64::INFINITY, ln_pdf(0.0));
    //     test_case(0.0, 0.0, f64::NEG_INFINITY, ln_pdf(5.0));
    //     test_case(0.0, 0.1, f64::NEG_INFINITY, ln_pdf(-5.0));
    //     test_almost(0.0, 0.1, 2.302585092994045684018, 1e-15, ln_pdf(0.05));
    //     test_case(0.0, 0.1, f64::NEG_INFINITY, ln_pdf(5.0));
    //     test_case(0.0, 1.0, f64::NEG_INFINITY, ln_pdf(-5.0));
    //     test_case(0.0, 1.0, 0.0, ln_pdf(0.5));
    //     test_case(0.0, 0.1, f64::NEG_INFINITY, ln_pdf(5.0));
    //     test_case(0.0, 10.0, f64::NEG_INFINITY, ln_pdf(-5.0));
    //     test_case(0.0, 10.0, -2.302585092994045684018, ln_pdf(1.0));
    //     test_case(0.0, 10.0, -2.302585092994045684018, ln_pdf(5.0));
    //     test_case(0.0, 10.0, f64::NEG_INFINITY, ln_pdf(11.0));
    //     test_case(-5.0, 100.0, f64::NEG_INFINITY, ln_pdf(-10.0));
    //     test_case(-5.0, 100.0, -4.653960350157523371101, ln_pdf(-5.0));
    //     test_case(-5.0, 100.0, -4.653960350157523371101, ln_pdf(0.0));
    //     test_case(-5.0, 100.0, f64::NEG_INFINITY, ln_pdf(101.0));
    //     test_case(0.0, f64::INFINITY, f64::NEG_INFINITY, ln_pdf(-5.0));
    //     test_case(0.0, f64::INFINITY, f64::NEG_INFINITY, ln_pdf(10.0));
    //     test_case(0.0, f64::INFINITY, f64::NEG_INFINITY, ln_pdf(f64::INFINITY));
    // }

    // #[test]
    // fn test_cdf() {
    //     let cdf = |arg: f64| move |x: MultivariateUniform| x.cdf(arg);
    //     test_case(0.0, 0.0, 0.0, cdf(0.0));
    //     test_case(0.0, 0.1, 0.5, cdf(0.05));
    //     test_case(0.0, 1.0, 0.5, cdf(0.5));
    //     test_case(0.0, 10.0, 0.1, cdf(1.0));
    //     test_case(0.0, 10.0, 0.5, cdf(5.0));
    //     test_case(-5.0, 100.0, 0.0, cdf(-5.0));
    //     test_case(-5.0, 100.0, 0.04761904761904761904762, cdf(0.0));
    //     test_case(0.0, f64::INFINITY, 0.0, cdf(10.0));
    //     test_case(0.0, f64::INFINITY, 1.0, cdf(f64::INFINITY));
    // }

    // #[test]
    // fn test_cdf_lower_bound() {
    //     let cdf = |arg: f64| move |x: MultivariateUniform| x.cdf(arg);
    //     test_case(0.0, 3.0, 0.0, cdf(-1.0));
    // }

    // #[test]
    // fn test_cdf_upper_bound() {
    //     let cdf = |arg: f64| move |x: MultivariateUniform| x.cdf(arg);
    //     test_case(0.0, 3.0, 1.0, cdf(5.0));
    // }


    // #[test]
    // fn test_sf() {
    //     let sf = |arg: f64| move |x: MultivariateUniform| x.sf(arg);
    //     test_case(0.0, 0.0, 1.0, sf(0.0));
    //     test_case(0.0, 0.1, 0.5, sf(0.05));
    //     test_case(0.0, 1.0, 0.5, sf(0.5));
    //     test_case(0.0, 10.0, 0.9, sf(1.0));
    //     test_case(0.0, 10.0, 0.5, sf(5.0));
    //     test_case(-5.0, 100.0, 1.0, sf(-5.0));
    //     test_case(-5.0, 100.0, 0.9523809523809523, sf(0.0));
    //     test_case(0.0, f64::INFINITY, 1.0, sf(10.0));
    //     test_case(0.0, f64::INFINITY, 0.0, sf(f64::INFINITY));
    // }

    // #[test]
    // fn test_sf_lower_bound() {
    //     let sf = |arg: f64| move |x: MultivariateUniform| x.sf(arg);
    //     test_case(0.0, 3.0, 1.0, sf(-1.0));
    // }

    // #[test]
    // fn test_sf_upper_bound() {
    //     let sf = |arg: f64| move |x: MultivariateUniform| x.sf(arg);
    //     test_case(0.0, 3.0, 0.0, sf(5.0));
    // }

    // #[test]
    // fn test_continuous() {
    //     test::check_continuous_distribution(&try_create(0.0, 10.0), 0.0, 10.0);
    //     test::check_continuous_distribution(&try_create(-2.0, 15.0), -2.0, 15.0);
    // }

    // #[test]
    // fn test_samples_in_range() {
    //     use rand::rngs::StdRng;
    //     use rand::SeedableRng;
    //     use rand::distributions::Distribution;

    //     let seed = [
    //         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    //         19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    //     ];
    //     let mut r: StdRng = SeedableRng::from_seed(seed);

    //     let min = -0.5;
    //     let max = 0.5;
    //     let num_trials = 10_000;
    //     let n = try_create(min, max);

    //     assert!((0..num_trials)
    //         .map(|_| n.sample::<StdRng>(&mut r))
    //         .all(|v| (min <= v) && (v < max))
    //     );
    // }
}
