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
/// assert_eq!(n.mean().unwrap(), DVector::from_vec(vec![-0.5, 0.5]);
/// assert_eq!(n.pdf(DVector::from_vec(vec![-0.5, 0.5])), 1.);
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
            || min.iter().zip(max.iter()).any(|(f, g)| f > g)
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
                    sample_limits_unchecked[i] = Some(RandUniform::new_inclusive(min[i], max[i]))
                }
                sample_limits = Some(DVector::from_vec(
                    sample_limits_unchecked
                        .into_iter()
                        .flatten()
                        .collect::<Vec<RandUniform<f64>>>(),
                ));
            }
            Ok(MultivariateUniform {
                dim,
                min,
                max,
                sample_limits,
            })
        }
    }

    /// Returns a uniform distribution on the unit hypercube [0,1]^dim
    pub fn standard(dim: usize) -> Result<MultivariateUniform> {
        let mut sample_limits: Vec<Option<RandUniform<f64>>> = vec![None; dim];
        let min = DVector::from_vec(vec![0.; dim]);
        let max = DVector::from_vec(vec![1.; dim]);
        for i in 0..dim {
            sample_limits[i] = Some(RandUniform::new_inclusive(0., 1.))
        }
        let sample_limits = Some(DVector::from_vec(
            sample_limits
                .into_iter()
                .flatten()
                .collect::<Vec<RandUniform<f64>>>(),
        ));
        Ok(MultivariateUniform {
            dim,
            min,
            max,
            sample_limits,
        })
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
            }
            None => None,
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
        let mut p = 1.;
        for i in 0..self.dim {
            if x[i] <= self.min[i]
                || (self.max[i].is_infinite() || self.min[i].is_infinite()) && x[i] < self.max[i]
            {
                p = 0.;
                break;
            } else if x[i] < self.max[i] {
                p *= (x[i] - self.min[i]) / (self.max[i] - self.min[i])
            }
        }
        return p;
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
        let mut p = 1.;
        for i in 0..self.dim {
            if x[i] >= self.max[i]
                || (self.max[i].is_infinite() || self.min[i].is_infinite()) && x[i] > self.min[i]
            {
                p = 0.;
                break;
            } else if x[i] > self.min[i] {
                p *= (self.max[i] - x[i]) / (self.max[i] - self.min[i])
            }
        }
        return p;
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
        if x.iter()
            .zip(self.min.iter().zip(self.max.iter()))
            .any(|(f, (g, h))| f < g || f > h)
        {
            0.0
        } else {
            1. / self
                .min
                .iter()
                .zip(self.max.iter())
                .map(|(f, g)| g - f)
                .product::<f64>()
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
    /// ln(1 / ∏(max - min)) = -∑ln(max -min))
    /// ```
    fn ln_pdf(&self, x: &'a DVector<f64>) -> f64 {
        if x.iter()
            .zip(self.min.iter().zip(self.max.iter()))
            .any(|(f, (g, h))| f < g || f > h)
        {
            f64::NEG_INFINITY
        } else {
            -self
                .min
                .iter()
                .zip(self.max.iter())
                .map(|(f, g)| (g - f).ln())
                .sum::<f64>()
        }
    }
}

#[rustfmt::skip]
#[cfg(all(test, feature = "nightly"))]
mod tests {
    use crate::statistics::*;
    use crate::distribution::{
        ContinuousCDF, Continuous, MultivariateUniform, ContinuousMultivariateCDF
    };
    use crate::distribution::internal::*;
    use crate::consts::ACC;
    use core::fmt::Debug;
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

    fn get_value<F, T>(min: Vec<f64>, max: Vec<f64>, eval: F) -> T
        where F: FnOnce(MultivariateUniform) -> T
    {
        let n = try_create(min, max);
        eval(n)
    }

    fn test_case<F, T>(min: Vec<f64>, max: Vec<f64>, expected: T, eval: F)
        where 
            T: Debug + PartialEq,
            F: FnOnce(MultivariateUniform) -> T
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
        create_case(vec![0.; 100], vec![1.; 100]);
        create_case(vec![1.; 5], vec![1.; 5]);
    }

    #[test]
    fn test_bad_create() {
        bad_create_case(vec![f64::NAN, 5.], vec![0., 6.]);
        bad_create_case(vec![1., 1.,], vec![0., 0.]);
        bad_create_case(vec![0., 0.], vec![-1., 1.]);
        bad_create_case(vec![0., 0.,], vec![f64::NAN, 3.]);
        bad_create_case(vec![0.], vec![1., 2.]);
        bad_create_case(vec![0.; 10], vec![-1.; 10]);
    }

    #[test]
    fn test_mode() {
        let mode = |x: MultivariateUniform| x.mode().unwrap();
        test_case(vec![0., 0.,], vec![1., 1.,], dvec![0.5, 0.5], mode);
        test_case(vec![-1., 0.], vec![1., 2.], dvec![0., 1.], mode);
        test_case(vec![0., 0., f64::NEG_INFINITY], vec![1., 1., 0.], dvec![0.5, 0.5, f64::NEG_INFINITY], mode);
        test_case(vec![0.], vec![f64::INFINITY], dvec![f64::INFINITY], mode);
    }

    #[test]
    fn test_median() {
        let median = |x: MultivariateUniform| x.median();
        test_case(vec![-5., 5.], vec![0., 10.], dvec![-2.5, 7.5], median);
        test_case(vec![10.0, 5.0], vec![11.0, 6.0], dvec![10.5, 5.5], median);
        test_case(vec![0., 0., 0.,], vec![1., 2., 3.,], dvec![0.5, 1., 1.5], median);
        test_case(vec![f64::NEG_INFINITY, f64::NEG_INFINITY], vec![0., 0.], dvec![f64::NEG_INFINITY, f64::NEG_INFINITY], median);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: Vec<f64>| move |x: MultivariateUniform| x.pdf(&DVector::from_vec(arg));
        test_case(vec![0., 0.,], vec![1., 1.,], 0.0, pdf(vec![-10., -10.]));
        test_case(vec![0., 0.,], vec![1., 1.,], 0.0, pdf(vec![-10., 0.5]));
        test_case(vec![0., 0.,], vec![1., 1.,], 0.0, pdf(vec![2., 0.5]));
        test_case(vec![0., 0.,], vec![1., 1.,], 0.0, pdf(vec![2., 2.]));
        test_case(vec![0., 0.,], vec![1., 1.,], 1., pdf(vec![0.5, 0.5]));
        test_case(vec![0., 0.,], vec![1., 1.,], 1., pdf(vec![1., 0.8]));
        test_case(vec![0., 0.,], vec![1., 1.,], 1., pdf(vec![1., 1.]));
        test_case(vec![-5., -10.], vec![5., 10.], 0.005, pdf(vec![0., 0.]));
        test_case(vec![0., f64::NEG_INFINITY], vec![1., 0.], 0., pdf(vec![0.5, -1.]));
        test_case(vec![-1., -2., -4.], vec![1., 2., 4.,], 0.5*0.25*0.125, pdf(vec![0., 0., 0.]));
        test_case(vec![0.], vec![f64::INFINITY], 0., pdf(vec![200.]))
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: Vec<f64>| move |x: MultivariateUniform| x.ln_pdf(&DVector::from_vec(arg));
        test_case(vec![0., 0.,], vec![1., 1.,], 0., ln_pdf(vec![1., 1.]));
        test_case(vec![0., 0.,], vec![1., 1.,], 0., ln_pdf(vec![0., 0.]));
        test_case(vec![0., 0.,], vec![1., 1.,], 0., ln_pdf(vec![0.5, 0.2]));
        test_case(vec![-1., -1.], vec![1., 1.], f64::NEG_INFINITY, ln_pdf(vec![-2., -2.]));
        test_case(vec![0.; 10], vec![1.; 10], 0., ln_pdf(vec![0.5; 10]));
        test_case(vec![0.; 10], vec![f64::INFINITY; 10], f64::NEG_INFINITY, ln_pdf(vec![f64::NEG_INFINITY; 10]));
        test_case(vec![0.; 10], vec![f64::INFINITY; 10], f64::NEG_INFINITY, ln_pdf(vec![f64::INFINITY; 10]));
    }

    #[test]
    fn test_cdf() {
        let cdf = |arg: Vec<f64>| move |x: MultivariateUniform| x.cdf(DVector::from_vec(arg));
        test_case(vec![0., 0.], vec![1., 1.], 0.25, cdf(vec![0.5, 0.5]));
        test_case(vec![0., 0.], vec![1., 1.], 0.0, cdf(vec![-1., 0.5]));
        test_case(vec![0., 0.], vec![1., 1.], 0.0, cdf(vec![0.5, -1.]));
        test_case(vec![0., 0.], vec![1., 1.], 1., cdf(vec![1., 1.]));
        test_case(vec![0., 0.], vec![1., 1.], 1., cdf(vec![2., 2.]));
        test_case(vec![0.; 100], vec![1.; 100], 0.5_f64.powi(100), cdf(vec![0.5; 100]));
        test_case(vec![0.; 100], vec![1.; 100], 1., cdf(vec![1.5; 100]));
        test_case(vec![0.; 5], vec![1.; 5], 0., cdf(vec![1., 1., 1., 1., -1.]));
        test_case(vec![0.; 5], vec![1.; 5], 0.5, cdf(vec![1., 1., 1., 1., 0.5]));
        test_case(vec![f64::NEG_INFINITY, 0.], vec![0., f64::INFINITY], 0., cdf(vec![-1., 1.]));
        test_case(vec![f64::NEG_INFINITY, 0.], vec![0., f64::INFINITY], 1., cdf(vec![0., f64::INFINITY]));
    }

    #[test]
    fn test_sf() {
        let sf = |arg: Vec<f64>| move |x: MultivariateUniform| x.sf(DVector::from_vec(arg));
        test_case(vec![0., 0.], vec![1., 1.], 0.25, sf(vec![0.5, 0.5]));
        test_case(vec![0., 0.], vec![1., 1.], 0.5, sf(vec![-1., 0.5]));
        test_case(vec![0., 0.], vec![1., 1.], 0.5, sf(vec![0.5, -1.]));
        test_case(vec![0., 0.], vec![1., 1.], 0., sf(vec![1., 1.]));
        test_case(vec![0., 0.], vec![1., 1.], 0., sf(vec![2., 2.]));
        test_case(vec![0.; 100], vec![1.; 100], 0.5_f64.powi(100), sf(vec![0.5; 100]));
        test_case(vec![0.; 100], vec![1.; 100], 0., sf(vec![1.5; 100]));
        test_case(vec![0.; 5], vec![1.; 5], 0., sf(vec![1., 1., 1., 1., -1.]));
        test_case(vec![0.; 5], vec![1.; 5], 0.5, sf(vec![0., 0., 0., 0., 0.5]));
        test_case(vec![f64::NEG_INFINITY, 0.], vec![0., f64::INFINITY], 0., sf(vec![-1., 1.]));
        test_case(vec![f64::NEG_INFINITY, 0.], vec![0., f64::INFINITY], 1., sf(vec![f64::NEG_INFINITY, 0.]));
    }
}
