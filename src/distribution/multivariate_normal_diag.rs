use crate::distribution::Continuous;
use crate::distribution::Normal;
use crate::statistics::{Max, MeanN, Min, Mode, VarianceN};
use crate::{consts, Result, StatsError};
use nalgebra::DVector;
use nalgebra::{
    base::allocator::Allocator, base::dimension::DimName, Cholesky, DefaultAllocator, Dim, DimMin,
    Matrix, LU, U1,
};
use rand::Rng;
use std::f64;
use std::f64::consts::{E, LN_2, PI};

/// Implements the [Multivariate Normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
/// distribution with a diagonal covariance matrix using the "nalgebra" crate for vector
/// operations. This specialization enables a considerably more efficient implementation than
/// the full covariance matrix used in the MultivariateNormal distribution.
///
/// # Examples
///
/// ```
/// use statrs::distribution::{MultivariateNormalDiag, Continuous};
/// use nalgebra::DVector;
/// use statrs::statistics::{MeanN, VarianceN};
/// use statrs::assert_almost_eq;
///
/// let mvn = MultivariateNormalDiag::new(vec![0., 0.], vec![1., 1.]).unwrap();
/// assert_eq!(mvn.mean().unwrap(), DVector::from_vec(vec![0., 0.]));
/// assert_eq!(mvn.variance().unwrap(), DVector::from_vec(vec![1., 1.]));
/// assert_almost_eq!(mvn.pdf(&DVector::from_vec(vec![1.,  1.])), 1e-16, 0.05854983152431917);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateNormalDiag {
    mu: DVector<f64>,
    std_dev: DVector<f64>,
}

impl MultivariateNormalDiag {
    /// Constructs a new multivariate normal distribution with a mean of `mean`
    /// and covariance matrix with a diagonal of `std_dev * std_dev`
    ///
    /// # Errors
    ///
    /// Returns an error if `mean` or `std_dev` are `NaN` or if
    /// `std_dev <= 0.0`
    pub fn new(mean: Vec<f64>, std_dev: Vec<f64>) -> Result<Self> {
        let mean = DVector::from_vec(mean);
        let std_dev = DVector::from_vec(std_dev);
        // Check that all std_devs are positive
        if std_dev.iter().any(|&f| f <= 0.)
        // Check that mean and std_dev do not contain NaN
            || mean.iter().any(|f| f.is_nan())
            || std_dev.iter().any(|f| f.is_nan())
        // Check that the dimensions match
            || mean.nrows() != std_dev.nrows()
        {
            return Err(StatsError::BadParams);
        }
        Ok(MultivariateNormalDiag { mu: mean, std_dev })
    }
    /// Returns the entropy of the multivariate normal distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / 2) * ln(det(2 * π * e * Σ))
    /// ```
    ///
    /// where `Σ` is the std_dev matrix and `det` is the determinant
    pub fn entropy(&self) -> Option<f64> {
        Some(self.std_dev.map(f64::ln).sum() + self.std_dev.nrows() as f64 * consts::LN_SQRT_2PIE)
    }
}

impl ::rand::distributions::Distribution<DVector<f64>> for MultivariateNormalDiag {
    /// Samples from the multivariate normal distribution
    ///
    /// # Formula
    /// std_dev * Z + μ
    ///
    /// where `L` is the Cholesky decomposition of the covariance matrix,
    /// `Z` is a vector of normally distributed random variables, and
    /// `μ` is the mean vector

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DVector<f64> {
        let d = Normal::new(0., 1.).unwrap();
        let z = DVector::<f64>::from_distribution(self.mu.nrows(), &d, rng);
        (&self.std_dev.component_mul(&z)) + &self.mu
    }
}

impl Min<DVector<f64>> for MultivariateNormalDiag {
    /// Returns the minimum value in the domain of the
    /// multivariate normal distribution represented by a real vector
    fn min(&self) -> DVector<f64> {
        DVector::from_vec(vec![f64::NEG_INFINITY; self.mu.nrows()])
    }
}

impl Max<DVector<f64>> for MultivariateNormalDiag {
    /// Returns the maximum value in the domain of the
    /// multivariate normal distribution represented by a real vector
    fn max(&self) -> DVector<f64> {
        DVector::from_vec(vec![f64::INFINITY; self.mu.nrows()])
    }
}

impl MeanN<DVector<f64>> for MultivariateNormalDiag {
    /// Returns the mean of the normal distribution
    ///
    /// # Remarks
    ///
    /// This is the same mean used to construct the distribution
    fn mean(&self) -> Option<DVector<f64>> {
        let mut vec = vec![];
        for elt in self.mu.clone().into_iter() {
            vec.push(*elt);
        }
        Some(DVector::from_vec(vec))
    }
}

impl VarianceN<DVector<f64>> for MultivariateNormalDiag {
    /// Returns the variance vector of the multivariate normal distribution
    fn variance(&self) -> Option<DVector<f64>> {
        Some(self.std_dev.component_mul(&self.std_dev))
    }
}

impl Mode<DVector<f64>> for MultivariateNormalDiag {
    /// Returns the mode of the multivariate normal distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// μ
    /// ```
    ///
    /// where `μ` is the mean
    fn mode(&self) -> DVector<f64> {
        self.mu.clone()
    }
}

impl<'a> Continuous<&'a DVector<f64>, f64> for MultivariateNormalDiag {
    /// Calculates the probability density function for the multivariate
    /// normal distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (2 * π) ^ (-k / 2) * det(Σ) ^ (1 / 2) * e ^ ( -(1 / 2) * transpose(x - μ) * inv(Σ) * (x - μ))
    /// ```
    ///
    /// where `μ` is the mean, `inv(Σ)` is the precision matrix, `det(Σ)` is the determinant
    /// of the covariance matrix, and `k` is the dimension of the distribution
    fn pdf(&self, x: &'a DVector<f64>) -> f64 {
        let z = (x - &self.mu).component_div(&self.std_dev);
        // TODO: Use Matrix product from newer nalgebra.
        (-0.5 * z.component_mul(&z).sum()).exp()
            / (&(&self.std_dev * consts::SQRT_2PI))
                .iter()
                .product::<f64>()
    }
    /// Calculates the log probability density function for the multivariate
    /// normal distribution at `x`. Equivalent to pdf(x).ln().
    fn ln_pdf(&self, x: &'a DVector<f64>) -> f64 {
        let z = (x - &self.mu).component_div(&self.std_dev);
        (-0.5 * z.component_mul(&z)).sum()
            - self
                .std_dev
                .map(f64::ln)
                .map(|x| x + consts::LN_SQRT_2PI)
                .sum()
    }
}

#[rustfmt::skip]
#[cfg(all(test, feature = "nightly"))]
mod tests  {
    use crate::distribution::{Continuous, MultivariateNormalDiag};
    use crate::statistics::*;
    use crate::consts::ACC;
    use core::fmt::Debug;
    use nalgebra::base::allocator::Allocator;
    use nalgebra::{
        DefaultAllocator, Dim, DimMin, DimName, DMatrix, Matrix2, Matrix3, Vector2, Vector3,
        U1, U2,
    };
    use rand::rngs::StdRng;
    use rand::distributions::Distribution;
    use rand::prelude::*;

    fn try_create(mean: Vec<f64>, std_dev: Vec<f64>) -> MultivariateNormalDiag
    {
        let mvn = MultivariateNormalDiag::new(mean, std_dev);
        assert!(mvn.is_ok());
        mvn.unwrap()
    }

    fn create_case(mean: Vec<f64>, std_dev: Vec<f64>)
    {
        let mvn = try_create(mean.clone(), std_dev.clone());
        assert_eq!(DVector::from_vec(mean.clone()), mvn.mean().unwrap());
        let std_dev = DVector::from_vec(std_dev);
        assert_eq!(std_dev.component_mul(&std_dev), mvn.variance().unwrap());
    }

    fn bad_create_case(mean: Vec<f64>, std_dev: Vec<f64>)
    {
        let mvn = MultivariateNormalDiag::new(mean, std_dev);
        assert!(mvn.is_err());
    }

    fn test_case<T, F>(mean: Vec<f64>, std_dev: Vec<f64>, expected: T, eval: F)
    where
        T: Debug + PartialEq,
        F: FnOnce(MultivariateNormalDiag) -> T,
    {
        let mvn = try_create(mean, std_dev);
        let x = eval(mvn);
        assert_eq!(expected, x);
    }

    fn test_almost<F>(
        mean: Vec<f64>,
        std_dev: Vec<f64>,
        expected: f64,
        acc: f64,
        eval: F,
    ) where
        F: FnOnce(MultivariateNormalDiag) -> f64,
    {
        let mvn = try_create(mean, std_dev);
        let x = eval(mvn);
        assert_almost_eq!(expected, x, acc);
    }

    use super::*;

    macro_rules! dvec {
        ($($x:expr),*) => (DVector::from_vec(vec![$($x),*]));
    }

    #[test]
    fn test_create() {
        create_case(vec![0., 0.], vec![1., 1.]);
        create_case(vec![10.,  5.], vec![2., 2.]);
        create_case(vec![4., 5., 6.], vec![2., 2., 2.]);
        create_case(vec![0., f64::INFINITY], vec![1., 1.]);
        create_case(vec![0., 0.], vec![f64::INFINITY, f64::INFINITY]);
    }

    #[test]
    fn test_bad_create() {
        // std_dev not positive
        bad_create_case(vec![0., 0.], vec![0., 1.]);
        // NaN in mean
        bad_create_case(vec![0., f64::NAN], vec![1., 1.]);
        // NaN in std_dev
        bad_create_case(vec![0., 0.], vec![1., f64::NAN]);
    }

    #[test]
    fn test_variance() {
        let variance = |x: MultivariateNormalDiag| x.variance().unwrap();
        test_case(vec![0., 0.], vec![1., 1.], dvec![1., 1.], variance);
        test_case(vec![0., 0.], vec![2., 2.], dvec![4., 4.], variance);
        test_case(vec![0., 0.], vec![f64::INFINITY, f64::INFINITY], dvec![f64::INFINITY, f64::INFINITY], variance);
    }

    #[test]
    fn test_entropy() {
        let entropy = |x: MultivariateNormalDiag| x.entropy().unwrap();
        test_case(vec![0., 0.], vec![1., 1.], 2.8378770664093453, entropy);
        test_case(vec![0., 0.], vec![f64::INFINITY, f64::INFINITY], f64::INFINITY, entropy);
    }

    #[test]
    fn test_mode() {
        let mode = |x: MultivariateNormalDiag| x.mode();
        test_case(vec![0., 0.], vec![1., 1.], dvec![0., 0.], mode);
        test_case(vec![f64::INFINITY, f64::INFINITY], vec![1., 1.], dvec![f64::INFINITY,  f64::INFINITY], mode);
    }

    #[test]
    fn test_min_max() {
        let min = |x: MultivariateNormalDiag| x.min();
        let max = |x: MultivariateNormalDiag| x.max();
        test_case(vec![0., 0.], vec![1., 1.], dvec![f64::NEG_INFINITY, f64::NEG_INFINITY], min);
        test_case(vec![0., 0.], vec![1., 1.], dvec![f64::INFINITY, f64::INFINITY], max);
        test_case(vec![10., 1.], vec![1., 1.], dvec![f64::NEG_INFINITY, f64::NEG_INFINITY], min);
        test_case(vec![-3., 5.], vec![1., 1.], dvec![f64::INFINITY, f64::INFINITY], max);
    }

    #[test]
    fn test_pdf() {
        let pdf = |arg: DVector<f64>| move |x: MultivariateNormalDiag| x.pdf(&arg);
        test_almost(vec![0., 0.], vec![1., 1.], 0.05854983152431917, 1e-15, pdf(dvec![1., 1.]));
        test_almost(vec![0., 0.], vec![1., 1.], 0.013064233284684921, 1e-15, pdf(dvec![1., 2.]));
        test_almost(vec![1., 2.], vec![3., 4.], 0.013262911924324607, 1e-15, pdf(dvec![1., 2.]));
        test_almost(vec![0., 0.], vec![1., 1.], 1.8618676045881531e-23, 1e-35, pdf(dvec![1., 10.]));
        test_almost(vec![0., 0.], vec![1., 1.], 5.920684802611216e-45, 1e-58, pdf(dvec![10., 10.]));
        test_almost(vec![1., 1.], vec![1., 1.], 5.920684802611216e-45, 1e-58, pdf(dvec![11., 11.]));
        test_case(vec![0., 0.], vec![f64::INFINITY, f64::INFINITY], 0.0, pdf(dvec![10., 10.]));
        test_case(vec![0., 0.], vec![f64::INFINITY, f64::INFINITY], 0.0, pdf(dvec![100., 100.]));
    }

    #[test]
    fn test_ln_pdf() {
        let ln_pdf = |arg: DVector<_>| move |x: MultivariateNormalDiag| x.ln_pdf(&arg);
        test_almost(vec![0., 0.], vec![1., 1.], (0.05854983152431917f64).ln(), 1e-15, ln_pdf(dvec![1., 1.]));
        test_almost(vec![0., 0.], vec![1., 1.], (0.013064233284684921f64).ln(), 1e-15, ln_pdf(dvec![1., 2.]));
        test_almost(vec![1., 2.], vec![3., 4.], (0.013262911924324607f64).ln(), 1e-15, ln_pdf(dvec![1., 2.]));
        test_almost(vec![0., 0.], vec![1., 1.], (1.8618676045881531e-23f64).ln(), 1e-15, ln_pdf(dvec![1., 10.]));
        test_almost(vec![0., 0.], vec![1., 1.], (5.920684802611216e-45f64).ln(), 1e-15, ln_pdf(dvec![10., 10.]));
        test_case(vec![0., 0.], vec![f64::INFINITY, f64::INFINITY], f64::NEG_INFINITY, ln_pdf(dvec![10., 10.]));
        test_case(vec![0., 0.], vec![f64::INFINITY, f64::INFINITY], f64::NEG_INFINITY, ln_pdf(dvec![100., 100.]));
    }

    #[test]
    fn test_sample() {
        const N: usize = 10000;
        let mean = dvec![1., 2.];
        let std_dev = dvec![3., 4.];
        let mvn = try_create(mean.iter().copied().collect(), std_dev.iter().copied().collect());
        let mut rng = StdRng::seed_from_u64(0);
        let mut samples = DMatrix::zeros(N, mean.nrows());
        for i in 0..N
        {
            samples.set_row(i, &mvn.sample(&mut rng).transpose());
        }

        for (i, &mean) in mean.iter().enumerate()
        {
            let est_mean = samples.column(i).mean();
            assert_almost_eq!(mean, est_mean, 0.1);
        }
        for (i, &std_dev) in std_dev.iter().enumerate()
        {
            let est_std_dev = samples.column(i).std_dev();
            assert_almost_eq!(std_dev, est_std_dev, 0.1);
        }
    }
}
