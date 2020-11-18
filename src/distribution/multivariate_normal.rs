use crate::distribution::Continuous;
use crate::distribution::Normal;
use crate::statistics::{Covariance, Entropy, Max, Mean, Min, Mode};
use crate::{Result, StatsError};
use nalgebra::{
    base::allocator::Allocator,
    base::{dimension::DimName, MatrixN, VectorN},
    Cholesky, DefaultAllocator, Dim, DimMin, LU, U1,
};
use rand::distributions::Distribution;
use rand::Rng;
use std::f64;
use std::f64::consts::{E, PI};

/// Implements the [Multivariate Normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
/// distribution using the "nalgebra" crate for matrix operations
///
/// # Examples
///
/// ```
/// use statrs::distribution::{MultivariateNormal, Continuous};
/// use nalgebra::base::dimension::U2;
/// use nalgebra::{Vector2, Matrix2};
/// use statrs::statistics::{Mean, Covariance};
///
/// let mvn = MultivariateNormal::<U2>::new(&Vector2::zeros(), &Matrix2::identity()).unwrap();
/// assert_eq!(mvn.mean(), Vector2::new(0., 0.));
/// assert_eq!(mvn.variance(), Matrix2::new(1., 0., 0., 1.));
/// assert_eq!(mvn.pdf(Vector2::new(1., 1.)), 0.05854983152431917);
/// ```
#[derive(Debug, Clone)]
pub struct MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    cov_chol_decomp: MatrixN<f64, N>,
    mu: VectorN<f64, N>,
    cov: MatrixN<f64, N>,
    precision: MatrixN<f64, N>,
    pdf_const: f64,
}

impl<N> MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    ///  Constructs a new multivariate normal distribution with a mean of `mean`
    /// and covariance matrix `cov`
    ///
    /// # Errors
    ///
    /// Returns an error if the given covariance matrix is not
    /// symmetric or positive-definite
    pub fn new(mean: &VectorN<f64, N>, cov: &MatrixN<f64, N>) -> Result<Self> {
        // Check that the provided covariance matrix is symmetric
        // Check that mean and covariance do not contain NaN
        if cov.lower_triangle() != cov.upper_triangle().transpose()
            || mean.iter().any(|f| f.is_nan())
            || cov.iter().any(|f| f.is_nan())
        {
            return Err(StatsError::BadParams);
        }
        let cov_det = LU::new(cov.clone()).determinant();
        let pdf_const = ((2. * PI).powi(mean.nrows() as i32) * cov_det.abs())
            .recip()
            .sqrt();
        // Store the Cholesky decomposition of the covariance matrix
        // for sampling
        match Cholesky::new(cov.clone()) {
            None => Err(StatsError::BadParams),
            Some(cholesky_decomp) => Ok(MultivariateNormal {
                cov_chol_decomp: cholesky_decomp.clone().unpack(),
                mu: mean.clone(),
                cov: cov.clone(),
                precision: cholesky_decomp.inverse(),
                pdf_const,
            }),
        }
    }
}

impl<N> Distribution<VectorN<f64, N>> for MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Samples from the multivariate normal distribution
    ///
    /// # Formula
    /// L * Z + μ
    ///
    /// where `L` is the Cholesky decomposition of the covariance matrix,
    /// `Z` is a vector of normally distributed random variables, and
    /// `μ` is the mean vector

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> VectorN<f64, N> {
        let d = Normal::new(0., 1.).unwrap();
        let z = VectorN::<f64, N>::from_distribution(&d, rng);
        (self.cov_chol_decomp.clone() * z) + self.mu.clone()
    }
}

impl<N> Min<VectorN<f64, N>> for MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Returns the minimum value in the domain of the
    /// multivariate normal distribution represented by a real vector
    fn min(&self) -> VectorN<f64, N> {
        VectorN::<f64, N>::repeat(f64::NEG_INFINITY)
    }
}

impl<N> Max<VectorN<f64, N>> for MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Returns the maximum value in the domain of the
    /// multivariate normal distribution represented by a real vector
    fn max(&self) -> VectorN<f64, N> {
        VectorN::<f64, N>::repeat(f64::INFINITY)
    }
}

impl<N> Mean<VectorN<f64, N>> for MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Returns the mean of the normal distribution
    ///
    /// # Remarks
    ///
    /// This is the same mean used to construct the distribution
    fn mean(&self) -> VectorN<f64, N> {
        self.mu.clone()
    }
}

impl<N> Covariance<MatrixN<f64, N>> for MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Returns the covariance matrix of the multivariate normal distribution
    fn variance(&self) -> MatrixN<f64, N> {
        self.cov.clone()
    }
}

impl<N> Entropy<f64> for MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Returns the entropy of the multivariate normal distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// (1 / 2) * ln(det(2 * π * e * Σ))
    /// ```
    ///
    /// where `Σ` is the covariance matrix and `det` is the determinant
    fn entropy(&self) -> f64 {
        0.5 * LU::new(self.variance().scale(2. * PI * E))
            .determinant()
            .ln()
    }
}

impl<N> Mode<VectorN<f64, N>> for MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Returns the mode of the multivariate normal distribution
    ///
    /// # Formula
    ///
    /// ```ignore
    /// μ
    /// ```
    ///
    /// where `μ` is the mean
    fn mode(&self) -> VectorN<f64, N> {
        self.mu.clone()
    }
}

impl<N> Continuous<VectorN<f64, N>, f64> for MultivariateNormal<N>
where
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<f64, N>,
    DefaultAllocator: Allocator<f64, N, N>,
    DefaultAllocator: Allocator<f64, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
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
    fn pdf(&self, x: VectorN<f64, N>) -> f64 {
        let dv = x - &self.mu;
        let exp_term = -0.5
            * *(&dv.transpose() * &self.precision * &dv)
                .get((0, 0))
                .unwrap();
        self.pdf_const * exp_term.exp()
    }
    /// Calculates the log probability density function for the multivariate
    /// normal distribution at `x`. Equivalent to pdf(x).ln().
    fn ln_pdf(&self, x: VectorN<f64, N>) -> f64 {
        let dv = x - &self.mu;
        let exp_term = -0.5
            * *(&dv.transpose() * &self.precision * &dv)
                .get((0, 0))
                .unwrap();
        self.pdf_const.ln() + exp_term
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod test {
    use std::f64;
    use crate::statistics::*;
    use crate::distribution::{MultivariateNormal, Continuous};
    use nalgebra::{Matrix2, Vector2, Matrix3, Vector3, VectorN, MatrixN, Dim, DimMin, DimName, DefaultAllocator, U1};
    use nalgebra::base::allocator::Allocator;
    use core::fmt::Debug;

    fn try_create<N>(mean: VectorN<f64, N>, covariance: MatrixN<f64, N>) -> MultivariateNormal<N>
        where
          N: Dim + DimMin<N, Output = N> + DimName,
          DefaultAllocator: Allocator<f64, N>,
          DefaultAllocator: Allocator<f64, N, N>,
          DefaultAllocator: Allocator<f64, U1, N>,
          DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
    {
        let mvn = MultivariateNormal::new(&mean, &covariance);
        assert!(mvn.is_ok());
        mvn.unwrap()
    }

    fn create_case<N>(mean: VectorN<f64, N>, covariance: MatrixN<f64, N>)
        where
          N: Dim + DimMin<N, Output = N> + DimName,
          DefaultAllocator: Allocator<f64, N>,
          DefaultAllocator: Allocator<f64, N, N>,
          DefaultAllocator: Allocator<f64, U1, N>,
          DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
    {
        let mvn = try_create(mean.clone(), covariance.clone());
        assert_eq!(mean, mvn.mean());
        assert_eq!(covariance, mvn.variance());
    }

    fn bad_create_case<N>(mean: VectorN<f64, N>, covariance: MatrixN<f64, N>)
        where
          N: Dim + DimMin<N, Output = N> + DimName,
          DefaultAllocator: Allocator<f64, N>,
          DefaultAllocator: Allocator<f64, N, N>,
          DefaultAllocator: Allocator<f64, U1, N>,
          DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
    {
        let mvn = MultivariateNormal::new(&mean, &covariance);
        assert!(mvn.is_err());
    }

    fn test_case<T, F, N>(mean: VectorN<f64, N>, covariance: MatrixN<f64, N>, expected: T, eval: F)
        where
            T: Debug + PartialEq,
            F: Fn(MultivariateNormal<N>) -> T,
            N: Dim + DimMin<N, Output = N> + DimName,
            DefaultAllocator: Allocator<f64, N>,
            DefaultAllocator: Allocator<f64, N, N>,
            DefaultAllocator: Allocator<f64, U1, N>,
            DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
    {
        let mvn = try_create(mean, covariance);
        let x = eval(mvn);
        assert_eq!(expected, x);
    }

    fn test_almost<F, N>(mean: VectorN<f64, N>, covariance: MatrixN<f64, N>, expected: f64, acc: f64, eval: F)
        where
          F: Fn(MultivariateNormal<N>) -> f64,
          N: Dim + DimMin<N, Output = N> + DimName,
          DefaultAllocator: Allocator<f64, N>,
          DefaultAllocator: Allocator<f64, N, N>,
          DefaultAllocator: Allocator<f64, U1, N>,
          DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
    {
        let mvn = try_create(mean, covariance);
        let x = eval(mvn);
        assert_almost_eq!(expected, x, acc);
    }

    #[test]
    fn test_create() {
        create_case(Vector2::new(0., 0.), Matrix2::new(1., 0., 0., 1.));
        create_case(Vector2::new(10., 5.), Matrix2::new(2., 1., 1., 2.));
        create_case(Vector3::new(4., 5., 6.), Matrix3::new(2., 1., 0., 1., 2., 1., 0., 1., 2.));
        create_case(Vector2::new(0., f64::INFINITY), Matrix2::identity());
        create_case(Vector2::zeros(), Matrix2::new(f64::INFINITY, 0., 0., f64::INFINITY));
    }

    #[test]
    fn test_bad_create() {
        // Covariance not symmetric
        bad_create_case(Vector2::zeros(), Matrix2::new(1., 1., 0., 1.));
        // Covariance not positive-definite
        bad_create_case(Vector2::zeros(), Matrix2::new(1., 2., 2., 1.));
        // NaN in mean
        bad_create_case(Vector2::new(0., f64::NAN), Matrix2::identity());
        // NaN in Covariance Matrix
        bad_create_case(Vector2::zeros(), Matrix2::new(1., 0., 0., f64::NAN));
    }

    #[test]
    fn test_variance() {
        test_case(Vector2::zeros(), Matrix2::identity(), Matrix2::new(1., 0., 0., 1.), |x| x.variance());
        test_case(Vector2::zeros(), Matrix2::new(f64::INFINITY, 0., 0., f64::INFINITY), Matrix2::new(f64::INFINITY, 0., 0., f64::INFINITY), |x| x.variance());
    }

    #[test]
    fn test_entropy() {
        test_case(Vector2::zeros(), Matrix2::identity(), 2.8378770664093453, |x| x.entropy());
        test_case(Vector2::zeros(), Matrix2::new(1., 0.5, 0.5, 1.), 2.694036030183455, |x| x.entropy());
        test_case(Vector2::zeros(), Matrix2::new(f64::INFINITY, 0., 0., f64::INFINITY), f64::INFINITY, |x| x.entropy());
    }

    #[test]
    fn test_mode() {
        test_case(Vector2::zeros(), Matrix2::identity(), Vector2::new(0., 0.), |x| x.mode());
        test_case(Vector2::<f64>::repeat(f64::INFINITY), Matrix2::identity(), Vector2::new(f64::INFINITY, f64::INFINITY), |x| x.mode());
    }

    #[test]
    fn test_min_max() {
        test_case(Vector2::zeros(), Matrix2::identity(), Vector2::new(f64::NEG_INFINITY, f64::NEG_INFINITY), |x| x.min());
        test_case(Vector2::zeros(), Matrix2::identity(), Vector2::new(f64::INFINITY, f64::INFINITY), |x| x.max());
        test_case(Vector2::new(10., 1.), Matrix2::identity(), Vector2::new(f64::NEG_INFINITY, f64::NEG_INFINITY), |x| x.min());
        test_case(Vector2::new(-3., 5.), Matrix2::identity(), Vector2::new(f64::INFINITY, f64::INFINITY), |x| x.max());
    }

    #[test]
    fn test_pdf() {
        test_case(Vector2::zeros(), Matrix2::identity(), 0.05854983152431917, |x| x.pdf(Vector2::new(1., 1.)));
        test_almost(Vector2::zeros(), Matrix2::identity(), 0.013064233284684921, 1e-15, |x| x.pdf(Vector2::new(1., 2.)));
        test_almost(Vector2::zeros(), Matrix2::identity(), 1.8618676045881531e-23, 1e-35, |x| x.pdf(Vector2::new(1., 10.)));
        test_almost(Vector2::zeros(), Matrix2::identity(), 5.920684802611216e-45, 1e-58, |x| x.pdf(Vector2::new(10., 10.)));
        test_almost(Vector2::zeros(), Matrix2::new(1., 0.9, 0.9, 1.), 1.6576716577547003e-05, 1e-18, |x| x.pdf(Vector2::new(1., -1.)));
        test_almost(Vector2::zeros(), Matrix2::new(1., 0.99, 0.99, 1.), 4.1970621773477824e-44, 1e-54, |x| x.pdf(Vector2::new(1., -1.)));
        test_almost(Vector2::new(0.5, -0.2), Matrix2::new(2.0, 0.3, 0.3, 0.5), 0.0013075203140666656, 1e-15, |x| x.pdf(Vector2::new(2., 2.)));
        test_case(Vector2::zeros(), Matrix2::new(f64::INFINITY, 0., 0., f64::INFINITY), 0.0, |x| x.pdf(Vector2::new(10., 10.)));
        test_case(Vector2::zeros(), Matrix2::new(f64::INFINITY, 0., 0., f64::INFINITY), 0.0, |x| x.pdf(Vector2::new(100., 100.)));
    }

    #[test]
    fn test_ln_pdf() {
        test_case(Vector2::zeros(), Matrix2::identity(), (0.05854983152431917f64).ln(), |x| x.ln_pdf(Vector2::new(1., 1.)));
        test_almost(Vector2::zeros(), Matrix2::identity(), (0.013064233284684921f64).ln(), 1e-15, |x| x.ln_pdf(Vector2::new(1., 2.)));
        test_almost(Vector2::zeros(), Matrix2::identity(), (1.8618676045881531e-23f64).ln(), 1e-15, |x| x.ln_pdf(Vector2::new(1., 10.)));
        test_almost(Vector2::zeros(), Matrix2::identity(), (5.920684802611216e-45f64).ln(), 1e-15, |x| x.ln_pdf(Vector2::new(10., 10.)));
        test_almost(Vector2::zeros(), Matrix2::new(1., 0.9, 0.9, 1.), (1.6576716577547003e-05f64).ln(), 1e-14, |x| x.ln_pdf(Vector2::new(1., -1.)));
        test_almost(Vector2::zeros(), Matrix2::new(1., 0.99, 0.99, 1.), (4.1970621773477824e-44f64).ln(), 1e-12, |x| x.ln_pdf(Vector2::new(1., -1.)));
        test_almost(Vector2::new(0.5, -0.2), Matrix2::new(2.0, 0.3, 0.3, 0.5), (0.0013075203140666656f64).ln(), 1e-15, |x| x.ln_pdf(Vector2::new(2., 2.)));
        test_case(Vector2::zeros(), Matrix2::new(f64::INFINITY, 0., 0., f64::INFINITY), f64::NEG_INFINITY, |x| x.ln_pdf(Vector2::new(10., 10.)));
        test_case(Vector2::zeros(), Matrix2::new(f64::INFINITY, 0., 0., f64::INFINITY), f64::NEG_INFINITY, |x| x.ln_pdf(Vector2::new(100., 100.)));
    }
}
