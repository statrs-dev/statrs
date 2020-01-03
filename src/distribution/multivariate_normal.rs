use crate::distribution::Continuous;
use crate::distribution::Normal;
use crate::statistics::{Covariance, Entropy, Max, Mean, Min, Mode};
use crate::{Result, StatsError};
use nalgebra::{
    base::allocator::Allocator,
    base::{dimension::DimName, MatrixN, VectorN},
    Cholesky, DefaultAllocator, Dim, DimMin, RealField, LU, U1,
};
use num_traits::bounds::Bounded;
use rand::distributions::Distribution;
use rand::Rng;

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
/// let mvn = MultivariateNormal::<f64, U2>::new(&Vector2::<f64>::zeros(), &Matrix2::<f64>::identity()).unwrap();
/// assert_eq!(mvn.mean(), Vector2::<f64>::new(0., 0.));
/// assert_eq!(mvn.variance(), Matrix2::<f64>::new(1., 0., 0., 1.));
/// assert_eq!(mvn.pdf(Vector2::<f64>::new(1., 1.)), 0.05854983152431917);
/// ```
#[derive(Debug, Clone)]
pub struct MultivariateNormal<Real, N>
where
    Real: RealField,
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<Real, N>,
    DefaultAllocator: Allocator<Real, N, N>,
    DefaultAllocator: Allocator<Real, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    cov_chol_decomp: MatrixN<Real, N>,
    mu: VectorN<Real, N>,
    cov: MatrixN<Real, N>,
    precision: MatrixN<Real, N>,
    pdf_const: Real,
}

impl<Real, N> MultivariateNormal<Real, N>
where
    Real: RealField,
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<Real, N>,
    DefaultAllocator: Allocator<Real, N, N>,
    DefaultAllocator: Allocator<Real, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    ///  Constructs a new multivariate normal distribution with a mean of `mean`
    /// and covariance matrix `cov`
    ///
    /// # Errors
    ///
    /// Returns an error if the given covariance matrix is not
    /// symmetric or positive-definite
    pub fn new(mean: &VectorN<Real, N>, cov: &MatrixN<Real, N>) -> Result<Self> {
        // Check that the provided covariance matrix is symmetric
        if cov.lower_triangle() != cov.upper_triangle().transpose() {
            return Err(StatsError::BadParams);
        }
        let cov_det = LU::new(cov.clone()).determinant();
        let pdf_const = (Real::two_pi()
            .powi(mean.nrows() as i32)
            .recip()
            .mul(cov_det.abs()))
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
                pdf_const: pdf_const,
            }),
        }
    }
}

impl<N> Distribution<VectorN<f64, N>> for MultivariateNormal<f64, N>
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

impl<Real, N> Min<VectorN<Real, N>> for MultivariateNormal<Real, N>
where
    Real: RealField,
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<Real, N>,
    DefaultAllocator: Allocator<Real, N, N>,
    DefaultAllocator: Allocator<Real, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Returns the minimum value in the domain of the
    /// multivariate normal distribution represented by a real vector
    fn min(&self) -> VectorN<Real, N> {
        VectorN::min_value()
    }
}

impl<Real, N> Max<VectorN<Real, N>> for MultivariateNormal<Real, N>
where
    Real: RealField,
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<Real, N>,
    DefaultAllocator: Allocator<Real, N, N>,
    DefaultAllocator: Allocator<Real, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Returns the maximum value in the domain of the
    /// multivariate normal distribution represented by a real vector
    fn max(&self) -> VectorN<Real, N> {
        VectorN::max_value()
    }
}

impl<Real, N> Mean<VectorN<Real, N>> for MultivariateNormal<Real, N>
where
    Real: RealField,
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<Real, N>,
    DefaultAllocator: Allocator<Real, N, N>,
    DefaultAllocator: Allocator<Real, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Returns the mean of the normal distribution
    ///
    /// # Remarks
    ///
    /// This is the same mean used to construct the distribution
    fn mean(&self) -> VectorN<Real, N> {
        self.mu.clone()
    }
}

impl<Real, N> Covariance<MatrixN<Real, N>> for MultivariateNormal<Real, N>
where
    Real: RealField,
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<Real, N>,
    DefaultAllocator: Allocator<Real, N, N>,
    DefaultAllocator: Allocator<Real, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    /// Returns the covariance matrix of the multivariate normal distribution
    fn variance(&self) -> MatrixN<Real, N> {
        self.cov.clone()
    }
}

impl<Real, N> Entropy<Real> for MultivariateNormal<Real, N>
where
    Real: RealField,
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<Real, N>,
    DefaultAllocator: Allocator<Real, N, N>,
    DefaultAllocator: Allocator<Real, U1, N>,
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
    fn entropy(&self) -> Real {
        LU::new(self.variance().clone().scale(Real::two_pi() * Real::e()))
            .determinant()
            .ln()
    }
}

impl<Real, N> Mode<VectorN<Real, N>> for MultivariateNormal<Real, N>
where
    Real: RealField,
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<Real, N>,
    DefaultAllocator: Allocator<Real, N, N>,
    DefaultAllocator: Allocator<Real, U1, N>,
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
    fn mode(&self) -> VectorN<Real, N> {
        self.mu.clone()
    }
}

impl<Real, N> Continuous<VectorN<Real, N>, Real> for MultivariateNormal<Real, N>
where
    Real: RealField,
    N: Dim + DimMin<N, Output = N> + DimName,
    DefaultAllocator: Allocator<Real, N>,
    DefaultAllocator: Allocator<Real, N, N>,
    DefaultAllocator: Allocator<Real, U1, N>,
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
    fn pdf(&self, x: VectorN<Real, N>) -> Real {
        let dv = x - &self.mu;
        let exp_term = nalgebra::convert::<f64, Real>(-0.5)
            * *(&dv.transpose() * &self.precision * &dv)
                .get((0, 0))
                .unwrap();
        self.pdf_const * exp_term.exp()
    }
    /// Calculates the log probability density function for the multivariate
    /// normal distribution at `x`
    ///
    /// # Formula
    ///
    /// ```ignore
    /// ln((2 * π) ^ (-k / 2) * det(Σ) ^ (1 / 2) * e ^ ( -(1 / 2) * transpose(x - μ) * inv(Σ) * (x - μ)))
    /// ```
    ///
    /// where `μ` is the mean, `inv(Σ)` is the precision matrix, `det(Σ)` is the determinant
    /// of the covariance matrix, and `k` is the dimension of the distribution
    fn ln_pdf(&self, x: VectorN<Real, N>) -> Real {
        let dv = x - &self.mu;
        let exp_term = nalgebra::convert::<f64, Real>(-0.5)
            * *(&dv.transpose() * &self.precision * &dv)
                .get((0, 0))
                .unwrap();
        self.pdf_const.ln() + exp_term
    }
}
