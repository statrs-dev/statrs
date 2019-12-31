use crate::distribution::Continuous;
use crate::distribution::Normal;
use crate::statistics::{Covariance, Entropy, Max, Mean, Min, Mode};
use crate::{Result, StatsError};
use nalgebra::{
    base::allocator::Allocator,
    base::{dimension::DimName, dimension::DimSub, MatrixN, VectorN},
    Cholesky, DefaultAllocator, Dim, DimMin, Dynamic, RealField, LU, U1,
};
use num_traits::bounds::Bounded;
use rand::distributions::Distribution;
use rand::Rng;

pub struct MultivariateNormal<Real, N>
where
    Real: RealField,
    N: Dim + DimMin<N, Output = N>,
    DefaultAllocator: Allocator<Real, N>,
    DefaultAllocator: Allocator<Real, N, N>,
    DefaultAllocator: Allocator<Real, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    mvn: nalgebra_mvn::MultivariateNormal<Real, N>,
    cov_chol_decomp: MatrixN<Real, N>,
}

impl<Real, N> MultivariateNormal<Real, N>
where
    Real: RealField,
    N: Dim + DimMin<N, Output = N> + DimSub<Dynamic>,
    DefaultAllocator: Allocator<Real, N>,
    DefaultAllocator: Allocator<Real, N, N>,
    DefaultAllocator: Allocator<Real, U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as DimMin<N>>::Output>,
{
    pub fn new(mean: &VectorN<Real, N>, cov: &MatrixN<Real, N>) -> Result<Self> {
        match nalgebra_mvn::MultivariateNormal::from_mean_and_covariance(&mean, &cov.clone()) {
            Ok(mvn) => {
                // Store the Cholesky decomposition of the covariance matrix
                // for sampling
                let cholesky_decomp = Cholesky::new(cov.clone()).unwrap().unpack();
                Ok(MultivariateNormal {
                    mvn: mvn,
                    cov_chol_decomp: cholesky_decomp,
                })
            }
            Err(_) => Err(StatsError::BadParams),
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
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> VectorN<f64, N> {
        let d = Normal::new(0., 1.).unwrap();
        let z = VectorN::<f64, N>::from_distribution(&d, rng);
        (self.cov_chol_decomp.clone() * z) + self.mvn.mean()
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
    fn mean(&self) -> VectorN<Real, N> {
        self.mvn.mean()
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
    fn variance(&self) -> MatrixN<Real, N> {
        Cholesky::new(self.mvn.precision().clone())
            .unwrap()
            .inverse()
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
    fn mode(&self) -> VectorN<Real, N> {
        self.mvn.mean()
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
    fn pdf(&self, x: VectorN<Real, N>) -> Real {
        *self.mvn.pdf::<U1>(&x.transpose()).get((0, 0)).unwrap()
    }
    fn ln_pdf(&self, x: VectorN<Real, N>) -> Real {
        *self.mvn.logpdf::<U1>(&x.transpose()).get((0, 0)).unwrap()
    }
}
