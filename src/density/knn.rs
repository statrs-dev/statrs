use super::Container;
use crate::{
    density::{DensityError, DensityEstimator},
    function::gamma::gamma,
};
use core::f64::consts::PI;

impl<S, X> DensityEstimator<'_, S, X>
where
    S: AsRef<[X]> + Container,
    X: AsRef<[f64]> + Container + PartialEq,
{
    /// Computes the `k`-nearest neighbor density estimate at `x`.
    ///
    /// The optimal `k` is computed using [Orava's](https://www.sav.sk/journals/uploads/0127102604orava.pdf)
    /// formula when `bandwidth` is `None`.
    ///
    /// # Errors
    ///
    /// Returns [`DensityError::EmptyNeighborhood`] if no sample falls inside the
    /// neighborhood of `x`.
    pub fn knn_pdf(&self, x: &X, bandwidth: Option<f64>) -> Result<f64, DensityError> {
        let neighbors = self.nearest_neighbors(x, bandwidth)?;
        if neighbors.is_empty() {
            return Err(DensityError::EmptyNeighborhood);
        }
        // k / (n * V_d(r)), with V_d(r) = pi^(d/2) r^d / Gamma(d/2 + 1) the
        // volume of the d-ball containing the neighborhood
        let radius = neighbors.radius;
        let d = x.length() as f64;
        Ok((neighbors.k / self.n_samples())
            * (gamma(d / 2. + 1.) / (PI.powf(d / 2.) * radius.powf(d))))
    }
}

/// Computes the `k`-nearest neighbor density estimate for a given point `x`
/// using the samples provided.
///
/// The optimal `k` is computed using [Orava's](https://www.sav.sk/journals/uploads/0127102604orava.pdf)
/// formula when `bandwidth` is `None`.
///
/// # Performance
///
/// This builds a k-d tree over `samples` on every call. To evaluate the density
/// at more than a couple of points, build a
/// [`DensityEstimator`](crate::density::DensityEstimator) once and call
/// [`DensityEstimator::knn_pdf`] instead - about 5x faster over a grid.
///
/// # Examples
///
/// ```
/// use statrs::density::knn::knn_pdf;
///
/// let samples: Vec<[f64; 1]> = vec![[-1.0], [0.0], [1.0]];
/// let density = knn_pdf(&[0.0], &samples, Some(1.0)).unwrap();
/// assert!(density > 0.0);
/// ```
pub fn knn_pdf<X, S>(x: &X, samples: &S, bandwidth: Option<f64>) -> Result<f64, DensityError>
where
    S: AsRef<[X]> + Container,
    X: AsRef<[f64]> + Container + PartialEq,
{
    DensityEstimator::new(samples)?.knn_pdf(x, bandwidth)
}

#[cfg(test)]
mod tests {
    use core::f32::consts::PI;

    use super::*;
    use crate::distribution::Normal;
    use crate::function::kernel::Kernel;
    use nalgebra::{Vector1, Vector2};
    use rand::SeedableRng;
    use rand::distr::Distribution;
    use rand::rngs::StdRng;

    #[test]
    fn test_knn_pdf() {
        let law = Normal::new(0., 1.).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let gaussian = crate::function::kernel::Gaussian;
        let samples_1d = (0..100000)
            .map(|_| Vector1::new(law.sample(&mut rng)))
            .collect::<Vec<_>>();
        let x = Vector1::new(0.);
        let knn_density_with_bandwidth = knn_pdf(&x, &samples_1d, Some(0.05));
        let knn_density = knn_pdf(&x, &samples_1d, None);
        let reference_value = gaussian.evaluate(0.);
        assert!(knn_density.is_ok());
        assert!(knn_density_with_bandwidth.is_ok());
        assert!((knn_density.unwrap() - reference_value).abs() < 2e-2);
        assert!((knn_density_with_bandwidth.unwrap() - reference_value).abs() < 3e-2);

        let samples_2d = (0..100000)
            .map(|_| Vector2::new(law.sample(&mut rng), law.sample(&mut rng)))
            .collect::<Vec<_>>();

        let x = Vector2::new(0., 0.);
        let knn_density_with_bandwidth = knn_pdf(&x, &samples_2d, Some(0.05));
        let knn_density = knn_pdf(&x, &samples_2d, None);
        let reference_value = 1. / (2. * PI) as f64;
        assert!(knn_density.is_ok());
        assert!(knn_density_with_bandwidth.is_ok());
        assert!((knn_density.unwrap() - reference_value).abs() < 2e-2);
        assert!((knn_density_with_bandwidth.unwrap() - reference_value).abs() < 3e-2);
    }

    #[test]
    fn test_knn_pdf_empty_samples() {
        let samples: Vec<[f64; 1]> = vec![];
        let x = 3.0;
        let result = knn_pdf(&[x], &samples, None);
        assert!(matches!(result, Err(DensityError::EmptySample)));
    }
}
