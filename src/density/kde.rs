use kdtree::distance::squared_euclidean;

use crate::{
    density::{Container, DensityError, DensityEstimator},
    function::kernel::{Gaussian, Kernel},
};

impl<S, X> DensityEstimator<'_, S, X>
where
    S: AsRef<[X]> + Container,
    X: AsRef<[f64]> + Container + PartialEq,
{
    /// Computes the kernel density estimate at `x`, using the distance to the
    /// furthest neighbor as a local bandwidth.
    ///
    /// The optimal `k` is computed using [Orava's](https://www.sav.sk/journals/uploads/0127102604orava.pdf)
    /// formula when `bandwidth` is `None`.
    ///
    /// # Errors
    ///
    /// Returns [`DensityError::EmptyNeighborhood`] if no sample falls inside the
    /// neighborhood of `x`.
    pub fn kde_pdf(&self, x: &X, bandwidth: Option<f64>) -> Result<f64, DensityError> {
        // The neighborhood is used only for its radius: the kernel sum below
        // runs over every sample, so this stays O(n) per evaluation.
        let neighbors = self.nearest_neighbors(x, bandwidth)?;
        if neighbors.is_empty() {
            return Err(DensityError::EmptyNeighborhood);
        }
        let radius = neighbors.radius;
        let d = x.length() as i32;
        Ok((1. / (self.n_samples() * radius.powi(d)))
            * self
                .samples()
                .as_ref()
                .iter()
                .map(|xi| {
                    Gaussian.evaluate(squared_euclidean(x.as_ref(), xi.as_ref()).sqrt() / radius)
                        / crate::consts::SQRT_2PI.powi(d - 1)
                })
                .sum::<f64>())
    }
}

/// Computes the kernel density estimate for a given point `x`
/// using the samples provided and a specified kernel.
///
/// The optimal `k` is computed using [Orava's](https://www.sav.sk/journals/uploads/0127102604orava.pdf)
/// formula when `bandwidth` is `None`.
///
/// # Performance
///
/// This builds a k-d tree over `samples` on every call. To evaluate the density
/// at more than a couple of points, build a
/// [`DensityEstimator`] once and call
/// [`DensityEstimator::kde_pdf`] instead.
///
/// # Examples
///
/// ```
/// use statrs::density::kde::kde_pdf;
///
/// let samples: Vec<[f64; 1]> = vec![[-1.0], [0.0], [1.0]];
/// let density = kde_pdf(&[0.0], &samples, Some(1.0)).unwrap();
/// assert!(density > 0.0);
/// ```
pub fn kde_pdf<S, X>(x: &X, samples: &S, bandwidth: Option<f64>) -> Result<f64, DensityError>
where
    S: AsRef<[X]> + Container,
    X: AsRef<[f64]> + Container + PartialEq,
{
    DensityEstimator::new(samples)?.kde_pdf(x, bandwidth)
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
    fn test_kde_pdf() {
        let law = Normal::new(0., 1.).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let gaussian = crate::function::kernel::Gaussian;
        let samples_1d = (0..100000)
            .map(|_| Vector1::new(law.sample(&mut rng)))
            .collect::<Vec<_>>();
        let x = Vector1::new(0.);
        let kde_density_with_bandwidth = kde_pdf(&x, &samples_1d, Some(0.05));
        let kde_density = kde_pdf(&x, &samples_1d, None);
        let reference_value = gaussian.evaluate(0.);
        assert!(kde_density.is_ok());
        assert!(kde_density_with_bandwidth.is_ok());
        assert!((kde_density.unwrap() - reference_value).abs() < 2e-2);
        assert!((kde_density_with_bandwidth.unwrap() - reference_value).abs() < 3e-2);

        let samples_2d = (0..100000)
            .map(|_| Vector2::new(law.sample(&mut rng), law.sample(&mut rng)))
            .collect::<Vec<_>>();

        let x = Vector2::new(0., 0.);
        let kde_density_with_bandwidth = kde_pdf(&x, &samples_2d, Some(0.05));
        let kde_density = kde_pdf(&x, &samples_2d, None);
        let reference_value = 1. / (2. * PI) as f64;
        assert!(kde_density.is_ok());
        assert!(kde_density_with_bandwidth.is_ok());
        assert!((kde_density.unwrap() - reference_value).abs() < 2e-2);
        assert!((kde_density_with_bandwidth.unwrap() - reference_value).abs() < 3e-2);
    }
}
