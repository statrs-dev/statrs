use super::Container;
use crate::{
    density::{DensityError, nearest_neighbors},
    function::gamma::gamma,
};
use core::f64::consts::PI;

/// Computes the `k`-nearest neighbor density estimate for a given point `x`
/// using the samples provided.
///
/// The optimal `k` is computed using [Orava's][orava] formula when `bandwidth` is `None`.
///
/// orava: K-nearest neighbour kernel density estimation, the choice of optimal k; Jan Orava 2012.
pub fn knn_pdf<X, S>(x: &X, samples: &S, bandwidth: Option<f64>) -> Result<f64, DensityError>
where
    S: AsRef<[X]> + Container,
    X: AsRef<[f64]> + Container + PartialEq,
{
    let n_samples = samples.length() as f64;
    let (neighbors, k) = nearest_neighbors(x, samples, bandwidth)?;
    if neighbors.is_empty() {
        Err(DensityError::EmptyNeighborhood)
    } else {
        let radius = neighbors.last().unwrap().sqrt();
        let d = x.length() as f64;
        Ok((k / n_samples) * (gamma(d / 2. + 1.) / (PI.powf(d / 2.) * radius.powf(d))))
    }
}

#[cfg(test)]
mod tests {
    use core::f32::consts::PI;

    use super::*;
    use crate::distribution::Normal;
    use crate::function::kernel::Kernel;
    use nalgebra::{Vector1, Vector2};
    use rand::distr::Distribution;

    #[test]
    fn test_knn_pdf() {
        let law = Normal::new(0., 1.).unwrap();
        let mut rng = rand::rng();
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
