use core::f64::consts::PI;

use kdtree::{distance::squared_euclidean, ErrorKind, KdTree};
use thiserror::Error;

use crate::function::gamma::gamma;

#[derive(Error, Debug)]
pub enum DensityError {
    /// Error when the k-d tree cannot be built or queried.
    #[error(transparent)]
    KdTree(#[from] ErrorKind),
    EmptySample,
    EmptyNeighborhood,
}

impl core::fmt::Display for DensityError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            DensityError::KdTree(err) => write!(f, "K-d tree error: {}", err),
            DensityError::EmptySample => write!(f, "No samples provided"),
            DensityError::EmptyNeighborhood => write!(f, "No neighbors found"),
        }
    }
}

fn orava_optimal_k(n_samples: f64) -> f64 {
    // Adapted from K-nearest neighbour kernel density estimation, the choice of optimal k; Jan Orava 2012
    (0.587 * n_samples.powf(4.0 / 5.0)).round().max(1.)
}

/// Computes the k-nearest neighbor density estimate for a given point `x`
/// using the samples provided.
///
/// The optimal `k` is computed using Orava's formula.
///
/// Returns `None` when `samples` is empty.
pub fn knn_pdf(x: f64, samples: &[f64]) -> Result<f64, DensityError> {
    if samples.is_empty() {
        return Err(DensityError::EmptySample);
    }
    let n_samples = samples.len() as f64;
    let k = orava_optimal_k(n_samples);
    let mut tree = KdTree::with_capacity(1, n_samples.log2() as usize);
    for (position, sample) in samples.iter().enumerate() {
        tree.add([*sample], position)?;
    }
    let neighbors = tree.nearest(&[x], k as usize, &squared_euclidean)?;
    if neighbors.is_empty() {
        Err(DensityError::EmptyNeighborhood)
    } else {
        let radius = neighbors.last().unwrap().0.sqrt();
        let d = 1.;
        Ok((k / n_samples) * (gamma(d / 2. + 1.) / (PI.powf(d / 2.) * radius.powf(d))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::{Continuous, Normal};
    use rand::distributions::Distribution;

    #[test]
    fn test_knn_pdf() {
        let law = Normal::new(0., 1.).unwrap();
        let mut rng = rand::thread_rng();
        let samples = (0..100000)
            .map(|_| law.sample(&mut rng))
            .collect::<Vec<_>>();
        let x = 0.0;
        let knn_density = knn_pdf(x, &samples);
        println!("Knn: {:?}", knn_density);
        println!("Pdf: {:?}", law.pdf(x));
    }

    #[test]
    fn test_knn_pdf_empty_samples() {
        let samples: Vec<f64> = vec![];
        let x = 3.0;
        let result = knn_pdf(x, &samples);
        assert!(matches!(result, Err(DensityError::EmptySample)));
    }
}
