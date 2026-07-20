//! Nearest-neighbor [density estimation](https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation)
//! for samples in R^d, backed by a k-d tree for neighbor search.
//!
//! Two estimators are provided, differing in how they turn a neighborhood
//! into a density:
//! - [`knn::knn_pdf`] uses the distance to the `k`-th nearest neighbor directly.
//! - [`kde::kde_pdf`] additionally weights every sample in that neighborhood
//!   by a Gaussian kernel, using the `k`-th neighbor's distance as a local
//!   bandwidth.
//!
//! Both accept an explicit `bandwidth` (a fixed search radius), or fall back
//! to a `k` chosen by [Orava's formula](https://www.sav.sk/journals/uploads/0127102604orava.pdf)
//! when `bandwidth` is `None`.

pub mod kde;
pub mod knn;
use kdtree::{ErrorKind, KdTree, distance::squared_euclidean};
use thiserror::Error;

/// Errors that can occur when estimating a density from a sample.
#[derive(Error, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DensityError {
    /// The k-d tree backing the nearest-neighbor search could not be built or queried.
    #[error("K-d tree error: {0}")]
    KdTree(ErrorKind),

    /// The sample provided was empty, so no density can be estimated.
    #[error("No samples provided")]
    EmptySample,

    /// No sample points fell within the queried neighborhood.
    #[error("No neighbors found")]
    EmptyNeighborhood,
}

impl From<ErrorKind> for DensityError {
    fn from(err: ErrorKind) -> Self {
        DensityError::KdTree(err)
    }
}

fn orava_optimal_k(n_samples: f64) -> f64 {
    // Adapted from K-nearest neighbour kernel density estimation, the choice of optimal k; Jan Orava 2012
    (0.587 * n_samples.powf(4.0 / 5.0)).round().max(1.)
}

/// Handles variable/point types for which nearest neighbors can be computed.
pub trait Container: Clone {
    type Elem;
    fn length(&self) -> usize;
}

macro_rules! impl_container {
    ($($t:ty),*) => {
        $(
            impl<T: Clone> Container for $t {
                type Elem = T;
                fn length(&self) -> usize {
                    self.len()
                }

            }
        )*
    };
}
impl_container!(
    [T; 1],
    [T; 2],
    [T; 3],
    Vec<T>,
    nalgebra::Vector1<T>,
    nalgebra::Vector2<T>,
    nalgebra::Vector3<T>,
    nalgebra::Vector4<T>,
    nalgebra::Vector5<T>,
    nalgebra::Vector6<T>
);
pub type NearestNeighbors = (Vec<f64>, f64);

pub(crate) fn nearest_neighbors<S, X>(
    x: &X,
    samples: &S,
    bandwidth: Option<f64>,
) -> Result<NearestNeighbors, DensityError>
where
    S: AsRef<[X]> + Container,
    X: AsRef<[f64]> + Container + PartialEq,
{
    if samples.length() == 0 {
        return Err(DensityError::EmptySample);
    }
    let n_samples = samples.length() as f64;
    let d = x.length();
    let mut tree = KdTree::with_capacity(d, 2usize.pow(n_samples.log2() as u32));
    for (position, sample) in samples.as_ref().iter().enumerate() {
        tree.add(sample.clone(), position)?;
    }
    if let Some(bandwidth) = bandwidth {
        let neighbors = tree.within(x.as_ref(), bandwidth, &squared_euclidean)?;
        let k = neighbors.len() as f64;
        Ok((neighbors.into_iter().map(|r| r.0).collect(), k))
    } else {
        let k = orava_optimal_k(n_samples);
        Ok((
            tree.nearest(x.as_ref(), k as usize, &squared_euclidean)?
                .into_iter()
                .map(|r| r.0)
                .collect(),
            k,
        ))
    }
}
#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;

    #[test]
    fn test_vec_container() {
        let v1 = vec![1.0, 2.0, 3.0];
        assert_eq!(v1.length(), 3);
        let v2 = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(v2.length(), 3);
    }
}
