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
//!
//! The free functions build the k-d tree on every call. To evaluate a density at
//! more than a couple of points, build a [`DensityEstimator`] once and query it
//! repeatedly - measured at ~7x over a 200-point grid.

pub mod kde;
pub mod knn;
use alloc::vec::Vec;
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
/// Outcome of a neighborhood query around a point.
#[derive(Clone, Debug, PartialEq)]
pub struct NearestNeighbors {
    /// Squared distances from the query point to each neighbor found.
    ///
    /// The order is **unspecified**. `KdTree::within` returned these sorted in
    /// `kdtree` 0.7 but returns raw heap order from 0.8 onward, so the last
    /// element is not the furthest neighbor; use [`Self::radius`] instead of
    /// indexing this.
    pub squared_distances: Vec<f64>,

    /// How many neighbors were found, as an `f64`.
    pub k: f64,

    /// Distance - not squared - to the furthest neighbor, i.e. the radius of
    /// the neighborhood actually used by the estimators.
    pub radius: f64,
}

impl NearestNeighbors {
    fn new(squared_distances: Vec<f64>, k: f64) -> Self {
        // Computed as a maximum rather than by taking the last element, so that
        // it does not depend on the ordering of `squared_distances`.
        let radius = squared_distances
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            .sqrt();
        Self {
            squared_distances,
            k,
            radius,
        }
    }

    /// Whether the neighborhood is empty.
    pub fn is_empty(&self) -> bool {
        self.squared_distances.is_empty()
    }
}

/// Leaf bucket size for the backing k-d tree.
///
/// This is a deliberate trade: measured at `n = 1e5`, `d = 1`, a capacity of
/// `~n` builds in 1.3 ms against 19.5 ms for the crate's default of 16, while
/// queries are only ~1.1x slower - the query is dominated by materialising the
/// result vector, not by traversal. For query-heavy use of
/// [`DensityEstimator`], where the build is amortised away, a small constant is
/// worth roughly 15% of query time instead.
fn leaf_capacity(n_samples: usize) -> usize {
    2usize.pow((n_samples as f64).log2() as u32)
}

/// A reusable nearest-neighbor index over a fixed sample set.
///
/// The free functions [`knn::knn_pdf`] and [`kde::kde_pdf`] rebuild the backing
/// k-d tree on every call, which is `O(n log n)` per evaluation - about 59% of a
/// single call at `n = 1e5`. Building the index once and querying it repeatedly
/// is ~5x faster when evaluating a density over more than a handful of points:
///
/// ```
/// use statrs::density::DensityEstimator;
///
/// let samples: Vec<[f64; 1]> = vec![[-1.0], [-0.5], [0.0], [0.25], [0.5], [1.0]];
/// let estimator = DensityEstimator::new(&samples).unwrap();
///
/// // one tree, many evaluations
/// let densities: Vec<f64> = (0..5)
///     .map(|i| -1.0 + 0.5 * i as f64)
///     .map(|g| estimator.knn_pdf(&[g], Some(1.0)).unwrap())
///     .collect();
/// assert!(densities.iter().all(|d| *d > 0.0));
/// ```
#[derive(Clone, Debug)]
pub struct DensityEstimator<'a, S, X: AsRef<[f64]> + PartialEq> {
    samples: &'a S,
    tree: KdTree<f64, usize, X>,
}

impl<'a, S, X> DensityEstimator<'a, S, X>
where
    S: AsRef<[X]> + Container,
    X: AsRef<[f64]> + Container + PartialEq,
{
    /// Builds the index over `samples`.
    ///
    /// # Errors
    ///
    /// Returns [`DensityError::EmptySample`] if `samples` is empty, or
    /// [`DensityError::KdTree`] if a sample's dimension disagrees with the
    /// first one's.
    pub fn new(samples: &'a S) -> Result<Self, DensityError> {
        let points = samples.as_ref();
        let Some(first) = points.first() else {
            return Err(DensityError::EmptySample);
        };
        // Dimensionality is taken from the samples rather than from a query
        // point, so a mismatched query is rejected by the tree instead of
        // silently building a tree of the wrong shape.
        let mut tree = KdTree::with_capacity(first.length(), leaf_capacity(points.len()));
        for (position, sample) in points.iter().enumerate() {
            tree.add(sample.clone(), position)?;
        }
        Ok(Self { samples, tree })
    }

    /// The samples this index was built over.
    pub fn samples(&self) -> &'a S {
        self.samples
    }

    pub(crate) fn n_samples(&self) -> f64 {
        self.samples.as_ref().len() as f64
    }

    /// Finds the neighborhood of `x`: within a fixed squared radius when
    /// `bandwidth` is `Some`, otherwise the `k` nearest neighbors for a `k`
    /// chosen by Orava's formula.
    ///
    /// # Errors
    ///
    /// Returns [`DensityError::KdTree`] if `x`'s dimension disagrees with the
    /// samples'.
    pub fn nearest_neighbors(
        &self,
        x: &X,
        bandwidth: Option<f64>,
    ) -> Result<NearestNeighbors, DensityError> {
        // Both queries use `squared_euclidean`, so `bandwidth` is a squared
        // radius and every returned distance is squared.
        if let Some(bandwidth) = bandwidth {
            let neighbors = self
                .tree
                .within(x.as_ref(), bandwidth, &squared_euclidean)?;
            let k = neighbors.len() as f64;
            Ok(NearestNeighbors::new(
                neighbors.into_iter().map(|r| r.0).collect(),
                k,
            ))
        } else {
            let k = orava_optimal_k(self.n_samples());
            let neighbors = self
                .tree
                .nearest(x.as_ref(), k as usize, &squared_euclidean)?;
            Ok(NearestNeighbors::new(
                neighbors.into_iter().map(|r| r.0).collect(),
                k,
            ))
        }
    }
}
#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;

    /// The neighborhood radius must be the distance to the *furthest* neighbor
    /// no matter what order the backing tree returns results in.
    ///
    /// `knn_pdf`/`kde_pdf` used to read it as `distances.last()`, which was only
    /// correct because `kdtree` 0.7's `within` happened to sort; 0.8 returns raw
    /// heap order, which silently scaled both densities by the wrong volume.
    /// Hand-placed samples make the expected values exact, unlike the
    /// Monte Carlo tests that could not reliably detect this.
    #[test]
    fn test_nearest_neighbors_radius_is_the_maximum() {
        let samples: Vec<[f64; 1]> = vec![[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]];
        // `bandwidth` is a *squared* radius, so 4.0 admits everything within 2.0
        let estimator = DensityEstimator::new(&samples).unwrap();
        let nn = estimator.nearest_neighbors(&[0.0], Some(4.0)).unwrap();
        assert_eq!(nn.k, 5.0, "expected the 5 samples within distance 2");
        assert_eq!(nn.radius, 2.0);
        // and it agrees with an order-independent maximum by construction
        let max = nn
            .squared_distances
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            .sqrt();
        assert_eq!(nn.radius, max);

        // the `nearest` path reports a radius too
        let nn = estimator.nearest_neighbors(&[0.0], None).unwrap();
        assert!(nn.radius > 0.0 && nn.radius.is_finite());
    }

    /// With the radius pinned, `knn_pdf` is exactly `(k / n) / V_d(r)`:
    /// `(5 / 7) / (2 * 2) = 5 / 28` in one dimension.
    #[test]
    fn test_knn_pdf_deterministic() {
        let samples: Vec<[f64; 1]> = vec![[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]];
        let got = crate::density::knn::knn_pdf(&[0.0], &samples, Some(4.0)).unwrap();
        // the tolerance is set by `gamma(d / 2 + 1)` in the ball-volume factor,
        // not by anything in this module
        crate::prec::assert_relative_eq!(got, 5.0 / 28.0, epsilon = 0.0, max_relative = 1e-14);
    }

    /// The prepared estimator must be bit-identical to the free functions - it
    /// is the same code path with the tree hoisted out of the loop, so any
    /// difference would mean the hoist changed the maths.
    #[test]
    fn test_estimator_matches_free_functions() {
        let samples: Vec<[f64; 1]> = (0..500).map(|i| [(i as f64 * 0.37).sin() * 2.0]).collect();
        let estimator = DensityEstimator::new(&samples).unwrap();
        for i in 0..40 {
            let g = [-2.0 + 4.0 * i as f64 / 40.0];
            for bw in [Some(0.05), Some(1.0), None] {
                assert_eq!(
                    estimator.knn_pdf(&g, bw).ok(),
                    crate::density::knn::knn_pdf(&g, &samples, bw).ok(),
                    "knn_pdf mismatch at {g:?} bandwidth {bw:?}"
                );
                assert_eq!(
                    estimator.kde_pdf(&g, bw).ok(),
                    crate::density::kde::kde_pdf(&g, &samples, bw).ok(),
                    "kde_pdf mismatch at {g:?} bandwidth {bw:?}"
                );
            }
        }
    }

    #[test]
    fn test_estimator_empty_sample() {
        let empty: Vec<[f64; 1]> = vec![];
        assert_eq!(
            DensityEstimator::new(&empty).unwrap_err(),
            DensityError::EmptySample
        );
    }

    /// A query point of the wrong dimension is rejected by the tree rather than
    /// silently answered, because the tree's dimension comes from the samples.
    #[test]
    fn test_estimator_rejects_dimension_mismatch() {
        let samples: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let estimator = DensityEstimator::new(&samples).unwrap();
        assert!(estimator.nearest_neighbors(&vec![0.0], Some(1.0)).is_err());
        assert!(estimator.knn_pdf(&vec![0.0, 0.0], Some(4.0)).is_ok());
    }

    #[test]
    fn test_vec_container() {
        let v1 = vec![1.0, 2.0, 3.0];
        assert_eq!(v1.length(), 3);
        let v2 = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(v2.length(), 3);
    }
}
