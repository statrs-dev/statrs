pub mod kde;
pub mod knn;
use kdtree::ErrorKind;
use thiserror::Error;

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

/// Handles variable/point types for which nearest neighbors can be computed.
pub trait Container: Clone {
    type Elem;
    fn length(&self) -> usize;
}

macro_rules! impl_container_for_num {
    ($($t:ty),*) => {
        $(
            impl Container for $t {
                type Elem = $t;
                fn length(&self) -> usize {
                    1
                }
            }
        )*
    };
}
impl_container_for_num!(f32, f64);

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

    #[test]
    fn test_num_container() {
        let a: f64 = 3.0;
        let b: f64 = 5.0;
        assert_eq!(a.length(), 1);
        assert_eq!(b.length(), 1);
    }
}
