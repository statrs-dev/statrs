use core::f64::consts::PI;

use kdtree::{distance::squared_euclidean, KdTree};

use crate::{density::DensityError, function::gamma::gamma};

use super::{orava_optimal_k, Container};

/// Computes the `k`-nearest neighbor density estimate for a given point `x`
/// using the samples provided.
///
/// The optimal `k` is computed using Orava's formula when `bandwidth` is `None`.
pub fn knn_pdf<X, S>(x: X, samples: &S, bandwidth: Option<f64>) -> Result<f64, DensityError>
where
    S: AsRef<[X]> + Container<Elem = X>,
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
    let (neighbors, k) = if let Some(bandwidth) = bandwidth {
        let neighbors = tree.within(x.as_ref(), bandwidth, &squared_euclidean)?;
        let k = neighbors.len() as f64;
        (neighbors, k)
    } else {
        let k = orava_optimal_k(n_samples);
        (tree.nearest(x.as_ref(), k as usize, &squared_euclidean)?, k)
    };
    if neighbors.is_empty() {
        Err(DensityError::EmptyNeighborhood)
    } else {
        let radius = neighbors.last().unwrap().0.sqrt();
        let d = d as f64;
        Ok((k / n_samples) * (gamma(d / 2. + 1.) / (PI.powf(d / 2.) * radius.powf(d))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::Normal;
    use nalgebra::Vector2;
    use rand::distributions::Distribution;

    #[test]
    fn test_knn_pdf() {
        let law = Normal::new(0., 1.).unwrap();
        let mut rng = rand::thread_rng();
        let samples = (0..100000)
            .map(|_| Vector2::new(law.sample(&mut rng), law.sample(&mut rng)))
            // .map(|_| Vector1::new(law.sample(&mut rng)))
            .collect::<Vec<_>>();
        let x = Vector2::new(0.0, 0.0);
        // let x = Vector1::new(0.0);
        let knn_density_with_bandwidth = knn_pdf(x, &samples, Some(0.2));
        let knn_density = knn_pdf(x, &samples, None);
        println!("Knn: {:?}", knn_density);
        println!("Knn with bandwidth: {:?}", knn_density_with_bandwidth);
        // println!("Pdf: {:?}", law.pdf(x));
        assert!(knn_density.is_ok());
    }

    #[test]
    fn test_knn_pdf_empty_samples() {
        let samples: Vec<[f64; 1]> = vec![];
        let x = 3.0;
        let result = knn_pdf([x], &samples, None);
        assert!(matches!(result, Err(DensityError::EmptySample)));
    }
}
