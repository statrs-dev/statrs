use core::f64::consts::{PI, SQRT_2};

use kdtree::distance::squared_euclidean;

use crate::density::{nearest_neighbors, Container, DensityError};

/// The implemented [kernel functions][source]
///
/// source: https://en.wikipedia.org/wiki/Kernel_(statistics)
#[derive(Debug, Default, Clone, Copy)]
pub enum Kernel {
    #[default]
    Epanechnikov,
    Gaussian {
        dim: i32,
    },
    Uniform,
    Triangular,
    Biweigth,
    Triweight,
    Tricube,
    Cosine,
    Logistic,
    Sigmoid,
    Silverman,
}

impl Kernel {
    pub fn evaluate(&self, u: f64) -> f64 {
        match self {
            Self::Epanechnikov => {
                if u.abs() >= 1. {
                    0.0
                } else {
                    0.75 * (1. - u.powi(2))
                }
            }
            Self::Gaussian { dim } => (-0.5 * u.powi(2)).exp() / crate::consts::SQRT_2PI.powi(*dim),
            Self::Uniform => {
                if u.abs() > 1. {
                    0.0
                } else {
                    0.5
                }
            }
            Self::Triangular => {
                let abs_u = u.abs();
                if abs_u >= 1. {
                    0.0
                } else {
                    1. - abs_u
                }
            }
            Self::Biweigth => {
                if u.abs() >= 1. {
                    0.0
                } else {
                    (15. / 16.) * (1. - u.powi(2)).powi(2)
                }
            }
            Self::Triweight => {
                if u.abs() >= 1. {
                    0.0
                } else {
                    (35. / 32.) * (1. - u.powi(2)).powi(3)
                }
            }
            Self::Tricube => {
                let abs_u = u.abs();
                if abs_u >= 1. {
                    0.0
                } else {
                    (70. / 81.) * (1. - abs_u.powi(3)).powi(3)
                }
            }
            Self::Cosine => {
                if u.abs() >= 1. {
                    0.0
                } else {
                    (PI / 4.) * ((PI / 2.) * u).cos()
                }
            }
            Self::Logistic => 0.5 / (1. + u.cosh()),
            Self::Sigmoid => 1. / (PI * u.cosh()),
            Self::Silverman => {
                let abs_u_over_sqrt2 = u.abs() / SQRT_2;
                0.5 * (-abs_u_over_sqrt2).exp() * (PI / 4. + abs_u_over_sqrt2).sin()
            }
        }
    }
}

/// Computes the kernel density estimate for a given point `x`
/// using the samples provided and a specified kernel.
///
/// The optimal `k` is computed using [Orava's][orava] formula when `bandwidth` is `None`.
///
/// orava: K-nearest neighbour kernel density estimation, the choice of optimal k; Jan Orava 2012.
pub fn kde_pdf<S, X>(x: X, samples: &S, bandwidth: Option<f64>) -> Result<f64, DensityError>
where
    S: AsRef<[X]> + Container,
    X: AsRef<[f64]> + Container + PartialEq,
{
    let n_samples = samples.length() as f64;
    let d = x.length();
    let neighbors = nearest_neighbors(&x, samples, bandwidth)?.0;
    if neighbors.is_empty() {
        Err(DensityError::EmptyNeighborhood)
    } else {
        let radius = neighbors.last().unwrap().sqrt();
        let gaussian_kernel = Kernel::Gaussian { dim: d as i32 };
        Ok((1. / (n_samples * radius.powi(d as i32)))
            * samples
                .as_ref()
                .iter()
                .map(|xi| {
                    gaussian_kernel
                        .evaluate(squared_euclidean(x.as_ref(), xi.as_ref()).sqrt() / radius)
                })
                .sum::<f64>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::Normal;
    use nalgebra::Vector2;
    use rand::distributions::Distribution;

    #[test]
    fn test_kernel_1d() {
        let kernel = Kernel::Epanechnikov;
        assert_eq!(kernel.evaluate(0.5), 0.75 * 0.75);
        assert_eq!(kernel.evaluate(1.5), 0.0);

        let kernel = Kernel::Gaussian { dim: 1 };
        assert!((kernel.evaluate(0.0) - (1. / (SQRT_2 * PI.sqrt()))).abs() < 1e-10);
    }

    #[test]
    fn test_kde_pdf() {
        let law = Normal::new(0., 1.).unwrap();
        let mut rng = rand::thread_rng();
        let samples = (0..100000)
            .map(|_| Vector2::new(law.sample(&mut rng), law.sample(&mut rng)))
            // .map(|_| Vector1::new(law.sample(&mut rng)))
            .collect::<Vec<_>>();
        let x = Vector2::new(0.0, 0.0);
        // let x = Vector1::new(0.0);
        let kde_density_with_bandwidth = kde_pdf(x, &samples, Some(0.05));
        let kde_density = kde_pdf(x, &samples, None);
        println!("Kde: {:?}", kde_density);
        println!("Kde with bandwidth: {:?}", kde_density_with_bandwidth);
        // println!("Pdf: {:?}", law.pdf(x[0]));
        assert!(kde_density.is_ok());
    }
}
