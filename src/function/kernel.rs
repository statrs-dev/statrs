//! Kernel functions for use in kernel-based methods such as
//! kernel density estimation (KDE), local regression, and smoothing.
//!
//! Each kernel maps a normalized distance `x` (often `|x_i - x_0| / h`)
//! to a nonnegative weight. Kernels with bounded support return zero
//! outside a finite interval (e.g., `[-1, 1]`).
//!
//! # Implemented Kernels
//! | Kernel | Formula | Support |
//! |---------|----------|----------|
//! | Gaussian | `exp(-0.5 * x²) / √(2π)` | (-∞, ∞) |
//! | Epanechnikov | `0.75 * (1 - x²)` | [-1, 1] |
//! | Triangular | `1 - |x|` | [-1, 1] |
//! | Tricube | `(1 - |x|³)³` | [-1, 1] |
//! | Quartic (biweight) | `(15/16) * (1 - x²)²` | [-1, 1] |
//! | Uniform | `0.5` | [-1, 1] |
//! | Cosine | `(π/4) * cos(πx/2)` | [-1, 1] |
//! | Logistic | `1 / (2 + exp(x) + exp(-x))` | (-∞, ∞) |
//! | Sigmoid | `(2 / π) * (1 / (exp(πx) + exp(-πx)))` | (-∞, ∞) |
//!
//! # Example
//! ```
//! use statrs::function::kernel::{Kernel, Gaussian, Epanechnikov};
//!
//! let g = Gaussian;
//! let e = Epanechnikov;
//! assert!((g.evaluate(0.0) - 0.39894).abs() < 1e-5);
//! assert!((e.evaluate(0.0) - 0.75).abs() < 1e-12);
//! ```

use std::f64::consts::{FRAC_PI_2, PI};

/// Common interface for kernel functions used in KDE and smoothing.
pub trait Kernel {
    /// Evaluate the kernel at normalized distance `x`.
    fn evaluate(&self, x: f64) -> f64;

    /// Evaluate the kernel with bandwidth scaling.
    ///
    /// The result is scaled by `1 / bandwidth` to ensure
    /// that the kernel integrates to 1 after scaling.
    fn evaluate_with_bandwidth(&self, x: f64, bandwidth: f64) -> f64 {
        self.evaluate(x / bandwidth) / bandwidth
    }

    /// Returns the support of the kernel if bounded (e.g. `[-1, 1]`).
    fn support(&self) -> Option<(f64, f64)> {
        None
    }
}

/// Gaussian kernel: (1 / √(2π)) * exp(-0.5 * x²)
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Gaussian;

impl Kernel for Gaussian {
    fn evaluate(&self, x: f64) -> f64 {
        (-(x * x) / 2.0).exp() / (2.0 * PI).sqrt()
    }
}

/// Epanechnikov kernel: ¾(1 - x²) for |x| ≤ 1
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Epanechnikov;

impl Kernel for Epanechnikov {
    fn evaluate(&self, x: f64) -> f64 {
        let a = x.abs();
        if a <= 1.0 { 0.75 * (1.0 - a * a) } else { 0.0 }
    }

    fn support(&self) -> Option<(f64, f64)> {
        Some((-1.0, 1.0))
    }
}

/// Triangular kernel: (1 - |x|) for |x| ≤ 1
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Triangular;

impl Kernel for Triangular {
    fn evaluate(&self, x: f64) -> f64 {
        let a = x.abs();
        if a <= 1.0 { 1.0 - a } else { 0.0 }
    }

    fn support(&self) -> Option<(f64, f64)> {
        Some((-1.0, 1.0))
    }
}

/// Tricube kernel: (1 - |x|³)³ for |x| ≤ 1
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Tricube;

impl Kernel for Tricube {
    fn evaluate(&self, x: f64) -> f64 {
        let a = x.abs();
        if a <= 1.0 {
            let t = 1.0 - a.powi(3);
            t.powi(3)
        } else {
            0.0
        }
    }

    fn support(&self) -> Option<(f64, f64)> {
        Some((-1.0, 1.0))
    }
}

/// Quartic (biweight) kernel: (15/16) * (1 - x²)² for |x| ≤ 1
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Quartic;

impl Kernel for Quartic {
    fn evaluate(&self, x: f64) -> f64 {
        let a = x.abs();
        if a <= 1.0 {
            let t = 1.0 - a * a;
            (15.0 / 16.0) * t * t
        } else {
            0.0
        }
    }

    fn support(&self) -> Option<(f64, f64)> {
        Some((-1.0, 1.0))
    }
}

/// Uniform (rectangular) kernel: 0.5 for |x| ≤ 1, 0 otherwise
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Uniform;

impl Kernel for Uniform {
    fn evaluate(&self, x: f64) -> f64 {
        if x.abs() <= 1.0 { 0.5 } else { 0.0 }
    }

    fn support(&self) -> Option<(f64, f64)> {
        Some((-1.0, 1.0))
    }
}

/// Cosine kernel: (π/4) * cos(πx/2) for |x| ≤ 1, 0 otherwise
///
/// This kernel integrates to 1 over [-1, 1].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Cosine;

impl Kernel for Cosine {
    fn evaluate(&self, x: f64) -> f64 {
        let a = x.abs();
        if a <= 1.0 {
            (PI / 4.0) * (FRAC_PI_2 * a).cos()
        } else {
            0.0
        }
    }

    fn support(&self) -> Option<(f64, f64)> {
        Some((-1.0, 1.0))
    }
}

/// Logistic kernel: 1 / (2 + exp(x) + exp(-x))
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Logistic;

impl Kernel for Logistic {
    fn evaluate(&self, x: f64) -> f64 {
        1.0 / (2.0 + x.exp() + (-x).exp())
    }
}

/// Sigmoid kernel: (1 / (π * cosh(πx))) ≈ (2 / π) * (1 / (exp(πx) + exp(-πx)))
///
/// Note: Integrates to 1/π over (-∞, ∞).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Sigmoid;

impl Kernel for Sigmoid {
    fn evaluate(&self, x: f64) -> f64 {
        (2.0 / PI) * 1.0 / ((PI * x).exp() + (-PI * x).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prec::assert_abs_diff_eq;

    fn integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
        let h = (b - a) / n as f64;
        let mut sum = 0.0;
        for i in 0..n {
            let x0 = a + i as f64 * h;
            let x1 = a + (i + 1) as f64 * h;
            sum += 0.5 * (f(x0) + f(x1)) * h;
        }
        sum
    }

    #[test]
    fn uniform_behavior() {
        let k = Uniform;
        assert_eq!(k.evaluate(0.0), 0.5);
        assert_eq!(k.evaluate(0.8), 0.5);
        assert_eq!(k.evaluate(1.0), 0.5);
        assert_eq!(k.evaluate(1.01), 0.0);
        assert_eq!(k.evaluate(-1.01), 0.0);
        // symmetry
        assert_abs_diff_eq!(k.evaluate(0.5), k.evaluate(-0.5), epsilon = 1e-15);
        // normalization check
        let integral = integrate(|u| k.evaluate(u), -1.0, 1.0, 10_000);
        assert_abs_diff_eq!(integral, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn cosine_behavior() {
        let k = Cosine;
        assert_abs_diff_eq!(k.evaluate(0.0), PI / 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(k.evaluate(1.0), 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(k.evaluate(-1.0), 0.0, epsilon = 1e-15);
        assert!(k.evaluate(0.25) > k.evaluate(0.75));
        assert_abs_diff_eq!(k.evaluate(0.3), k.evaluate(-0.3), epsilon = 1e-15);
        let integral = integrate(|u| k.evaluate(u), -1.0, 1.0, 10_000);
        assert_abs_diff_eq!(integral, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn logistic_behavior() {
        let k = Logistic;
        assert_abs_diff_eq!(k.evaluate(0.0), 0.25, epsilon = 1e-12);
        assert!(k.evaluate(0.0) > k.evaluate(2.0));
        assert_abs_diff_eq!(k.evaluate(0.5), k.evaluate(-0.5), epsilon = 1e-15);
        // integral over a wide range should approximate 1.0
        let integral = integrate(|u| k.evaluate(u), -10.0, 10.0, 50_000);
        assert_abs_diff_eq!(integral, 1.0, epsilon = 1e-3);
    }

    #[test]
    fn sigmoid_behavior() {
        let k = Sigmoid;
        assert_abs_diff_eq!(k.evaluate(0.0), 1.0 / PI, epsilon = 1e-12);
        assert!(k.evaluate(0.0) > k.evaluate(2.0));
        assert_abs_diff_eq!(k.evaluate(0.5), k.evaluate(-0.5), epsilon = 1e-15);
        let integral = integrate(|u| k.evaluate(u), -10.0, 10.0, 50_000);
        assert_abs_diff_eq!(integral, 1.0 / PI, epsilon = 1e-3);
    }

    #[test]
    fn tricube_basic_properties() {
        let k = Tricube;
        assert_abs_diff_eq!(k.evaluate(0.5), k.evaluate(-0.5), epsilon = 1e-15);
        assert_eq!(k.evaluate(1.0), 0.0);
        assert_eq!(k.evaluate(-1.0), 0.0);
        assert_eq!(k.evaluate(0.0), 1.0);
        assert!(k.evaluate(0.25) > k.evaluate(0.5));
        assert!(k.evaluate(0.5) > k.evaluate(0.75));
    }

    #[test]
    fn epanechnikov_behavior() {
        let k = Epanechnikov;
        assert_abs_diff_eq!(k.evaluate(0.3), k.evaluate(-0.3), epsilon = 1e-15);
        assert_eq!(k.evaluate(1.0), 0.0);
        assert_eq!(k.evaluate(-1.0), 0.0);
        assert!(k.evaluate(0.0) > k.evaluate(0.8));
        assert!(k.evaluate(0.5) > 0.0);
        assert!(k.evaluate(0.5) < k.evaluate(0.0));
    }

    #[test]
    fn quartic_behavior() {
        let k = Quartic;
        assert_abs_diff_eq!(k.evaluate(0.0), 15.0 / 16.0, epsilon = 1e-12);
        assert_eq!(k.evaluate(1.0), 0.0);
        assert_eq!(k.evaluate(-1.0), 0.0);
        assert_abs_diff_eq!(k.evaluate(0.3), k.evaluate(-0.3), epsilon = 1e-15);
        assert!(k.evaluate(0.25) > k.evaluate(0.75));
        assert_eq!(k.evaluate(1.1), 0.0);
    }

    #[test]
    fn triangular_behavior() {
        let k = Triangular;
        assert_eq!(k.evaluate(0.0), 1.0);
        assert_eq!(k.evaluate(1.0), 0.0);
        assert_eq!(k.evaluate(-1.0), 0.0);
        assert_abs_diff_eq!(k.evaluate(0.3), k.evaluate(-0.3), epsilon = 1e-15);
        assert!(k.evaluate(0.25) > k.evaluate(0.75));
        assert_eq!(k.evaluate(1.2), 0.0);
    }

    #[test]
    fn gaussian_behavior() {
        let k = Gaussian;
        assert_abs_diff_eq!(k.evaluate(0.5), k.evaluate(-0.5), epsilon = 1e-15);
        assert!(k.evaluate(0.0) > k.evaluate(1.0));
        assert!(k.evaluate(2.0) < 0.2);
        for u in [-3.0, -1.0, 0.0, 1.0, 3.0] {
            assert!(k.evaluate(u) >= 0.0);
        }
    }

    #[test]
    fn kernel_trait_usage() {
        struct Linear;
        impl Kernel for Linear {
            fn evaluate(&self, x: f64) -> f64 {
                (1.0 - x.abs()).max(0.0)
            }
        }

        let lin = Linear;
        assert_eq!(lin.evaluate(0.0), 1.0);
        assert_eq!(lin.evaluate(1.5), 0.0);

        let t = Tricube;
        let g = Gaussian;
        assert_eq!(t.evaluate(0.0), 1.0);
        assert!(g.evaluate(1.0) < 1.0);
        assert_abs_diff_eq!(t.evaluate(0.5), Tricube.evaluate(0.5), epsilon = 1e-15);
    }

    #[test]
    fn bandwidth_scaling_equivalence() {
        let g = Gaussian;
        let scaled = g.evaluate_with_bandwidth(0.5, 2.0);
        let manual = g.evaluate(0.25) / 2.0;
        assert_abs_diff_eq!(scaled, manual, epsilon = 1e-14);
    }

    #[test]
    fn monotonicity_samples() {
        let kernels: [&dyn Kernel; 4] = [&Tricube, &Epanechnikov, &Quartic, &Triangular];
        let samples = [0.0_f64, 0.25, 0.5, 0.75, 0.99];
        for k in kernels {
            let mut prev = k.evaluate(0.0);
            for &u in &samples[1..] {
                let cur = k.evaluate(u);
                assert!(
                    cur <= prev + 1e-12,
                    "kernel not nonincreasing at u={}, prev={}, cur={}",
                    u,
                    prev,
                    cur
                );
                prev = cur;
            }
        }
    }

    #[test]
    fn integrate_tricube_to_expected() {
        let k = Tricube;
        let integral = integrate(|u| k.evaluate(u), -1.0, 1.0, 10_000);
        let expected = 81.0 / 70.0; // ≈ 1.1571
        assert!(
            (integral - expected).abs() < 1e-3,
            "Tricube integral ≈ {}, expected ≈ {}",
            integral,
            expected
        );
    }

    #[test]
    fn integrate_epanechnikov_to_one() {
        let k = Epanechnikov;
        let integral = integrate(|u| k.evaluate(u), -1.0, 1.0, 10_000);
        assert!(
            (integral - 1.0).abs() < 1e-3,
            "Epanechnikov integral ≈ {}",
            integral
        );
    }

    #[test]
    fn integrate_quartic_to_one() {
        let k = Quartic;
        let integral = integrate(|u| k.evaluate(u), -1.0, 1.0, 10_000);
        assert!(
            (integral - 1.0).abs() < 1e-3,
            "Quartic integral ≈ {}",
            integral
        );
    }

    #[test]
    fn integrate_triangular_to_one() {
        let k = Triangular;
        let integral = integrate(|u| k.evaluate(u), -1.0, 1.0, 10_000);
        assert!(
            (integral - 1.0).abs() < 1e-3,
            "Triangular integral ≈ {}",
            integral
        );
    }
}
