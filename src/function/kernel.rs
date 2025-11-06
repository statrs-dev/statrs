//! Kernel functions for use in kernel-based methods such as
//! kernel density estimation (KDE), local regression, and smoothing.
//!
//! ## Overview
//!
//! Each kernel maps a normalized distance `x` (often `|x_i - x_0| / h`)
//! to a nonnegative weight. Kernels with bounded support return zero
//! outside a finite interval (e.g., `[-1, 1]`).
//!
//! This module provides both trait-based kernels (for statistical correctness)
//! and enum-based kernel selection (for convenience in local methods like LOESS).
//!
//! ## Statistical vs Weighting Evaluation
//!
//! This module distinguishes between two evaluation modes:
//!
//! - **`evaluate()`**: Normalized for statistical correctness (integrates to 1).  
//!   Used in kernel density estimation and other statistical applications.
//!
//! - **`evaluate_weight()`**: Unnormalized for local weighting (LOESS-style).  
//!   Used in local regression where weights are normalized later by the neighborhood.
//!
//! ### Example: Bisquare Kernel at Origin
//!
//! ```rust
//! use statrs::function::kernel::{Kernel, Bisquare};
//!
//! let bisquare = Bisquare;
//!
//! // Statistical evaluation (normalized): returns 15/16 ≈ 0.9375
//! assert!((bisquare.evaluate(0.0) - 0.9375).abs() < 1e-10);
//!
//! // Weight evaluation (unnormalized): returns 1.0
//! assert_eq!(bisquare.evaluate_weight(0.0), 1.0);
//! ```
//!
//! ## Boundary Behavior
//!
//! For bounded kernels, the boundary handling is consistent:
//! - `evaluate(x)` returns 0 for `|x| >= 1`
//! - The boundary point `x = ±1` is excluded from the support
//! - This ensures `support()` returns `(-1, 1)` as an open interval
//!
//! ## Implemented Kernels
//!
//! | Kernel       | Formula                       | Support   | Efficiency† | R(K)      | μ₂(K)     |
//! |--------------|-------------------------------|-----------|-------------|-----------|-----------|
//! | Gaussian     | `exp(-x²/2) / √(2π)`         | (-∞, ∞)  | 0.9512     | 1/(2√π)  | 1         |
//! | Epanechnikov | `0.75(1 - x²)`               | [-1, 1]  | 1.0000     | 3/5      | 1/5       |
//! | Triangular   | `1 - \|x\|`                  | [-1, 1]  | 0.9859     | 2/3      | 1/6       |
//! | Tricube      | `(70/81)(1 - \|x\|³)³`       | [-1, 1]  | 0.9979     | 175/247  | 35/243    |
//! | Bisquare     | `(15/16)(1 - x²)²`           | [-1, 1]  | 0.9939     | 5/7      | 1/7       |
//! | Uniform      | `1/2`                        | [-1, 1]  | 0.9295     | 1/2      | 1/3       |
//! | Cosine       | `(π/4)cos(πx/2)`             | [-1, 1]  | 0.9995     | π²/16    | 1-8/π²    |
//! | Logistic     | `1/(2 + e^x + e^{-x})`       | (-∞, ∞)  | 0.8878     | 1/6      | π²/3      |
//! | Sigmoid      | `(2/π)/(e^x + e^{-x})`       | (-∞, ∞)  | 0.8424     | 2/π²     | π²/4      |
//!
//! †Efficiency = [R(Epan)/R(K)] × √[μ₂(Epan)/μ₂(K)] from Silverman (1986) Table 3.1.  
//! Values represent relative asymptotic efficiency based on MISE. Epanechnikov (1.0) is optimal.
//!
//! ## Usage
//!
//! ```rust
//! use statrs::function::kernel::{Kernel, KernelType, Gaussian, Tricube};
//!
//! // Direct kernel usage
//! let gaussian = Gaussian;
//! let weight = gaussian.evaluate(0.5);
//!
//! // Enum-based selection (useful for runtime configuration)
//! let kernel = KernelType::Tricube;
//! let weights = kernel.compute_distance_weights(&[0.1, 0.5, 0.9], 1.0);
//! ```
//!
//! ## References
//!
//! - Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*.
//! - Wand, M. P., & Jones, M. C. (1995). *Kernel Smoothing*.

use crate::consts::*;
use core::f64::consts::{FRAC_PI_2, PI};

// ============================================================================
// Helper Functions
// ============================================================================

/// Helper function for computing square root at compile time.
const fn const_sqrt(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }

    // Newton-Raphson: x_{n+1} = (x_n + x/x_n) / 2
    let mut guess = x / 2.0;

    let mut i = 0;
    while i < 20 {
        guess = (guess + x / guess) / 2.0;
        i += 1;
    }
    guess
}

#[inline]
fn validate_bandwidth(bandwidth: f64) {
    if bandwidth <= 0.0 {
        panic!("Bandwidth must be positive, got {}", bandwidth);
    }
    if bandwidth.is_infinite() || bandwidth.is_nan() {
        panic!("Bandwidth must be finite, got {}", bandwidth);
    }
}

#[inline]
fn validate_distance(d: f64) {
    if d.is_nan() {
        panic!("Distance cannot be NaN");
    }
}

// ============================================================================
// Macros for Reducing Redundancy
// ============================================================================

/// Macro to define kernel struct with constants
macro_rules! define_kernel {
    ($name:ident, $variance:ident, $roughness:ident) => {
        #[derive(Debug, Clone, Copy, Default, PartialEq)]
        pub struct $name;

        impl $name {
            pub const VARIANCE: f64 = $variance;
            pub const ROUGHNESS: f64 = $roughness;
            // AMISE Efficiency from Silverman (1986) Table 3.1:
            // Efficiency(K) = [R(Epan)/R(K)] × √[μ₂(Epan)/μ₂(K)]
            // This represents the relative asymptotic efficiency based on MISE.
            // Epanechnikov is optimal with efficiency = 1.0.
            pub const EFFICIENCY: f64 = (ROUGHNESS_EPANECHNIKOV / $roughness)
                * const_sqrt(VARIANCE_EPANECHNIKOV / $variance);
            // Canonical bandwidth factor: √(1/μ₂) from Silverman Table 3.1
            pub const BANDWIDTH_FACTOR: f64 = const_sqrt(1.0 / $variance);
        }
    };
}

/// Macro to implement common Kernel trait methods
macro_rules! impl_kernel_properties {
    () => {
        fn bandwidth_factor(&self) -> f64 {
            Self::BANDWIDTH_FACTOR
        }

        fn variance(&self) -> f64 {
            Self::VARIANCE
        }

        fn efficiency(&self) -> f64 {
            Self::EFFICIENCY
        }

        fn roughness(&self) -> f64 {
            Self::ROUGHNESS
        }

        fn clone_box(&self) -> Box<dyn Kernel> {
            Box::new(*self)
        }
    };
}

// ============================================================================
// Kernel Trait
// ============================================================================

/// Common interface for kernel functions used in KDE and smoothing.
pub trait Kernel: Send + Sync + std::fmt::Debug {
    /// Evaluate the kernel at point `x`.
    ///
    /// Returns the statistically normalized kernel value (integrates to 1).
    /// For bounded kernels, returns 0 outside the support.
    fn evaluate(&self, x: f64) -> f64;

    /// Evaluate the kernel for weighting purposes (LOESS-style).
    ///
    /// Returns unnormalized weights suitable for local regression.
    /// Default implementation delegates to `evaluate()`.
    #[inline(always)]
    fn evaluate_weight(&self, x: f64) -> f64 {
        self.evaluate(x)
    }

    /// Returns the support of the kernel as an open interval.
    ///
    /// - `Some((a, b))` for bounded kernels (typically `(-1, 1)`)
    /// - `None` for unbounded kernels (e.g., Gaussian)
    fn support(&self) -> Option<(f64, f64)> {
        None
    }

    /// Evaluate kernel with explicit bandwidth scaling: `K((x - x₀) / h) / h`
    #[inline]
    fn evaluate_with_bandwidth(&self, x: f64, bandwidth: f64) -> f64 {
        self.evaluate(x / bandwidth) / bandwidth
    }

    /// AMISE-optimal bandwidth factor relative to Epanechnikov.
    fn bandwidth_factor(&self) -> f64;

    /// Second moment (variance): μ₂ = ∫ u² K(u) du
    fn variance(&self) -> f64;

    /// AMISE efficiency relative to Epanechnikov (1.0 = optimal)
    fn efficiency(&self) -> f64;

    /// Roughness: R(K) = ∫ K(u)² du
    fn roughness(&self) -> f64;

    /// Clone the kernel into a boxed trait object
    fn clone_box(&self) -> Box<dyn Kernel>;
}

// ============================================================================
// Epanechnikov Kernel (Special Case - Optimal)
// ============================================================================

/// Epanechnikov kernel: ¾(1 - x²) for |x| < 1
///
/// This is the AMISE-optimal kernel (efficiency = 1.0).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Epanechnikov;

impl Epanechnikov {
    pub const VARIANCE: f64 = VARIANCE_EPANECHNIKOV;
    pub const ROUGHNESS: f64 = ROUGHNESS_EPANECHNIKOV;
    pub const EFFICIENCY: f64 = 1.0;
    // Canonical bandwidth: √(1/μ₂) = √(1/(1/5)) = √5 ≈ 2.236
    pub const BANDWIDTH_FACTOR: f64 = const_sqrt(1.0 / VARIANCE_EPANECHNIKOV);
}

impl Kernel for Epanechnikov {
    #[inline(always)]
    fn evaluate(&self, x: f64) -> f64 {
        let a = x.abs();
        if a >= 1.0 { 0.0 } else { 0.75 * (1.0 - a * a) }
    }

    fn support(&self) -> Option<(f64, f64)> {
        Some((-1.0, 1.0))
    }

    impl_kernel_properties!();
}

// ============================================================================
// Kernel Implementations
// ============================================================================

// Use the macro to define all kernels except Epanechnikov
define_kernel!(Gaussian, VARIANCE_GAUSSIAN, ROUGHNESS_GAUSSIAN);
define_kernel!(Triangular, VARIANCE_TRIANGULAR, ROUGHNESS_TRIANGULAR);
define_kernel!(Tricube, VARIANCE_TRICUBE, ROUGHNESS_TRICUBE);
define_kernel!(Bisquare, VARIANCE_BISQUARE, ROUGHNESS_BISQUARE);
define_kernel!(Uniform, VARIANCE_UNIFORM, ROUGHNESS_UNIFORM);
define_kernel!(Cosine, VARIANCE_COSINE, ROUGHNESS_COSINE);
define_kernel!(Logistic, VARIANCE_LOGISTIC, ROUGHNESS_LOGISTIC);
define_kernel!(Sigmoid, VARIANCE_SIGMOID, ROUGHNESS_SIGMOID);

// Implement Kernel trait for each kernel
impl Kernel for Gaussian {
    #[inline(always)]
    fn evaluate(&self, x: f64) -> f64 {
        (-(x * x) / 2.0).exp() / (2.0 * PI).sqrt()
    }

    #[inline(always)]
    fn evaluate_weight(&self, x: f64) -> f64 {
        (-(x * x) / 2.0).exp()
    }

    impl_kernel_properties!();
}

impl Kernel for Triangular {
    #[inline(always)]
    fn evaluate(&self, x: f64) -> f64 {
        let a = x.abs();
        if a >= 1.0 { 0.0 } else { 1.0 - a }
    }

    fn support(&self) -> Option<(f64, f64)> {
        Some((-1.0, 1.0))
    }

    impl_kernel_properties!();
}

impl Kernel for Tricube {
    #[inline(always)]
    fn evaluate(&self, x: f64) -> f64 {
        let a = x.abs();
        if a >= 1.0 {
            0.0
        } else {
            let u = 1.0 - a * a * a;
            (70.0 / 81.0) * u * u * u
        }
    }

    #[inline(always)]
    fn evaluate_weight(&self, x: f64) -> f64 {
        let a = x.abs();
        if a >= 1.0 {
            0.0
        } else {
            let u = 1.0 - a * a * a;
            u * u * u
        }
    }

    fn support(&self) -> Option<(f64, f64)> {
        Some((-1.0, 1.0))
    }

    impl_kernel_properties!();
}

impl Kernel for Bisquare {
    #[inline(always)]
    fn evaluate(&self, x: f64) -> f64 {
        if x.abs() >= 1.0 {
            0.0
        } else {
            let u = 1.0 - x * x;
            (15.0 / 16.0) * u * u
        }
    }

    #[inline(always)]
    fn evaluate_weight(&self, x: f64) -> f64 {
        if x.abs() >= 1.0 {
            0.0
        } else {
            let u = 1.0 - x * x;
            u * u
        }
    }

    fn support(&self) -> Option<(f64, f64)> {
        Some((-1.0, 1.0))
    }

    impl_kernel_properties!();
}

impl Kernel for Uniform {
    #[inline(always)]
    fn evaluate(&self, x: f64) -> f64 {
        if x.abs() >= 1.0 { 0.0 } else { 0.5 }
    }

    #[inline(always)]
    fn evaluate_weight(&self, x: f64) -> f64 {
        if x.abs() >= 1.0 { 0.0 } else { 1.0 }
    }

    fn support(&self) -> Option<(f64, f64)> {
        Some((-1.0, 1.0))
    }

    impl_kernel_properties!();
}

impl Kernel for Cosine {
    #[inline(always)]
    fn evaluate(&self, x: f64) -> f64 {
        if x.abs() >= 1.0 {
            0.0
        } else {
            (PI / 4.0) * (FRAC_PI_2 * x).cos()
        }
    }

    #[inline(always)]
    fn evaluate_weight(&self, x: f64) -> f64 {
        if x.abs() >= 1.0 {
            0.0
        } else {
            (FRAC_PI_2 * x).cos()
        }
    }

    fn support(&self) -> Option<(f64, f64)> {
        Some((-1.0, 1.0))
    }

    impl_kernel_properties!();
}

impl Kernel for Logistic {
    #[inline(always)]
    fn evaluate(&self, x: f64) -> f64 {
        1.0 / (2.0 + x.exp() + (-x).exp())
    }

    impl_kernel_properties!();
}

impl Kernel for Sigmoid {
    #[inline(always)]
    fn evaluate(&self, x: f64) -> f64 {
        2.0 / (PI * (x.exp() + (-x).exp()))
    }

    impl_kernel_properties!();
}

// ============================================================================
// Custom Kernel
// ============================================================================

/// A custom kernel function with optional metadata.
///
/// Allows users to define their own kernel functions with configurable properties.
#[derive(Clone)]
pub struct CustomKernel<F: Fn(f64) -> f64> {
    func: F,
    variance: f64,
    efficiency: f64,
    bandwidth_factor: f64,
    roughness: f64,
    support: Option<(f64, f64)>,
}

impl<F: Fn(f64) -> f64> CustomKernel<F> {
    /// Creates a new custom kernel from a function.
    ///
    /// Default metadata:
    /// - `variance = 1.0`
    /// - `efficiency = 1.0`
    /// - `bandwidth_factor = 1.0`
    /// - `roughness = 0.0`
    /// - `support = None` (unbounded)
    pub fn new(func: F) -> Self {
        Self {
            func,
            variance: 1.0,
            efficiency: 1.0,
            bandwidth_factor: 1.0,
            roughness: 0.0,
            support: None,
        }
    }

    /// Sets the variance (second moment).
    pub fn with_variance(mut self, variance: f64) -> Self {
        self.variance = variance;
        self
    }

    /// Sets the AMISE efficiency.
    pub fn with_efficiency(mut self, efficiency: f64) -> Self {
        self.efficiency = efficiency;
        self
    }

    /// Sets the bandwidth factor.
    pub fn with_bandwidth_factor(mut self, bandwidth_factor: f64) -> Self {
        self.bandwidth_factor = bandwidth_factor;
        self
    }

    /// Sets the roughness.
    pub fn with_roughness(mut self, roughness: f64) -> Self {
        self.roughness = roughness;
        self
    }

    /// Sets the support interval.
    pub fn with_support(mut self, support: (f64, f64)) -> Self {
        self.support = Some(support);
        self
    }
}

impl<F: Fn(f64) -> f64 + Clone + 'static + Send + Sync> Kernel for CustomKernel<F> {
    #[inline(always)]
    fn evaluate(&self, x: f64) -> f64 {
        (self.func)(x)
    }

    fn support(&self) -> Option<(f64, f64)> {
        self.support
    }

    fn bandwidth_factor(&self) -> f64 {
        self.bandwidth_factor
    }

    fn variance(&self) -> f64 {
        self.variance
    }

    fn efficiency(&self) -> f64 {
        self.efficiency
    }

    fn roughness(&self) -> f64 {
        self.roughness
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

impl<F: Fn(f64) -> f64> core::fmt::Debug for CustomKernel<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CustomKernel")
            .field("variance", &self.variance)
            .field("efficiency", &self.efficiency)
            .field("bandwidth_factor", &self.bandwidth_factor)
            .field("roughness", &self.roughness)
            .field("support", &self.support)
            .finish()
    }
}

// ============================================================================
// KernelType Enum
// ============================================================================

/// Enumeration of available kernel types.
///
/// Provides a convenient way to select kernels at runtime for methods like LOESS.
#[derive(Debug)]
pub enum KernelType {
    Gaussian,
    Epanechnikov,
    Triangular,
    Tricube,
    Bisquare,
    Uniform,
    Cosine,
    Logistic,
    Sigmoid,
    Custom(Box<dyn Kernel>),
}

impl Clone for KernelType {
    fn clone(&self) -> Self {
        match self {
            Self::Gaussian => Self::Gaussian,
            Self::Epanechnikov => Self::Epanechnikov,
            Self::Triangular => Self::Triangular,
            Self::Tricube => Self::Tricube,
            Self::Bisquare => Self::Bisquare,
            Self::Uniform => Self::Uniform,
            Self::Cosine => Self::Cosine,
            Self::Logistic => Self::Logistic,
            Self::Sigmoid => Self::Sigmoid,
            Self::Custom(k) => Self::Custom(k.clone_box()),
        }
    }
}

impl KernelType {
    /// Returns the kernel name.
    pub fn name(&self) -> &str {
        match self {
            Self::Gaussian => "Gaussian",
            Self::Epanechnikov => "Epanechnikov",
            Self::Triangular => "Triangular",
            Self::Tricube => "Tricube",
            Self::Bisquare => "Bisquare",
            Self::Uniform => "Uniform",
            Self::Cosine => "Cosine",
            Self::Logistic => "Logistic",
            Self::Sigmoid => "Sigmoid",
            Self::Custom(_) => "Custom",
        }
    }

    /// Checks if the kernel has bounded support.
    pub fn is_bounded(&self) -> bool {
        matches!(
            self,
            Self::Epanechnikov
                | Self::Triangular
                | Self::Tricube
                | Self::Bisquare
                | Self::Uniform
                | Self::Cosine
        ) || matches!(self, Self::Custom(k) if k.support().is_some())
    }

    /// Evaluates the kernel at a given point.
    #[inline(always)]
    pub fn evaluate(&self, x: f64) -> f64 {
        match self {
            Self::Gaussian => Gaussian.evaluate(x),
            Self::Epanechnikov => Epanechnikov.evaluate(x),
            Self::Triangular => Triangular.evaluate(x),
            Self::Tricube => Tricube.evaluate(x),
            Self::Bisquare => Bisquare.evaluate(x),
            Self::Uniform => Uniform.evaluate(x),
            Self::Cosine => Cosine.evaluate(x),
            Self::Logistic => Logistic.evaluate(x),
            Self::Sigmoid => Sigmoid.evaluate(x),
            Self::Custom(k) => k.evaluate(x),
        }
    }

    /// Fast evaluation with early return for bounded kernels outside support.
    #[inline(always)]
    pub fn evaluate_fast(&self, x: f64) -> f64 {
        if self.is_bounded() && x.abs() >= 1.0 {
            0.0
        } else {
            self.evaluate(x)
        }
    }

    /// Batch evaluation of multiple points.
    pub fn evaluate_batch(&self, values: &[f64]) -> Vec<f64> {
        values.iter().map(|&x| self.evaluate_fast(x)).collect()
    }

    /// Returns the AMISE-optimal bandwidth factor.
    pub fn bandwidth_factor(&self) -> f64 {
        match self {
            Self::Gaussian => Gaussian::BANDWIDTH_FACTOR,
            Self::Epanechnikov => Epanechnikov::BANDWIDTH_FACTOR,
            Self::Triangular => Triangular::BANDWIDTH_FACTOR,
            Self::Tricube => Tricube::BANDWIDTH_FACTOR,
            Self::Bisquare => Bisquare::BANDWIDTH_FACTOR,
            Self::Uniform => Uniform::BANDWIDTH_FACTOR,
            Self::Cosine => Cosine::BANDWIDTH_FACTOR,
            Self::Logistic => Logistic::BANDWIDTH_FACTOR,
            Self::Sigmoid => Sigmoid::BANDWIDTH_FACTOR,
            Self::Custom(k) => k.bandwidth_factor(),
        }
    }

    /// Returns the variance (second moment).
    pub fn variance(&self) -> f64 {
        match self {
            Self::Gaussian => Gaussian::VARIANCE,
            Self::Epanechnikov => Epanechnikov::VARIANCE,
            Self::Triangular => Triangular::VARIANCE,
            Self::Tricube => Tricube::VARIANCE,
            Self::Bisquare => Bisquare::VARIANCE,
            Self::Uniform => Uniform::VARIANCE,
            Self::Cosine => Cosine::VARIANCE,
            Self::Logistic => Logistic::VARIANCE,
            Self::Sigmoid => Sigmoid::VARIANCE,
            Self::Custom(k) => k.variance(),
        }
    }

    /// Returns the AMISE efficiency.
    pub fn efficiency(&self) -> f64 {
        match self {
            Self::Gaussian => Gaussian::EFFICIENCY,
            Self::Epanechnikov => Epanechnikov::EFFICIENCY,
            Self::Triangular => Triangular::EFFICIENCY,
            Self::Tricube => Tricube::EFFICIENCY,
            Self::Bisquare => Bisquare::EFFICIENCY,
            Self::Uniform => Uniform::EFFICIENCY,
            Self::Cosine => Cosine::EFFICIENCY,
            Self::Logistic => Logistic::EFFICIENCY,
            Self::Sigmoid => Sigmoid::EFFICIENCY,
            Self::Custom(k) => k.efficiency(),
        }
    }

    /// Returns the roughness.
    pub fn roughness(&self) -> f64 {
        match self {
            Self::Gaussian => Gaussian::ROUGHNESS,
            Self::Epanechnikov => Epanechnikov::ROUGHNESS,
            Self::Triangular => Triangular::ROUGHNESS,
            Self::Tricube => Tricube::ROUGHNESS,
            Self::Bisquare => Bisquare::ROUGHNESS,
            Self::Uniform => Uniform::ROUGHNESS,
            Self::Cosine => Cosine::ROUGHNESS,
            Self::Logistic => Logistic::ROUGHNESS,
            Self::Sigmoid => Sigmoid::ROUGHNESS,
            Self::Custom(k) => k.roughness(),
        }
    }

    /// Returns the kernel as a trait object.
    pub fn as_kernel(&self) -> Box<dyn Kernel> {
        match self {
            Self::Gaussian => Box::new(Gaussian),
            Self::Epanechnikov => Box::new(Epanechnikov),
            Self::Triangular => Box::new(Triangular),
            Self::Tricube => Box::new(Tricube),
            Self::Bisquare => Box::new(Bisquare),
            Self::Uniform => Box::new(Uniform),
            Self::Cosine => Box::new(Cosine),
            Self::Logistic => Box::new(Logistic),
            Self::Sigmoid => Box::new(Sigmoid),
            Self::Custom(k) => k.clone_box(),
        }
    }

    /// Computes normalized weights from distances using this kernel.
    ///
    /// # Arguments
    ///
    /// * `distances` - Distances from the query point
    /// * `bandwidth` - Bandwidth parameter (must be positive and finite)
    ///
    /// # Returns
    ///
    /// Normalized weights that sum to 1.0
    ///
    /// # Panics
    ///
    /// Panics if bandwidth is non-positive, infinite, or NaN.
    /// Panics if any distance is NaN.
    pub fn compute_distance_weights(&self, distances: &[f64], bandwidth: f64) -> Vec<f64> {
        if distances.is_empty() {
            return Vec::new();
        }

        validate_bandwidth(bandwidth);

        let inv_bandwidth = 1.0 / bandwidth;

        let mut weights: Vec<f64> = distances
            .iter()
            .map(|&d| {
                validate_distance(d);
                let u = (d * inv_bandwidth).abs();

                // Check if outside support
                let outside_support = match self {
                    // Built-in bounded kernels with standard support (-1, 1)
                    Self::Epanechnikov
                    | Self::Triangular
                    | Self::Tricube
                    | Self::Bisquare
                    | Self::Uniform
                    | Self::Cosine => u >= 1.0,
                    // Custom kernels with custom support
                    Self::Custom(k) => {
                        if let Some((_a, b)) = k.support() {
                            u >= b
                        } else {
                            false
                        }
                    }
                    // Unbounded kernels
                    _ => false,
                };

                if outside_support {
                    0.0
                } else {
                    self.evaluate(u)
                }
            })
            .collect();

        normalize_weights(&mut weights);
        weights
    }

    /// Returns the recommended kernel for kernel density estimation (KDE).
    ///
    /// Gaussian is preferred for KDE because:
    /// - Unbounded support (no boundary bias)
    /// - Smooth and differentiable everywhere
    /// - Well-studied theoretical properties
    pub fn recommended_for_kde() -> Self {
        Self::Gaussian
    }

    /// Returns the recommended kernel for LOESS regression.
    ///
    /// Tricube is preferred for LOESS because:
    /// - Compact support (computationally efficient)
    /// - High efficiency (0.9984)
    /// - Smooth transitions to zero at boundaries
    pub fn recommended_for_loess() -> Self {
        Self::Tricube
    }

    /// Returns the most AMISE-efficient kernel.
    ///
    /// Epanechnikov is theoretically optimal (efficiency = 1.0).
    pub fn most_efficient() -> Self {
        Self::Epanechnikov
    }
}

impl Default for KernelType {
    fn default() -> Self {
        Self::Gaussian
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Computes robust bisquare weights for iterative reweighting (IRLS).
///
/// # Arguments
///
/// * `residuals` - Residuals from the fit
/// * `scale` - Scale parameter (e.g., MAD)
/// * `tuning_constant` - Tuning constant (default: 6.0 for bisquare)
///
/// # Returns
///
/// Vector of robust weights in [0, 1]
pub fn robust_reweights(residuals: &[f64], scale: f64, tuning_constant: Option<f64>) -> Vec<f64> {
    let c = tuning_constant.unwrap_or(6.0);
    let bisquare = Bisquare;

    residuals
        .iter()
        .map(|&r| {
            let u = (r / scale).abs() / c;
            if u >= 1.0 {
                0.0
            } else {
                bisquare.evaluate_weight(u)
            }
        })
        .collect()
}

/// Normalizes weights to sum to 1.0.
///
/// If all weights are zero, assigns uniform weights.
pub fn normalize_weights(weights: &mut [f64]) {
    let sum: f64 = weights.iter().sum();

    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for w in weights.iter_mut() {
            *w *= inv_sum;
        }
    } else if !weights.is_empty() {
        // All weights are zero - assign uniform weights
        let uniform = 1.0 / weights.len() as f64;
        for w in weights.iter_mut() {
            *w = uniform;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod compile_time_tests {
    use super::*;

    // Compile-time assertions to verify efficiency calculations
    // These match Silverman (1986) Table 3.1 exact values

    /// Helper macro to assert values are approximately equal at compile time
    macro_rules! assert_approx_const {
        ($name:expr, $actual:expr, $expected:expr, $tolerance:expr) => {
            const _: () = {
                let diff = if $actual > $expected {
                    $actual - $expected
                } else {
                    $expected - $actual
                };
                assert!(diff < $tolerance, $name);
            };
        };
    }

    // Test Gaussian efficiency: (R_Epan/R_Gauss) × √(μ₂_Epan/μ₂_Gauss) ≈ 0.9512
    const GAUSSIAN_EFF: f64 = (ROUGHNESS_EPANECHNIKOV / ROUGHNESS_GAUSSIAN)
        * const_sqrt(VARIANCE_EPANECHNIKOV / VARIANCE_GAUSSIAN);
    assert_approx_const!(
        "Gaussian efficiency should be ≈ 0.9512",
        GAUSSIAN_EFF,
        0.9512,
        0.0001
    );

    // Test Bisquare efficiency: ≈ 0.9939
    const BISQUARE_EFF: f64 = (ROUGHNESS_EPANECHNIKOV / ROUGHNESS_BISQUARE)
        * const_sqrt(VARIANCE_EPANECHNIKOV / VARIANCE_BISQUARE);
    assert_approx_const!(
        "Bisquare efficiency should be ≈ 0.9939",
        BISQUARE_EFF,
        0.9939,
        0.0001
    );

    // Test Triangular efficiency: ≈ 0.9859
    const TRIANGULAR_EFF: f64 = (ROUGHNESS_EPANECHNIKOV / ROUGHNESS_TRIANGULAR)
        * const_sqrt(VARIANCE_EPANECHNIKOV / VARIANCE_TRIANGULAR);
    assert_approx_const!(
        "Triangular efficiency should be ≈ 0.9859",
        TRIANGULAR_EFF,
        0.9859,
        0.0001
    );

    // Test Uniform efficiency: ≈ 0.9295
    const UNIFORM_EFF: f64 = (ROUGHNESS_EPANECHNIKOV / ROUGHNESS_UNIFORM)
        * const_sqrt(VARIANCE_EPANECHNIKOV / VARIANCE_UNIFORM);
    assert_approx_const!(
        "Uniform efficiency should be ≈ 0.9295",
        UNIFORM_EFF,
        0.9295,
        0.0001
    );

    // Test Tricube efficiency: ≈ 0.9979
    const TRICUBE_EFF: f64 = (ROUGHNESS_EPANECHNIKOV / ROUGHNESS_TRICUBE)
        * const_sqrt(VARIANCE_EPANECHNIKOV / VARIANCE_TRICUBE);
    assert_approx_const!(
        "Tricube efficiency should be ≈ 0.9979",
        TRICUBE_EFF,
        0.9979,
        0.001
    );

    // Test bandwidth factors: √(1/μ₂)

    // Epanechnikov: √5 ≈ 2.236
    const EPAN_BW: f64 = const_sqrt(1.0 / VARIANCE_EPANECHNIKOV);
    assert_approx_const!(
        "Epanechnikov bandwidth factor should be √5 ≈ 2.236",
        EPAN_BW,
        2.236,
        0.001
    );

    // Gaussian: 1.0
    const GAUSS_BW: f64 = const_sqrt(1.0 / VARIANCE_GAUSSIAN);
    assert_approx_const!(
        "Gaussian bandwidth factor should be 1.0",
        GAUSS_BW,
        1.0,
        0.001
    );

    // Triangular: √6 ≈ 2.449
    const TRIANG_BW: f64 = const_sqrt(1.0 / VARIANCE_TRIANGULAR);
    assert_approx_const!(
        "Triangular bandwidth factor should be √6 ≈ 2.449",
        TRIANG_BW,
        2.449,
        0.001
    );

    // Bisquare: √7 ≈ 2.646
    const BISQ_BW: f64 = const_sqrt(1.0 / VARIANCE_BISQUARE);
    assert_approx_const!(
        "Bisquare bandwidth factor should be √7 ≈ 2.646",
        BISQ_BW,
        2.646,
        0.001
    );

    // Uniform: √3 ≈ 1.732
    const UNIF_BW: f64 = const_sqrt(1.0 / VARIANCE_UNIFORM);
    assert_approx_const!(
        "Uniform bandwidth factor should be √3 ≈ 1.732",
        UNIF_BW,
        1.732,
        0.001
    );

    // Test that Epanechnikov is optimal (efficiency = 1.0)
    const EPAN_EFF: f64 = 1.0;
    const _: () = assert!(
        EPAN_EFF == 1.0,
        "Epanechnikov should have efficiency exactly 1.0 (optimal)"
    );

    // Test that all efficiencies are ≤ 1.0 (Epanechnikov is optimal)
    const _: () = {
        assert!(GAUSSIAN_EFF <= 1.0, "Gaussian efficiency must be ≤ 1.0");
        assert!(BISQUARE_EFF <= 1.0, "Bisquare efficiency must be ≤ 1.0");
        assert!(TRIANGULAR_EFF <= 1.0, "Triangular efficiency must be ≤ 1.0");
        assert!(UNIFORM_EFF <= 1.0, "Uniform efficiency must be ≤ 1.0");
        assert!(TRICUBE_EFF <= 1.0, "Tricube efficiency must be ≤ 1.0");
    };

    // Runtime tests for verification
    #[test]
    fn test_efficiency_values() {
        // Gaussian
        let gaussian_eff = Gaussian::EFFICIENCY;
        assert!(
            (gaussian_eff - 0.9512).abs() < 0.001,
            "Gaussian efficiency: expected ≈0.9512, got {}",
            gaussian_eff
        );

        // Bisquare
        let bisquare_eff = Bisquare::EFFICIENCY;
        assert!(
            (bisquare_eff - 0.9939).abs() < 0.001,
            "Bisquare efficiency: expected ≈0.9939, got {}",
            bisquare_eff
        );

        // Triangular
        let triangular_eff = Triangular::EFFICIENCY;
        assert!(
            (triangular_eff - 0.9859).abs() < 0.001,
            "Triangular efficiency: expected ≈0.9859, got {}",
            triangular_eff
        );

        // Uniform
        let uniform_eff = Uniform::EFFICIENCY;
        assert!(
            (uniform_eff - 0.9295).abs() < 0.001,
            "Uniform efficiency: expected ≈0.9295, got {}",
            uniform_eff
        );

        // Tricube: ≈ 0.9979
        let tricube_eff = Tricube::EFFICIENCY;
        assert!(
            (tricube_eff - 0.9979).abs() < 0.001,
            "Tricube efficiency: expected ≈0.9979, got {}",
            tricube_eff
        );

        // Epanechnikov (optimal)
        assert_eq!(
            Epanechnikov::EFFICIENCY,
            1.0,
            "Epanechnikov must be exactly 1.0 (optimal)"
        );
    }

    #[test]
    fn test_bandwidth_factors() {
        // Test that bandwidth factors match √(1/μ₂)
        assert!(
            (Gaussian::BANDWIDTH_FACTOR - 1.0).abs() < 0.001,
            "Gaussian BW factor: expected 1.0, got {}",
            Gaussian::BANDWIDTH_FACTOR
        );

        assert!(
            (Epanechnikov::BANDWIDTH_FACTOR - 2.236).abs() < 0.001,
            "Epanechnikov BW factor: expected √5≈2.236, got {}",
            Epanechnikov::BANDWIDTH_FACTOR
        );

        assert!(
            (Triangular::BANDWIDTH_FACTOR - 2.449).abs() < 0.001,
            "Triangular BW factor: expected √6≈2.449, got {}",
            Triangular::BANDWIDTH_FACTOR
        );

        assert!(
            (Bisquare::BANDWIDTH_FACTOR - 2.646).abs() < 0.001,
            "Bisquare BW factor: expected √7≈2.646, got {}",
            Bisquare::BANDWIDTH_FACTOR
        );

        assert!(
            (Uniform::BANDWIDTH_FACTOR - 1.732).abs() < 0.001,
            "Uniform BW factor: expected √3≈1.732, got {}",
            Uniform::BANDWIDTH_FACTOR
        );
    }

    #[test]
    fn test_all_efficiencies_less_than_or_equal_to_epanechnikov() {
        // All kernels should have efficiency ≤ 1.0 (Epanechnikov is optimal)
        assert!(Gaussian::EFFICIENCY <= 1.0);
        assert!(Triangular::EFFICIENCY <= 1.0);
        assert!(Tricube::EFFICIENCY <= 1.0);
        assert!(Bisquare::EFFICIENCY <= 1.0);
        assert!(Uniform::EFFICIENCY <= 1.0);
        assert!(Cosine::EFFICIENCY <= 1.0);
        assert!(Logistic::EFFICIENCY <= 1.0);
        assert!(Sigmoid::EFFICIENCY <= 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prec::assert_abs_diff_eq;

    // ========================================================================
    // Module-specific precision constants
    // ========================================================================

    const KERNEL_EXACT_EPS: f64 = 1e-12;
    const KERNEL_INTEGRATION_EPS: f64 = 1e-4;
    const KERNEL_HEAVY_TAIL_EPS: f64 = 0.05;
    const KERNEL_SYMMETRY_EPS: f64 = 1e-15;

    // Helper function for numerical integration
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
    fn kernel_type_enum_evaluation() {
        let tricube = KernelType::Tricube;
        let gaussian = KernelType::Gaussian;

        assert_abs_diff_eq!(
            tricube.evaluate(0.0),
            70.0 / 81.0,
            epsilon = KERNEL_EXACT_EPS
        );
        assert_eq!(tricube.evaluate(1.0), 0.0);
        assert!(gaussian.evaluate(0.0) > gaussian.evaluate(1.0));
    }

    #[test]
    fn kernel_type_distance_weights() {
        let distances = vec![0.0, 0.5, 1.0, 2.0];
        let weights = KernelType::Tricube.compute_distance_weights(&distances, 1.0);

        assert_eq!(weights.len(), 4);
        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);
        assert_abs_diff_eq!(weights[3], 0.0, epsilon = KERNEL_EXACT_EPS);

        let sum: f64 = weights.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = KERNEL_EXACT_EPS);
    }

    #[test]
    fn bisquare_kernel_evaluate_vs_weight() {
        let bisquare = Bisquare;

        let stat_val = bisquare.evaluate(0.0);
        assert_abs_diff_eq!(stat_val, 15.0 / 16.0, epsilon = KERNEL_EXACT_EPS);

        let weight_val = bisquare.evaluate_weight(0.0);
        assert_abs_diff_eq!(weight_val, 1.0, epsilon = KERNEL_EXACT_EPS);

        assert_eq!(bisquare.evaluate(1.0), 0.0);
        assert_eq!(bisquare.evaluate_weight(1.0), 0.0);
    }

    #[test]
    fn gaussian_kernel_evaluate_vs_weight() {
        let gaussian = Gaussian;

        let stat_val = gaussian.evaluate(0.0);
        assert_abs_diff_eq!(
            stat_val,
            1.0 / (2.0 * PI).sqrt(),
            epsilon = KERNEL_EXACT_EPS
        );

        let weight_val = gaussian.evaluate_weight(0.0);
        assert_abs_diff_eq!(weight_val, 1.0, epsilon = KERNEL_EXACT_EPS);
    }

    #[test]
    fn uniform_kernel_evaluate_vs_weight() {
        let uniform = Uniform;

        assert_eq!(uniform.evaluate(0.5), 0.5);
        assert_eq!(uniform.evaluate_weight(0.5), 1.0);
        assert_eq!(uniform.evaluate(1.5), 0.0);
        assert_eq!(uniform.evaluate_weight(1.5), 0.0);
    }

    #[test]
    fn cosine_kernel_evaluate_vs_weight() {
        let cosine = Cosine;

        let stat_val = cosine.evaluate(0.0);
        assert_abs_diff_eq!(stat_val, PI / 4.0, epsilon = KERNEL_EXACT_EPS);

        let weight_val = cosine.evaluate_weight(0.0);
        assert_abs_diff_eq!(weight_val, 1.0, epsilon = KERNEL_EXACT_EPS);

        assert_abs_diff_eq!(cosine.evaluate(1.0), 0.0, epsilon = KERNEL_SYMMETRY_EPS);
        assert_abs_diff_eq!(cosine.evaluate(-1.0), 0.0, epsilon = KERNEL_SYMMETRY_EPS);
    }

    #[test]
    fn kernel_weight_vs_statistical_relationship() {
        let bisquare = Bisquare;
        let x = 0.5;

        let weight = bisquare.evaluate_weight(x);
        let stat = bisquare.evaluate(x);

        assert_abs_diff_eq!(stat, (15.0 / 16.0) * weight, epsilon = KERNEL_EXACT_EPS);
    }

    #[test]
    fn kernel_second_moment_properties() {
        let bounded_kernels = [
            (
                "Epanechnikov",
                Box::new(Epanechnikov) as Box<dyn Kernel>,
                Epanechnikov::VARIANCE,
            ),
            (
                "Triangular",
                Box::new(Triangular) as Box<dyn Kernel>,
                Triangular::VARIANCE,
            ),
            (
                "Tricube",
                Box::new(Tricube) as Box<dyn Kernel>,
                Tricube::VARIANCE,
            ),
            (
                "Bisquare",
                Box::new(Bisquare) as Box<dyn Kernel>,
                Bisquare::VARIANCE,
            ),
            (
                "Uniform",
                Box::new(Uniform) as Box<dyn Kernel>,
                Uniform::VARIANCE,
            ),
            (
                "Cosine",
                Box::new(Cosine) as Box<dyn Kernel>,
                Cosine::VARIANCE,
            ),
        ];

        for (_name, kernel, expected_var) in bounded_kernels {
            let integral = integrate(|u| u * u * kernel.evaluate(u), -1.0, 1.0, 10_000);
            assert_abs_diff_eq!(integral, expected_var, epsilon = KERNEL_INTEGRATION_EPS);
        }

        let unbounded_kernels = [
            (
                "Gaussian",
                Box::new(Gaussian) as Box<dyn Kernel>,
                Gaussian::VARIANCE,
                10.0,
            ),
            (
                "Logistic",
                Box::new(Logistic) as Box<dyn Kernel>,
                Logistic::VARIANCE,
                50.0,
            ),
            (
                "Sigmoid",
                Box::new(Sigmoid) as Box<dyn Kernel>,
                Sigmoid::VARIANCE,
                30.0,
            ),
        ];

        for (_name, kernel, expected_var, bound) in unbounded_kernels {
            let integral = integrate(|u| u * u * kernel.evaluate(u), -bound, bound, 100_000);
            assert_abs_diff_eq!(integral, expected_var, epsilon = KERNEL_HEAVY_TAIL_EPS);
        }
    }

    #[test]
    fn custom_kernel_with_metadata() {
        let laplacian = |u: f64| (-u.abs()).exp();
        let custom = CustomKernel::new(laplacian)
            .with_variance(2.0)
            .with_efficiency(0.85)
            .with_bandwidth_factor(1.5)
            .with_roughness(0.5)
            .with_support((-5.0, 5.0));

        assert_abs_diff_eq!(custom.evaluate(0.0), 1.0, epsilon = KERNEL_EXACT_EPS);
        assert_abs_diff_eq!(
            custom.evaluate(1.0),
            (-1.0_f64).exp(),
            epsilon = KERNEL_EXACT_EPS
        );

        assert_abs_diff_eq!(custom.variance(), 2.0, epsilon = KERNEL_EXACT_EPS);
        assert_abs_diff_eq!(custom.efficiency(), 0.85, epsilon = KERNEL_EXACT_EPS);
        assert_abs_diff_eq!(custom.bandwidth_factor(), 1.5, epsilon = KERNEL_EXACT_EPS);
        assert_abs_diff_eq!(custom.roughness(), 0.5, epsilon = KERNEL_EXACT_EPS);
        assert_eq!(custom.support(), Some((-5.0, 5.0)));
    }

    #[test]
    fn custom_kernel_support_bounds_correctness() {
        // Custom kernel with support (-2, 2)
        // Function returns 1.0 if |u| < 2.0, else 0.0
        let bounded =
            CustomKernel::new(|u| if u.abs() < 2.0 { 1.0 } else { 0.0 }).with_support((-2.0, 2.0));

        let kernel = KernelType::Custom(Box::new(bounded));
        let distances = vec![0.0, 1.0, 1.99, 2.0, 3.0];
        let weights = kernel.compute_distance_weights(&distances, 1.0);

        assert!(weights[0] > 0.0);
        assert!(weights[1] > 0.0);
        assert!(weights[2] > 0.0);
        assert_abs_diff_eq!(weights[3], 0.0, epsilon = KERNEL_EXACT_EPS);
        assert_abs_diff_eq!(weights[4], 0.0, epsilon = KERNEL_EXACT_EPS);
    }

    #[test]
    fn kernel_type_custom_with_metadata() {
        let laplacian = |u: f64| (-u.abs()).exp();
        let custom = CustomKernel::new(laplacian)
            .with_variance(2.0)
            .with_efficiency(0.85);

        let kernel_type = KernelType::Custom(Box::new(custom));

        assert_abs_diff_eq!(kernel_type.variance(), 2.0, epsilon = KERNEL_EXACT_EPS);
        assert_abs_diff_eq!(kernel_type.efficiency(), 0.85, epsilon = KERNEL_EXACT_EPS);
        assert_abs_diff_eq!(kernel_type.evaluate(0.0), 1.0, epsilon = KERNEL_EXACT_EPS);
    }

    #[test]
    fn custom_kernel_bounded_support() {
        let bounded_kernel = |u: f64| {
            if u.abs() < 2.0 {
                1.0 - u.abs() / 2.0
            } else {
                0.0
            }
        };
        let custom = CustomKernel::new(bounded_kernel).with_support((-2.0, 2.0));

        let kernel_type = KernelType::Custom(Box::new(custom.clone()));

        assert!(kernel_type.is_bounded());
        assert_eq!(custom.support(), Some((-2.0, 2.0)));

        assert_eq!(kernel_type.evaluate_fast(3.0), 0.0);
    }

    #[test]
    fn kernel_type_custom_function_backward_compat() {
        let distances = vec![0.0, 1.0, 2.0];
        let simple_kernel = CustomKernel::new(|u: f64| if u < 2.0 { 1.0 / (1.0 + u) } else { 0.0 });
        let kernel_type = KernelType::Custom(Box::new(simple_kernel));

        let weights = kernel_type.compute_distance_weights(&distances, 1.0);

        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);

        let sum: f64 = weights.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = KERNEL_EXACT_EPS);
    }

    #[test]
    fn kernel_type_fast_evaluation() {
        let bounded_kernels = [
            KernelType::Epanechnikov,
            KernelType::Triangular,
            KernelType::Tricube,
            KernelType::Bisquare,
            KernelType::Uniform,
            KernelType::Cosine,
        ];

        for kernel in bounded_kernels {
            assert_eq!(kernel.evaluate_fast(1.0), 0.0);
            assert_eq!(kernel.evaluate_fast(1.5), 0.0);
            assert_eq!(kernel.evaluate_fast(10.0), 0.0);

            assert_eq!(kernel.evaluate_fast(0.5), kernel.evaluate(0.5));
            assert_eq!(kernel.evaluate_fast(0.0), kernel.evaluate(0.0));
        }
    }

    #[test]
    fn kernel_type_batch_evaluation() {
        let values = vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.5];
        let kernel = KernelType::Tricube;

        let batch_results = kernel.evaluate_batch(&values);
        let individual_results: Vec<f64> =
            values.iter().map(|&v| kernel.evaluate_fast(v)).collect();

        assert_eq!(batch_results, individual_results);
    }

    #[test]
    fn kernel_recommendations() {
        assert_eq!(
            KernelType::recommended_for_kde().name(),
            KernelType::Gaussian.name()
        );
        assert_eq!(
            KernelType::recommended_for_loess().name(),
            KernelType::Tricube.name()
        );
        assert_eq!(
            KernelType::most_efficient().name(),
            KernelType::Epanechnikov.name()
        );
    }

    #[test]
    fn kernel_properties() {
        let gaussian = KernelType::Gaussian;
        let tricube = KernelType::Tricube;
        let bisquare = KernelType::Bisquare;
        let triangular = KernelType::Triangular;
        let uniform = KernelType::Uniform;
        let epanechnikov = KernelType::Epanechnikov;
        let cosine = KernelType::Cosine;
        let logistic = KernelType::Logistic;
        let sigmoid = KernelType::Sigmoid;

        assert_eq!(gaussian.name(), "Gaussian");
        assert_eq!(tricube.name(), "Tricube");
        assert_eq!(epanechnikov.name(), "Epanechnikov");

        assert!(!gaussian.is_bounded());
        assert!(tricube.is_bounded());
        assert!(epanechnikov.is_bounded());

        // Epanechnikov is optimal with efficiency = 1.0
        assert_eq!(epanechnikov.efficiency(), 1.0);

        // Verify efficiency values match Silverman (1986) Table 3.1
        // Gaussian: ≈ 0.9512
        let gaussian_eff = gaussian.efficiency();
        assert!(
            (gaussian_eff - 0.9512).abs() < 0.001,
            "Gaussian efficiency: {}",
            gaussian_eff
        );

        // Bisquare: ≈ 0.9939
        let bisquare_eff = bisquare.efficiency();
        assert!(
            (bisquare_eff - 0.9939).abs() < 0.001,
            "Bisquare efficiency: {}",
            bisquare_eff
        );

        // Triangular: ≈ 0.9859
        let triangular_eff = triangular.efficiency();
        assert!(
            (triangular_eff - 0.9859).abs() < 0.001,
            "Triangular efficiency: {}",
            triangular_eff
        );

        // Uniform: ≈ 0.9295
        let uniform_eff = uniform.efficiency();
        assert!(
            (uniform_eff - 0.9295).abs() < 0.001,
            "Uniform efficiency: {}",
            uniform_eff
        );

        // Tricube: ≈ 0.9979 (calculated)
        let tricube_eff = tricube.efficiency();
        assert!(
            (tricube_eff - 0.9979).abs() < 0.001,
            "Tricube efficiency: {}",
            tricube_eff
        );

        // Cosine: ≈ 0.9995 (calculated)
        let cosine_eff = cosine.efficiency();
        assert!(
            (cosine_eff - 0.9995).abs() < 0.001,
            "Cosine efficiency: {}",
            cosine_eff
        );

        // Logistic: ≈ 0.8878 (calculated)
        let logistic_eff = logistic.efficiency();
        assert!(
            (logistic_eff - 0.8878).abs() < 0.001,
            "Logistic efficiency: {}",
            logistic_eff
        );

        // Sigmoid: ≈ 0.8424 (calculated)
        let sigmoid_eff = sigmoid.efficiency();
        assert!(
            (sigmoid_eff - 0.8424).abs() < 0.001,
            "Sigmoid efficiency: {}",
            sigmoid_eff
        );

        // All efficiencies should be ≤ 1.0 (Epanechnikov is optimal)
        assert!(gaussian_eff <= 1.0);
        assert!(bisquare_eff <= 1.0);
        assert!(triangular_eff <= 1.0);
        assert!(tricube_eff <= 1.0);
        assert!(uniform_eff <= 1.0);
        assert!(cosine_eff <= 1.0);
        assert!(logistic_eff <= 1.0);
        assert!(sigmoid_eff <= 1.0);

        // Verify canonical bandwidth factors
        // Epanechnikov: √5 ≈ 2.236
        assert!((epanechnikov.bandwidth_factor() - 2.236).abs() < 0.001);
        // Gaussian: √1 = 1.0
        assert!((gaussian.bandwidth_factor() - 1.0).abs() < 0.001);
        // Triangular: √6 ≈ 2.449
        assert!((triangular.bandwidth_factor() - 2.449).abs() < 0.001);
    }

    #[test]
    fn compute_distance_weights_error_handling() {
        let kernel = KernelType::Gaussian;

        let empty_weights = kernel.compute_distance_weights(&[], 1.0);
        assert!(empty_weights.is_empty());
    }

    #[test]
    #[should_panic(expected = "Bandwidth must be positive")]
    fn compute_distance_weights_negative_bandwidth() {
        let kernel = KernelType::Gaussian;
        let distances = vec![1.0, 2.0];
        kernel.compute_distance_weights(&distances, -1.0);
    }

    #[test]
    #[should_panic(expected = "Bandwidth must be positive")]
    fn compute_distance_weights_zero_bandwidth() {
        let kernel = KernelType::Gaussian;
        let distances = vec![1.0, 2.0];
        kernel.compute_distance_weights(&distances, 0.0);
    }

    #[test]
    #[should_panic(expected = "Bandwidth must be finite")]
    fn compute_distance_weights_infinite_bandwidth() {
        let kernel = KernelType::Gaussian;
        let distances = vec![1.0, 2.0];
        kernel.compute_distance_weights(&distances, f64::INFINITY);
    }

    #[test]
    #[should_panic(expected = "Distance cannot be NaN")]
    fn compute_distance_weights_nan_distance() {
        let kernel = KernelType::Gaussian;
        let distances = vec![1.0, f64::NAN, 2.0];
        kernel.compute_distance_weights(&distances, 1.0);
    }

    #[test]
    fn robust_reweights_test() {
        let residuals = vec![0.0, 1.0, 3.0, 10.0];
        let scale = 1.0;
        let weights = robust_reweights(&residuals, scale, Some(6.0));

        assert_abs_diff_eq!(weights[0], 1.0, epsilon = KERNEL_EXACT_EPS);
        assert!(weights[1] > weights[2]);
        assert!(weights[2] > 0.0);
        assert_abs_diff_eq!(weights[3], 0.0, epsilon = KERNEL_EXACT_EPS);
    }

    #[test]
    fn normalize_weights_test() {
        let mut weights = vec![2.0, 4.0, 6.0];
        normalize_weights(&mut weights);

        let sum: f64 = weights.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = KERNEL_EXACT_EPS);

        assert_abs_diff_eq!(weights[0], 1.0 / 6.0, epsilon = KERNEL_EXACT_EPS);
        assert_abs_diff_eq!(weights[1], 2.0 / 6.0, epsilon = KERNEL_EXACT_EPS);
        assert_abs_diff_eq!(weights[2], 3.0 / 6.0, epsilon = KERNEL_EXACT_EPS);
    }

    #[test]
    fn normalize_zero_weights() {
        let mut weights = vec![0.0, 0.0, 0.0];
        normalize_weights(&mut weights);

        assert_abs_diff_eq!(weights[0], 1.0 / 3.0, epsilon = KERNEL_EXACT_EPS);
        assert_abs_diff_eq!(weights[1], 1.0 / 3.0, epsilon = KERNEL_EXACT_EPS);
        assert_abs_diff_eq!(weights[2], 1.0 / 3.0, epsilon = KERNEL_EXACT_EPS);
    }

    #[test]
    fn kernel_properties_consistency() {
        let kernels = [
            KernelType::Gaussian,
            KernelType::Epanechnikov,
            KernelType::Triangular,
            KernelType::Tricube,
            KernelType::Bisquare,
            KernelType::Uniform,
            KernelType::Cosine,
            KernelType::Logistic,
            KernelType::Sigmoid,
        ];

        for kernel in kernels {
            if kernel.is_bounded() {
                for x in [-0.5, -0.1, 0.1, 0.5] {
                    assert_abs_diff_eq!(
                        kernel.evaluate(x),
                        kernel.evaluate(-x),
                        epsilon = KERNEL_SYMMETRY_EPS
                    );
                }
            }

            for x in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
                assert!(
                    kernel.evaluate(x) >= 0.0,
                    "Kernel {} negative at x={}",
                    kernel.name(),
                    x
                );
            }

            if kernel.is_bounded() {
                assert!(
                    kernel.evaluate(0.0) >= kernel.evaluate(0.5),
                    "Kernel {} not peaked at origin",
                    kernel.name()
                );
            }
        }
    }

    #[test]
    fn kernel_normalization_check() {
        let bounded_kernels = [
            (KernelType::Epanechnikov, 1.0),
            (KernelType::Triangular, 1.0),
            (KernelType::Bisquare, 1.0),
            (KernelType::Uniform, 1.0),
            (KernelType::Cosine, 1.0),
        ];

        for (kernel_type, expected) in bounded_kernels {
            let kernel_obj = kernel_type.as_kernel();
            let integral = integrate(|u| kernel_obj.evaluate(u), -1.0, 1.0, 20_000);
            assert_abs_diff_eq!(integral, expected, epsilon = KERNEL_INTEGRATION_EPS);
        }
    }

    #[test]
    fn kernel_variance_properties() {
        let gaussian_kernel = Gaussian;
        let epanechnikov_kernel = Epanechnikov;
        let triangular_kernel = Triangular;

        assert_abs_diff_eq!(
            gaussian_kernel.variance(),
            Gaussian::VARIANCE,
            epsilon = KERNEL_EXACT_EPS
        );
        assert_abs_diff_eq!(
            epanechnikov_kernel.variance(),
            Epanechnikov::VARIANCE,
            epsilon = KERNEL_EXACT_EPS
        );
        assert_abs_diff_eq!(
            triangular_kernel.variance(),
            Triangular::VARIANCE,
            epsilon = KERNEL_EXACT_EPS
        );
    }

    #[test]
    fn uniform_behavior() {
        let k = Uniform;
        assert_eq!(k.evaluate(0.0), 0.5);
        assert_eq!(k.evaluate(0.8), 0.5);

        assert_eq!(k.evaluate(1.0), 0.0);
        assert_eq!(k.evaluate(1.01), 0.0);
        assert_eq!(k.evaluate(-1.01), 0.0);
        assert_abs_diff_eq!(
            k.evaluate(0.5),
            k.evaluate(-0.5),
            epsilon = KERNEL_SYMMETRY_EPS
        );

        let integral = integrate(|u| k.evaluate(u), -1.0, 1.0, 20_000);
        assert_abs_diff_eq!(integral, 1.0, epsilon = KERNEL_INTEGRATION_EPS);
    }

    #[test]
    fn tricube_basic_properties() {
        let k = Tricube;
        assert_abs_diff_eq!(
            k.evaluate(0.5),
            k.evaluate(-0.5),
            epsilon = KERNEL_SYMMETRY_EPS
        );
        assert_eq!(k.evaluate(1.0), 0.0);
        assert_eq!(k.evaluate(-1.0), 0.0);

        let peak_value = 70.0 / 81.0;
        assert_abs_diff_eq!(k.evaluate(0.0), peak_value, epsilon = KERNEL_EXACT_EPS);

        assert!(k.evaluate(0.25) > k.evaluate(0.5));
        assert!(k.evaluate(0.5) > k.evaluate(0.75));
    }

    #[test]
    fn gaussian_behavior() {
        let k = Gaussian;
        assert_abs_diff_eq!(
            k.evaluate(0.5),
            k.evaluate(-0.5),
            epsilon = KERNEL_SYMMETRY_EPS
        );
        assert!(k.evaluate(0.0) > k.evaluate(1.0));
        assert!(k.evaluate(2.0) < 0.2);
        for u in [-3.0, -1.0, 0.0, 1.0, 3.0] {
            assert!(k.evaluate(u) >= 0.0);
        }
    }

    #[test]
    fn bandwidth_scaling_equivalence() {
        let g = Gaussian;
        let scaled = g.evaluate_with_bandwidth(0.5, 2.0);
        let manual = g.evaluate(0.25) / 2.0;
        assert_abs_diff_eq!(scaled, manual, epsilon = KERNEL_SYMMETRY_EPS);
    }
}
