//! Mathematical and statistical constants for the library.
//!
//! # Kernel Constants
//!
//! ## Core Properties
//! - **Variance (μ₂)**: Second moment `∫ u² K(u) du`
//! - **Roughness (R)**: Squared integral `∫ K(u)² du`
//! - **Efficiency**: AMISE efficiency relative to Epanechnikov (1.0 = optimal)
//! - **Bandwidth Factors**: AMISE-optimal scaling relative to Epanechnikov
//!
//! ## Usage
//! If Silverman's rule gives bandwidth `h` for Epanechnikov, then:
//! - Gaussian: `h * 0.45` (narrower)
//! - Triangular: `h * 1.10` (wider)
//!
//! ## References
//! - Silverman (1986), *Density Estimation*, Table 3.1
//! - Wand & Jones (1995), *Kernel Smoothing*, Table 2.2
// ====================
// General mathematical constants
// ====================

/// Constant value for `sqrt(2 * pi)`
pub const SQRT_2PI: f64 = 2.5066282746310005024157652848110452530069867406099;

/// Constant value for `sqrt(pi)`
pub const SQRT_PI: f64 = 1.7724538509055160272981674833411451827975494561223871;

/// Constant value for `ln(pi)`
pub const LN_PI: f64 = 1.1447298858494001741434273513530587116472948129153;

/// Constant value for `ln(sqrt(2 * pi))`
pub const LN_SQRT_2PI: f64 = 0.91893853320467274178032973640561763986139747363778;

/// Constant value for `ln(sqrt(2 * pi * e))`
pub const LN_SQRT_2PIE: f64 = 1.4189385332046727417803297364056176398613974736378;

/// Constant value for `ln(2 * sqrt(e / pi))`
pub const LN_2_SQRT_E_OVER_PI: f64 = 0.6207822376352452223455184457816472122518527279025978;

/// Constant value for `2 * sqrt(e / pi)`
pub const TWO_SQRT_E_OVER_PI: f64 = 1.8603827342052657173362492472666631120594218414085755;

/// Constant value for Euler-Mascheroni constant `γ = lim_{n→∞} (∑_{k=1}^n 1/k - ln n)`
pub const EULER_MASCHERONI: f64 =
    0.5772156649015328606065120900824024310421593359399235988057672348849;

// ====================
// Kernel constants
// ====================

use core::f64::consts::PI;

// Kernel Variance: μ₂ = ∫ u² K(u) du
pub const VARIANCE_GAUSSIAN: f64 = 1.0;
pub const VARIANCE_EPANECHNIKOV: f64 = 1.0 / 5.0;
pub const VARIANCE_TRIANGULAR: f64 = 1.0 / 6.0;
pub const VARIANCE_TRICUBE: f64 = 35.0 / 243.0;
pub const VARIANCE_BISQUARE: f64 = 1.0 / 7.0;
pub const VARIANCE_UNIFORM: f64 = 1.0 / 3.0;
pub const VARIANCE_COSINE: f64 = 1.0 - 8.0 / (PI * PI);
pub const VARIANCE_LOGISTIC: f64 = PI * PI / 3.0;
pub const VARIANCE_SIGMOID: f64 = PI * PI / 4.0;

// Kernel Roughness: R(K) = ∫ K(u)² du
pub const ROUGHNESS_GAUSSIAN: f64 = 1.0 / (2.0 * SQRT_PI);
pub const ROUGHNESS_EPANECHNIKOV: f64 = 3.0 / 5.0;
pub const ROUGHNESS_TRIANGULAR: f64 = 2.0 / 3.0;
pub const ROUGHNESS_TRICUBE: f64 = 175.0 / 247.0;
pub const ROUGHNESS_BISQUARE: f64 = 5.0 / 7.0;
pub const ROUGHNESS_UNIFORM: f64 = 1.0 / 2.0;
pub const ROUGHNESS_COSINE: f64 = PI * PI / 16.0;
pub const ROUGHNESS_LOGISTIC: f64 = 1.0 / 6.0;
pub const ROUGHNESS_SIGMOID: f64 = 2.0 / (PI * PI);
