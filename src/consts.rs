//! Defines mathematical expressions commonly used when computing distribution
//! values as constants

/// Constant value for `sqrt(2 * pi)`
pub const SQRT_2PI: f64 = 2.5066282746310005024157652848110452530069867406099;

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

/// Constant value for Euler-Masheroni constant `lim(n -> inf) { sum(k=1 -> n)
/// { 1/k - ln(n) } }`
pub const EULER_MASCHERONI: f64 =
    0.5772156649015328606065120900824024310421593359399235988057672348849;
