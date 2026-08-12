//! Provides the [beta](https://en.wikipedia.org/wiki/Beta_function) and related
//! function
//!
//! This module sets the default precision more tightly than crate defaults for `DEFAULT_EPS`
//!
//! The implementation combines the incomplete-beta series, recurrences, and
//! continued fraction in [DLMF 8.17](https://dlmf.nist.gov/8.17), the BGRAT
//! expansion in [Algorithm 708](https://doi.org/10.1145/131766.131776), the
//! large-parameter expansion in [DLMF 8.18](https://dlmf.nist.gov/8.18), and
//! the small-argument log-gamma series in [DLMF 5.7.3](https://dlmf.nist.gov/5.7.E3).

mod api;
mod asymptotic;
mod bgrat;
mod dd;
mod forward;
mod fraction;
mod inverse;
mod log_beta;
mod log_forward;
mod prefactor;
mod quantile;
mod recurrence;
mod scaled_gamma;
mod series;
mod small_gamma;

pub use api::{beta, beta_inc, beta_reg, checked_beta, checked_beta_inc};
pub use forward::checked_beta_reg;
pub use inverse::inv_beta_reg;
pub use log_beta::{checked_ln_beta, ln_beta};
pub(crate) use log_forward::checked_ln_beta_reg_complement;

use asymptotic::*;
use bgrat::*;
use dd::*;
use fraction::*;
use log_beta::*;
use log_forward::*;
use prefactor::*;
use quantile::*;
use recurrence::*;
use scaled_gamma::*;
use series::*;
use small_gamma::*;

use crate::consts;
use crate::function::{erf, gamma};
use crate::prec;
#[cfg(all(not(feature = "std"), not(test)))]
use num_traits::Float;

/// sample case of module level precision
#[cfg(test)]
const MODULE_EPS: f64 = 1e-15;
const STIRLING_MIN: f64 = 32.0;
const SCALED_GAMMA_MIN_X: f64 = 64.0;
const MAX_BETA_REG_ITERATIONS: u32 = 100_000;
const ASYMPTOTIC_MIN_SUM: f64 = 1.2e8;
const ASYMPTOTIC_MIN_SHAPE: f64 = 1.2e7;
const ASYMPTOTIC_MAX_DEVIANCE: f64 = 1.5;

/// Represents the errors that can occur when computing the natural logarithm
/// of the beta function or the regularized lower incomplete beta function.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum BetaFuncError {
    /// `a` is zero or less than zero.
    ANotGreaterThanZero,

    /// `b` is zero or less than zero.
    BNotGreaterThanZero,

    /// `x` is not in `[0, 1]`.
    XOutOfRange,

    /// The numerical method did not converge.
    ConvergenceFailed,
}

impl core::fmt::Display for BetaFuncError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            BetaFuncError::ANotGreaterThanZero => write!(f, "a is zero or less than zero"),
            BetaFuncError::BNotGreaterThanZero => write!(f, "b is zero or less than zero"),
            BetaFuncError::XOutOfRange => write!(f, "x is not in [0, 1]"),
            BetaFuncError::ConvergenceFailed => write!(f, "computation did not converge"),
        }
    }
}

impl core::error::Error for BetaFuncError {}

pub(crate) fn checked_ln_beta_reg(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    checked_ln_beta_reg_with_log_beta(a, b, x, None)
}

#[cfg(test)]
mod tests;
