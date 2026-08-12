//! Provides the [beta](https://en.wikipedia.org/wiki/Beta_function) and related
//! function
//!
//! This module sets the default precision more tightly than crate defaults for `DEFAULT_EPS`

use crate::consts;
use crate::function::{erf, gamma};
use crate::prec;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

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

/// Computes the natural logarithm
/// of the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter
/// and `a > 0`, `b > 0`.
///
/// # Panics
///
/// if `a <= 0.0` or `b <= 0.0`
pub fn ln_beta(a: f64, b: f64) -> f64 {
    checked_ln_beta(a, b).unwrap()
}

/// Computes the natural logarithm
/// of the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter
/// and `a > 0`, `b > 0`.
///
/// # Errors
///
/// if `a <= 0.0` or `b <= 0.0`
pub fn checked_ln_beta(a: f64, b: f64) -> Result<f64, BetaFuncError> {
    if a <= 0.0 {
        Err(BetaFuncError::ANotGreaterThanZero)
    } else if b <= 0.0 {
        Err(BetaFuncError::BNotGreaterThanZero)
    } else {
        Ok(ln_beta_stable(a, b))
    }
}

fn stirling_correction(x: f64) -> f64 {
    let reciprocal = 1.0 / x;
    let x2 = reciprocal * reciprocal;
    reciprocal
        * (1.0 / 12.0
            + x2 * (-1.0 / 360.0
                + x2 * (1.0 / 1260.0
                    + x2 * (-1.0 / 1680.0 + x2 * (1.0 / 1188.0 - x2 * 691.0 / 360360.0)))))
}

fn stirling_correction_log(log_x: f64) -> f64 {
    let reciprocal = (-log_x).exp();
    let x2 = reciprocal * reciprocal;
    reciprocal
        * (1.0 / 12.0
            + x2 * (-1.0 / 360.0
                + x2 * (1.0 / 1260.0
                    + x2 * (-1.0 / 1680.0 + x2 * (1.0 / 1188.0 - x2 * 691.0 / 360360.0)))))
}

fn ln_gamma_delta(base: f64, delta: f64) -> f64 {
    let log_ratio = (delta / base).ln_1p();
    let log_sum = base.ln() + log_ratio;
    delta * base.ln() + base.mul_add(log_ratio, (delta - 0.5) * log_ratio) - delta
        + stirling_correction_log(log_sum)
        - stirling_correction(base)
}

fn ln_gamma_stable(x: f64) -> f64 {
    if x < 0.5 {
        gamma::ln_gamma(1.0 + x) - x.ln()
    } else {
        gamma::ln_gamma(x)
    }
}

fn ln_gamma_one_plus_series(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 31] = [
        0.8224670334241132,
        -0.40068563438653143,
        0.27058080842778455,
        -0.20738555102867398,
        0.1695571769974082,
        -0.14404989676884612,
        0.12550966952474304,
        -0.11133426586956469,
        0.10009945751278181,
        -0.09095401714582904,
        0.083353840546109,
        -0.0769325164113522,
        0.07143294629536133,
        -0.06666870588242047,
        0.06250095514121304,
        -0.058823978658684585,
        0.055555767627403614,
        -0.05263167937961666,
        0.05000004769810169,
        -0.047619070330142226,
        0.04545455629320467,
        -0.04347826605304026,
        0.04166666915034121,
        -0.04000000119214014,
        0.03846153903467518,
        -0.037037037312989324,
        0.035714285847333355,
        -0.034482758684919304,
        0.03333333336437758,
        -0.03225806453115042,
        0.03125000000727597,
    ];
    let mut polynomial = *COEFFICIENTS.last().unwrap();
    for coefficient in COEFFICIENTS[..COEFFICIENTS.len() - 1].iter().rev() {
        polynomial = polynomial.mul_add(x, *coefficient);
    }
    x * (-consts::EULER_MASCHERONI + x * polynomial)
}

fn accurate_ln_dd(value: (f64, f64)) -> (f64, f64) {
    let logarithm = accurate_ln(value.0);
    dd_add(logarithm, ((value.1 / value.0).ln_1p(), 0.0))
}

fn accurate_ln_one_plus_dd(value: (f64, f64)) -> (f64, f64) {
    if value.0 == 0.0 && value.1 == 0.0 {
        return (0.0, 0.0);
    }
    if value.0.abs() > 0.5 {
        return accurate_ln_dd(dd_add((1.0, 0.0), value));
    }
    let ratio = dd_div(value, dd_add((2.0, 0.0), value));
    let ratio_squared = dd_mul(ratio, ratio);
    let mut term = ratio;
    let mut sum = ratio;
    for index in 1..=24 {
        term = dd_mul(term, ratio_squared);
        if term.0 == 0.0 && term.1 == 0.0 {
            break;
        }
        sum = dd_add(sum, dd_div_f64(term, f64::from(2 * index + 1)));
    }
    dd_mul((2.0, 0.0), sum)
}

fn accurate_ln_one_minus_dd(value: f64) -> (f64, f64) {
    if value <= 0.5 {
        accurate_ln_one_plus_dd((-value, 0.0))
    } else {
        let complement = two_sum(1.0, -value);
        accurate_ln_dd(complement)
    }
}

fn ln_gamma_stirling_parts(value: (f64, f64)) -> (f64, f64) {
    let shifted = dd_add(value, (-0.5, 0.0));
    let mut result = dd_mul(shifted, accurate_ln_dd(value));
    result = dd_add(result, (-value.0, -value.1));
    result = dd_add(result, (consts::LN_SQRT_2PI, -3.8782941580672414e-17));
    dd_add(result, (stirling_correction(value.0), 0.0))
}

fn ln_gamma_accurate_parts(x: f64) -> (f64, f64) {
    if x == 1.0 || x == 2.0 {
        return (0.0, 0.0);
    }
    if x <= 0.125 {
        let mut result = dd_add((x, 0.0), (1.0, 0.0));
        let mut recurrence = (0.0, 0.0);
        while result.0 < STIRLING_MIN {
            recurrence = dd_add(recurrence, accurate_ln_dd(result));
            result = dd_add(result, (1.0, 0.0));
        }
        let gamma_one_plus = dd_add(
            ln_gamma_stirling_parts(result),
            (-recurrence.0, -recurrence.1),
        );
        let logarithm = accurate_ln(x);
        return dd_add(gamma_one_plus, (-logarithm.0, -logarithm.1));
    }

    let mut shifted = (x, 0.0);
    let mut recurrence = (0.0, 0.0);
    while shifted.0 < STIRLING_MIN {
        recurrence = dd_add(recurrence, accurate_ln_dd(shifted));
        shifted = dd_add(shifted, (1.0, 0.0));
    }
    let result = ln_gamma_stirling_parts(shifted);
    dd_add(result, (-recurrence.0, -recurrence.1))
}

fn ln_gamma_fast_accurate(x: f64) -> f64 {
    if x <= 0.125 {
        ln_gamma_one_plus_series(x) - x.ln()
    } else {
        ln_gamma_stable(x)
    }
}

fn ln_gamma_delta_parts(base: f64, delta: f64) -> (f64, f64) {
    let base_log = accurate_ln(base);
    let ratio = dd_div_f64((delta, 0.0), base);
    let log_ratio = accurate_ln_one_plus_dd(ratio);
    let mut result = dd_mul((delta, 0.0), base_log);
    result = dd_add(result, dd_mul((base, 0.0), log_ratio));
    result = dd_add(result, dd_mul((delta - 0.5, 0.0), log_ratio));
    result = dd_add(result, (-delta, 0.0));
    result = dd_add(result, (stirling_correction(base + delta), 0.0));
    dd_add(result, (-stirling_correction(base), 0.0))
}

fn ln_beta_accurate_parts(a: f64, b: f64) -> (f64, f64) {
    let smaller = a.min(b);
    let larger = a.max(b);
    if larger >= STIRLING_MIN && (smaller < STIRLING_MIN || smaller <= 0.25 * larger) {
        let gamma = ln_gamma_accurate_parts(smaller);
        let delta = ln_gamma_delta_parts(larger, smaller);
        return dd_add(gamma, (-delta.0, -delta.1));
    }
    if a + b == f64::INFINITY {
        return (ln_beta_stable(a, b), 0.0);
    }
    let gamma_a = ln_gamma_accurate_parts(a);
    let gamma_b = ln_gamma_accurate_parts(b);
    let gamma_sum = ln_gamma_accurate_parts(a + b);
    dd_add(dd_add(gamma_a, gamma_b), (-gamma_sum.0, -gamma_sum.1))
}

fn ln_beta_stable_parts(a: f64, b: f64) -> (f64, f64) {
    let smaller = a.min(b);
    let larger = a.max(b);
    if larger >= STIRLING_MIN && (smaller < STIRLING_MIN || smaller <= 0.25 * larger) {
        ln_beta_accurate_parts(a, b)
    } else {
        (ln_beta_stable(a, b), 0.0)
    }
}

fn imbalanced_ln_beta(a: f64, b: f64) -> Option<f64> {
    let smaller = a.min(b);
    let larger = a.max(b);
    if larger >= STIRLING_MIN && smaller < STIRLING_MIN {
        Some(ln_gamma_stable(smaller) - ln_gamma_delta(larger, smaller))
    } else if smaller <= 1e-8 * larger {
        Some(ln_gamma_stable(smaller) - smaller * gamma::digamma(larger))
    } else {
        None
    }
}

fn log1pmx(x: f64) -> f64 {
    if x.abs() > 0.01 {
        return x.ln_1p() - x;
    }

    let mut term = -0.5 * x * x;
    let mut sum = term;
    for n in 3..=64 {
        term *= -x * f64::from(n - 1) / f64::from(n);
        sum += term;
    }
    sum
}

fn two_sum(left: f64, right: f64) -> (f64, f64) {
    let sum = left + right;
    let virtual_right = sum - left;
    let error = (left - (sum - virtual_right)) + (right - virtual_right);
    (sum, error)
}

fn dd_add((left, left_error): (f64, f64), (right, right_error): (f64, f64)) -> (f64, f64) {
    let (sum, error) = two_sum(left, right);
    two_sum(sum, error + left_error + right_error)
}

fn dd_mul((left, left_error): (f64, f64), (right, right_error): (f64, f64)) -> (f64, f64) {
    let product = left * right;
    let error = left.mul_add(right, -product)
        + left * right_error
        + left_error * right
        + left_error * right_error;
    two_sum(product, error)
}

fn dd_div_f64((numerator, numerator_error): (f64, f64), denominator: f64) -> (f64, f64) {
    let quotient = numerator / denominator;
    let remainder = (-quotient).mul_add(denominator, numerator) + numerator_error;
    two_sum(quotient, remainder / denominator)
}

fn dd_div(numerator: (f64, f64), denominator: (f64, f64)) -> (f64, f64) {
    let quotient = numerator.0 / denominator.0;
    let product = dd_mul((quotient, 0.0), denominator);
    let remainder = dd_add(numerator, (-product.0, -product.1));
    dd_add(
        (quotient, 0.0),
        ((remainder.0 + remainder.1) / denominator.0, 0.0),
    )
}

fn dd_exp((value, error): (f64, f64)) -> f64 {
    let combined = value + error;
    if combined < f64::from_bits(1).ln() - core::f64::consts::LN_2 {
        return 0.0;
    }
    let exponential = value.exp();
    let error_expm1 = error.exp_m1();
    if exponential == 0.0 || !error_expm1.is_finite() {
        return combined.exp();
    }
    exponential.mul_add(error_expm1, exponential)
}

fn dd_negative_expm1((value, error): (f64, f64)) -> f64 {
    let combined = value + error;
    if combined < f64::from_bits(1).ln() - core::f64::consts::LN_2 {
        return 1.0;
    }
    let exponential = value.exp();
    let error_expm1 = error.exp_m1();
    if exponential == 0.0 || !error_expm1.is_finite() {
        return -combined.exp_m1();
    }
    -value.exp_m1() - exponential * error_expm1
}

fn accurate_ln(value: f64) -> (f64, f64) {
    if value == 1.0 {
        return (0.0, 0.0);
    }
    let mut scaled = value;
    let mut exponent_adjustment = 0_i32;
    if scaled < f64::MIN_POSITIVE {
        scaled *= 18_014_398_509_481_984.0;
        exponent_adjustment = -54;
    }
    let value_bits = scaled.to_bits();
    let mut exponent = ((value_bits >> 52) & 0x7ff) as i32 - 1023 + exponent_adjustment;
    let mut mantissa = f64::from_bits((value_bits & 0x000f_ffff_ffff_ffff) | (1023_u64 << 52));
    if mantissa > core::f64::consts::SQRT_2 {
        mantissa *= 0.5;
        exponent += 1;
    }
    let numerator = dd_add((mantissa, 0.0), (-1.0, 0.0));
    let denominator = dd_add((mantissa, 0.0), (1.0, 0.0));
    let ratio = dd_div(numerator, denominator);
    let ratio_squared = dd_mul(ratio, ratio);
    let mut term = ratio;
    let mut sum = ratio;
    for index in 1..=24 {
        term = dd_mul(term, ratio_squared);
        sum = dd_add(sum, dd_div_f64(term, f64::from(2 * index + 1)));
    }
    let log_mantissa = dd_mul((2.0, 0.0), sum);
    let log_two = (core::f64::consts::LN_2, 2.3190468138462996e-17);
    dd_add(dd_mul((f64::from(exponent), 0.0), log_two), log_mantissa)
}

fn accurate_ln_one_minus(value: f64) -> (f64, f64) {
    accurate_ln_one_minus_dd(value)
}

fn compensated_ln(value: f64) -> (f64, f64) {
    let high = value.ln();
    let low = if value >= f64::MIN_POSITIVE && !(0.5..=2.0).contains(&value) {
        value.mul_add((-high).exp(), -1.0).ln_1p()
    } else {
        0.0
    };
    (high, low)
}

fn compensated_ln_one_minus(value: f64) -> (f64, f64) {
    if value <= 0.5 {
        ((-value).ln_1p(), 0.0)
    } else {
        let (complement, complement_error) = two_sum(1.0, -value);
        let (high, low) = compensated_ln(complement);
        (high, low + (complement_error / complement).ln_1p())
    }
}

fn beta_shape_statistics(a: f64, b: f64) -> (f64, f64, f64, f64) {
    let scale = a.max(b);
    let scaled_a = a / scale;
    let scaled_b = b / scale;
    let scaled_sum = scaled_a + scaled_b;
    let mean = scaled_a / scaled_sum;
    let complement = scaled_b / scaled_sum;
    let log_sum = scale.ln() + scaled_sum.ln();
    let root_sum = scale.sqrt() * scaled_sum.sqrt();
    (mean, complement, log_sum, root_sum)
}

fn beta_log_ratio(a: f64, b: f64, x: f64) -> (f64, f64) {
    let residual = x.mul_add(b, -((1.0 - x) * a));
    let log_ratio = a * log1pmx(residual / a) + b * log1pmx(-residual / b);
    (residual, log_ratio)
}

fn beta_reg_asymptotic(a: f64, b: f64, x: f64) -> Option<f64> {
    let (mean, complement, _, root_sum) = beta_shape_statistics(a, b);
    if root_sum < ASYMPTOTIC_MIN_SUM.sqrt() {
        return None;
    }

    if mean.min(complement) < 0.1 && a.min(b) < ASYMPTOTIC_MIN_SHAPE {
        return None;
    }

    let (residual, log_ratio) = beta_log_ratio(a, b, x);
    let scaled_deviance = -log_ratio;
    if scaled_deviance > ASYMPTOTIC_MAX_DEVIANCE {
        if scaled_deviance > -f64::from_bits(1).ln() {
            return Some(if residual < 0.0 { 0.0 } else { 1.0 });
        }
        return None;
    }

    let scale = a.max(b);
    let delta = (residual / scale) / (a / scale + b / scale);
    let root_variance = (mean * complement).sqrt();
    let eta = if residual == 0.0 {
        0.0
    } else {
        ((2.0 * scaled_deviance).sqrt() / root_sum).copysign(residual)
    };
    let c0 = if residual.abs() < 1e-4 * a.min(b) {
        let variance = mean * complement;
        (1.0 - 2.0 * mean) / (3.0 * root_variance)
            + (variance - 1.0) * (delta / variance) / (12.0 * root_variance)
    } else {
        1.0 / eta - a.sqrt() * b.sqrt() / residual
    };
    let normal_argument = -scaled_deviance.sqrt().copysign(residual);
    let leading = if normal_argument == 0.0 {
        0.5
    } else {
        let tail = 0.5 * gamma::gamma_ur(0.5, normal_argument * normal_argument);
        if normal_argument > 0.0 {
            tail
        } else {
            1.0 - tail
        }
    };
    let correction = (-scaled_deviance).exp() * c0 / (consts::SQRT_2PI * root_sum);
    let result = leading + correction;
    if (0.0..=1.0).contains(&result) {
        Some(result)
    } else {
        None
    }
}

fn beta_reg_central_log_power_parts(a: f64, b: f64, x: f64) -> Option<(f64, f64)> {
    if a >= STIRLING_MIN && b >= STIRLING_MIN && 1.0 - x < 1.0 {
        let (residual, log_ratio) = beta_log_ratio(a, b, x);
        if residual.abs() <= 0.1 * a.min(b) {
            let (_, _, log_sum, _) = beta_shape_statistics(a, b);
            let log_scale = consts::LN_SQRT_2PI
                + 0.5 * (log_sum - a.ln() - b.ln())
                + stirling_correction(a)
                + stirling_correction(b)
                - stirling_correction_log(log_sum);
            return Some(two_sum(log_ratio, -log_scale));
        }
    }
    None
}

fn beta_reg_log_power_parts_with_log_x(
    a: f64,
    b: f64,
    (log_x, log_x_error): (f64, f64),
    (log_y, log_y_error): (f64, f64),
    (log_beta, log_beta_error): (f64, f64),
) -> (f64, f64) {
    let a_log_x = a * log_x;
    let a_log_x_error = a.mul_add(log_x, -a_log_x) + a * log_x_error;
    let b_log_y = b * log_y;
    let b_log_y_error = b.mul_add(log_y, -b_log_y) + b * log_y_error;
    let (variable, variable_error) = two_sum(a_log_x, b_log_y);
    let variable_error = variable_error + a_log_x_error + b_log_y_error;
    let (result, result_error) = two_sum(variable, -log_beta);
    (result, result_error + variable_error - log_beta_error)
}

fn beta_reg_log_power_parts(a: f64, b: f64, x: f64) -> (f64, f64) {
    beta_reg_central_log_power_parts(a, b, x).unwrap_or_else(|| {
        let smaller = a.min(b);
        let larger = a.max(b);
        if larger >= STIRLING_MIN && (smaller < STIRLING_MIN || smaller <= 0.25 * larger) {
            return beta_reg_log_power_parts_with_log_x(
                a,
                b,
                accurate_ln(x),
                accurate_ln_one_minus(x),
                ln_beta_accurate_parts(a, b),
            );
        }
        beta_reg_log_power_parts_with_log_x(
            a,
            b,
            compensated_ln(x),
            compensated_ln_one_minus(x),
            ln_beta_stable_parts(a, b),
        )
    })
}

fn beta_reg_log_power_parts_with_log_beta(
    a: f64,
    b: f64,
    x: f64,
    log_beta: (f64, f64),
) -> (f64, f64) {
    beta_reg_central_log_power_parts(a, b, x).unwrap_or_else(|| {
        let smaller = a.min(b);
        let larger = a.max(b);
        if larger >= STIRLING_MIN && (smaller < STIRLING_MIN || smaller <= 0.25 * larger) {
            beta_reg_log_power_parts_with_log_x(
                a,
                b,
                accurate_ln(x),
                accurate_ln_one_minus(x),
                log_beta,
            )
        } else {
            beta_reg_log_power_parts_with_log_x(
                a,
                b,
                compensated_ln(x),
                compensated_ln_one_minus(x),
                log_beta,
            )
        }
    })
}

fn beta_reg_log_power_parts_accurate(a: f64, b: f64, x: f64) -> (f64, f64) {
    beta_reg_log_power_parts_accurate_with_log_beta(a, b, x, ln_beta_accurate_parts(a, b))
}

fn beta_reg_log_power_parts_accurate_with_log_beta(
    a: f64,
    b: f64,
    x: f64,
    log_beta: (f64, f64),
) -> (f64, f64) {
    beta_reg_central_log_power_parts(a, b, x).unwrap_or_else(|| {
        beta_reg_log_power_parts_with_log_x(
            a,
            b,
            accurate_ln(x),
            accurate_ln_one_minus(x),
            log_beta,
        )
    })
}

fn beta_continued_fraction(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    let y = 1.0 - x;
    let tiny = 16.0 * f64::MIN_POSITIVE;
    let mut fraction = a * (a * y - b * x + 1.0) / (a + 1.0);
    if fraction == 0.0 {
        fraction = tiny;
    }
    let mut c = fraction;
    let mut d = 0.0;

    for m in 1..=MAX_BETA_REG_ITERATIONS {
        let m = f64::from(m);
        let denominator = a + 2.0 * m - 1.0;
        let numerator =
            (m * (a + m - 1.0) / denominator) * ((a + b + m - 1.0) / denominator) * (b - m) * x * x;
        let denominator_term = m
            + m * (b - m) * x / denominator
            + (a + m) * (a * y - b * x + 1.0 + m * (2.0 - x)) / (a + 2.0 * m + 1.0);

        d = denominator_term + numerator * d;
        if d == 0.0 {
            d = tiny;
        }
        c = denominator_term + numerator / c;
        if c == 0.0 {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = c * d;
        fraction *= delta;

        if (delta - 1.0).abs() <= prec::F64_PREC {
            return Ok(fraction);
        }
    }

    Err(BetaFuncError::ConvergenceFailed)
}

fn beta_continued_fraction_dd(a: f64, b: f64, x: (f64, f64)) -> Result<(f64, f64), BetaFuncError> {
    let y = dd_add((1.0, 0.0), (-x.0, -x.1));
    let mut residual = dd_mul((a, 0.0), y);
    residual = dd_add(residual, dd_mul((-b, 0.0), x));
    residual = dd_add(residual, (1.0, 0.0));
    let mut fraction = dd_div_f64(dd_mul((a, 0.0), residual), a + 1.0);
    let mut c = fraction;
    let mut d = (0.0, 0.0);

    for integer in 1..=MAX_BETA_REG_ITERATIONS {
        let m = f64::from(integer);
        let denominator = a + 2.0 * m - 1.0;
        let mut numerator = dd_div_f64(dd_mul((m, 0.0), (a + m - 1.0, 0.0)), denominator);
        let a_plus_b_plus_m_minus_one = dd_add((b, 0.0), dd_add((a, 0.0), (m - 1.0, 0.0)));
        numerator = dd_mul(
            numerator,
            dd_div_f64(dd_mul(a_plus_b_plus_m_minus_one, x), denominator),
        );
        let b_minus_m = dd_add((b, 0.0), (-m, 0.0));
        numerator = dd_mul(numerator, dd_mul(b_minus_m, x));

        let first = dd_div_f64(dd_mul((m, 0.0), dd_mul(b_minus_m, x)), denominator);
        let inner = dd_add(residual, dd_mul((m, 0.0), dd_add((2.0, 0.0), (-x.0, -x.1))));
        let second = dd_div_f64(dd_mul((a + m, 0.0), inner), a + 2.0 * m + 1.0);
        let denominator_term = dd_add((m, 0.0), dd_add(first, second));

        d = dd_div((1.0, 0.0), dd_add(denominator_term, dd_mul(numerator, d)));
        c = dd_add(denominator_term, dd_div(numerator, c));
        let delta = dd_mul(c, d);
        fraction = dd_mul(fraction, delta);
        let convergence = dd_add(delta, (-1.0, 0.0));
        if (convergence.0 + convergence.1).abs() <= f64::EPSILON {
            return Ok(fraction);
        }
    }

    Err(BetaFuncError::ConvergenceFailed)
}

fn selected_beta_continued_fraction(a: f64, b: f64, x: f64) -> Result<(f64, f64), BetaFuncError> {
    if x <= f64::EPSILON {
        beta_continued_fraction_dd(a, b, (x, 0.0))
    } else {
        beta_continued_fraction(a, b, x).map(|fraction| (fraction, 0.0))
    }
}

fn use_exact_complement_continued_fraction(a: f64, b: f64, symm_transform: bool) -> bool {
    symm_transform && a >= 1.0 && b >= 2.0 * (a + 1.0)
}

fn beta_fraction_for_transformed_tail(
    a: f64,
    b: f64,
    x: f64,
    transformed_a: f64,
    transformed_b: f64,
    transformed_x: f64,
    symm_transform: bool,
) -> Result<(f64, f64), BetaFuncError> {
    if use_exact_complement_continued_fraction(a, b, symm_transform) {
        beta_continued_fraction_dd(transformed_a, transformed_b, two_sum(1.0, -x))
    } else {
        selected_beta_continued_fraction(transformed_a, transformed_b, transformed_x)
    }
}

fn beta_power_series_log_parts_with_log_beta(
    a: f64,
    b: f64,
    x: f64,
    log_beta: Option<(f64, f64)>,
) -> Result<(f64, f64), BetaFuncError> {
    let scaled_b = b * x;
    let scaled_b = (scaled_b, b.mul_add(x, -scaled_b));
    let a_minus_one = dd_add((a, 0.0), (-1.0, 0.0));
    let mut term = (1.0_f64, 0.0_f64);
    let mut sum = (1.0_f64, 0.0_f64);
    for n in 1..=MAX_BETA_REG_ITERATIONS {
        let n = f64::from(n);
        let shape_numerator = dd_add(a_minus_one, (n, 0.0));
        let scaled_numerator = dd_mul(shape_numerator, (x, 0.0));
        let factor = dd_div_f64(dd_add(scaled_numerator, scaled_b), a + n);
        term = dd_mul(term, factor);
        sum = dd_add(sum, term);
        if term.0.abs() <= f64::EPSILON * f64::EPSILON * sum.0.abs() {
            if sum.0 <= 0.0 {
                return Err(BetaFuncError::ConvergenceFailed);
            }
            let (log_sum, log_sum_error) = accurate_ln(sum.0);
            let log_sum_error = log_sum_error + (sum.1 / sum.0).ln_1p();
            if use_beta_gamma_limit(a, b, scaled_b.0) {
                let (log_scaled_b, log_scaled_b_error) = accurate_ln(scaled_b.0);
                let log_scaled_b_error = log_scaled_b_error + (scaled_b.1 / scaled_b.0).ln_1p();
                let mut result = dd_mul((a, 0.0), (log_scaled_b, log_scaled_b_error));
                result = dd_add(result, (-scaled_b.0, -scaled_b.1));
                let log_gamma = if a <= 1e-4 {
                    a * ln_gamma_one_plus_over_x(a)
                } else {
                    gamma::ln_gamma(1.0 + a)
                };
                result = dd_add(result, (-log_gamma, 0.0));
                return Ok(dd_add(result, (log_sum, log_sum_error)));
            }
            let (log_power, log_power_error) = if let Some(log_beta) = log_beta {
                beta_reg_log_power_parts_accurate_with_log_beta(a, b, x, log_beta)
            } else {
                beta_reg_log_power_parts_accurate(a, b, x)
            };
            let (variable, variable_error) = two_sum(log_power, log_sum);
            let log_a = accurate_ln(a);
            return Ok(dd_add(
                (variable, variable_error + log_power_error + log_sum_error),
                (-log_a.0, -log_a.1),
            ));
        }
    }
    Err(BetaFuncError::ConvergenceFailed)
}

fn beta_power_series_log_parts(a: f64, b: f64, x: f64) -> Result<(f64, f64), BetaFuncError> {
    beta_power_series_log_parts_with_log_beta(a, b, x, None)
}

fn beta_power_series_log(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    beta_power_series_log_parts(a, b, x).map(|(result, error)| result + error)
}

fn beta_small_shapes_series_log(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
) -> Result<Option<(f64, bool)>, BetaFuncError> {
    beta_small_shapes_series_log_with_log_beta(a, b, x, y, None)
}

fn beta_small_shapes_series_log_with_log_beta(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
    log_beta: Option<(f64, f64)>,
) -> Result<Option<(f64, bool)>, BetaFuncError> {
    if a.max(b) > 1.0 {
        return Ok(None);
    }
    let invert = !(a >= 0.2_f64.min(b) || x.powf(a) <= 0.9);
    let (transformed_a, transformed_b, transformed_x) = if invert { (b, a, y) } else { (a, b, x) };
    if transformed_x > 0.9 {
        return Ok(None);
    }
    beta_power_series_log_parts_with_log_beta(transformed_a, transformed_b, transformed_x, log_beta)
        .map(|result| Some((result.0 + result.1, invert)))
}

fn use_beta_gamma_limit(a: f64, b: f64, scaled_x: f64) -> bool {
    let correction_scale = a + scaled_x + 1.0;
    correction_scale.is_finite() && correction_scale / b.sqrt() <= 0.25 * f64::EPSILON.sqrt()
}

fn use_beta_power_series(a: f64, b: f64, x: f64) -> bool {
    let scaled_x = b * x;
    x < 1.0
        && ((scaled_x <= 0.7 && x <= 0.95)
            || (a <= f64::EPSILON.sqrt() && scaled_x <= 2.0 && x < beta_symmetry_split(a, b))
            || (a <= 0.3 && b >= 32.0 && scaled_x <= 2.0)
            || (a <= 40.0 && b >= 32.0 && x < beta_symmetry_split(a, b))
            || (use_beta_gamma_limit(a, b, scaled_x) && scaled_x <= 64.0))
}

fn use_beta_power_series_before_symmetry(a: f64, b: f64, x: f64) -> bool {
    let scaled_x = b * x;
    x < 1.0
        && !(a <= f64::EPSILON.sqrt() && b >= STIRLING_MIN && x.powf(a) > 0.5)
        && ((a <= f64::EPSILON.sqrt() && scaled_x <= 2.0 && x < beta_symmetry_split(a, b))
            || (a <= 0.3 && b >= 32.0 && scaled_x <= 2.0)
            || (a <= 40.0 && b >= 32.0 && x < beta_symmetry_split(a, b))
            || (use_beta_gamma_limit(a, b, scaled_x) && scaled_x <= 64.0))
}

fn beta_symmetry_split(a: f64, b: f64) -> f64 {
    let a1 = a + 1.0;
    let b1 = b + 1.0;
    let scale = a1.max(b1);
    (a1 / scale) / (a1 / scale + b1 / scale)
}

fn use_beta_symmetry(a: f64, b: f64, x: f64) -> bool {
    a < 1.0 && a <= f64::EPSILON.sqrt() && b >= STIRLING_MIN && x.powf(a) > 0.5
        || (a < 1.0 || x > f64::EPSILON) && 1.0 - x < 1.0 && x >= beta_symmetry_split(a, b)
}

fn beta_concentrated_quantile(a: f64, b: f64, probability: f64) -> Option<f64> {
    if a.min(b) < ASYMPTOTIC_MIN_SHAPE {
        return None;
    }
    let (mean, complement, _, root_sum) = beta_shape_statistics(a, b);
    if mean.min(complement) < 0.1 {
        return None;
    }
    let lower_spacing = mean - f64::from_bits(mean.to_bits() - 1);
    let upper_spacing = f64::from_bits(mean.to_bits() + 1) - mean;
    let standard_deviation = (mean * complement).sqrt() / root_sum;
    if 64.0 * standard_deviation < 0.5 * lower_spacing.min(upper_spacing) {
        let scale = a.max(b);
        let scaled_a = a / scale;
        let scaled_b = b / scale;
        let scaled_sum = scaled_a + scaled_b;
        let scaled_a_error = (-scaled_a).mul_add(scale, a) / scale;
        let scaled_b_error = (-scaled_b).mul_add(scale, b) / scale;
        let virtual_scaled_b = scaled_sum - scaled_a;
        let scaled_sum_error = (scaled_a - (scaled_sum - virtual_scaled_b))
            + (scaled_b - virtual_scaled_b)
            + scaled_a_error
            + scaled_b_error;
        let product = mean * scaled_sum;
        let product_error = mean.mul_add(scaled_sum, -product);
        let difference = scaled_a - product;
        let virtual_product = difference - scaled_a;
        let difference_error =
            (scaled_a - (difference - virtual_product)) + (-product - virtual_product);
        let mean_residual = difference
            + (difference_error + scaled_a_error - product_error - mean * scaled_sum_error);
        let mean_correction = mean_residual / scaled_sum;
        let normal_quantile = -core::f64::consts::SQRT_2 * erf::erfc_inv(2.0 * probability);
        let reciprocal_sum = (1.0 / root_sum) / root_sum;
        let skew_correction =
            (complement - mean) * normal_quantile.mul_add(normal_quantile, -1.0) * reciprocal_sum
                / 3.0;
        let offset = normal_quantile.mul_add(standard_deviation, mean_correction + skew_correction);
        Some(mean + offset)
    } else {
        None
    }
}

fn beta_a_step(a: f64, b: f64, x: f64, steps: usize) -> f64 {
    let power = beta_reg_log_power_parts(a, b, x);
    (power.0 + power.1 + beta_a_step_log_sum(a, b, x, steps) - a.ln()).exp()
}

fn beta_a_step_log_sum(a: f64, b: f64, x: f64, steps: usize) -> f64 {
    let mut log_sum = 0.0_f64;
    let mut log_term = 0.0_f64;
    let log_x = x.ln();
    for i in 0..steps.saturating_sub(1) {
        let i = i as f64;
        log_term += (a + b + i).ln() + log_x - (a + i + 1.0).ln();
        let maximum = log_sum.max(log_term);
        log_sum = maximum + (log_sum.min(log_term) - maximum).exp().ln_1p();
    }
    log_sum
}

fn beta_a_step_log(a: f64, b: f64, x: f64, steps: usize, log_beta: (f64, f64)) -> f64 {
    let power = beta_reg_log_power_parts_accurate_with_log_beta(a, b, x, log_beta);
    let log_a = accurate_ln(a);
    let result = dd_add(
        dd_add(power, (beta_a_step_log_sum(a, b, x, steps), 0.0)),
        (-log_a.0, -log_a.1),
    );
    result.0 + result.1
}

fn upper_gamma_scaled_asymptotic(shape: f64, x: f64) -> Result<f64, BetaFuncError> {
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    for n in 1..=64 {
        term *= (shape - f64::from(n)) / x;
        sum += term;
        if term.abs() <= prec::F64_PREC * sum.abs() {
            return Ok(sum / x);
        }
    }
    Err(BetaFuncError::ConvergenceFailed)
}

fn upper_gamma_scaled_continued_fraction(shape: f64, x: f64) -> Result<f64, BetaFuncError> {
    const BIG: f64 = 4_503_599_627_370_496.0;
    const BIG_INVERSE: f64 = 2.220446049250313e-16;

    let mut y = 1.0 - shape;
    let mut z = x + y + 1.0;
    let mut c = 0.0;
    let mut pkm2 = 1.0;
    let mut qkm2 = x;
    let mut pkm1 = x + 1.0;
    let mut qkm1 = z * x;
    let mut result = pkm1 / qkm1;
    for _ in 0..256 {
        y += 1.0;
        z += 2.0;
        c += 1.0;
        let yc = y * c;
        let pk = pkm1 * z - pkm2 * yc;
        let qk = qkm1 * z - qkm2 * yc;

        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if pk.abs() > BIG {
            pkm2 *= BIG_INVERSE;
            pkm1 *= BIG_INVERSE;
            qkm2 *= BIG_INVERSE;
            qkm1 *= BIG_INVERSE;
        }

        if qk != 0.0 {
            let next = pk / qk;
            let relative_change = ((result - next) / next).abs();
            result = next;
            if relative_change <= 4.0 * prec::F64_PREC {
                return if result > 0.0 && result.is_finite() {
                    Ok(result)
                } else {
                    Err(BetaFuncError::ConvergenceFailed)
                };
            }
        }
    }
    Err(BetaFuncError::ConvergenceFailed)
}

fn expm1c(x: f64) -> f64 {
    if x.abs() < 1e-5 {
        1.0 + x * (0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0 + x / 120.0)))
    } else {
        x.exp_m1() / x
    }
}

fn ln_gamma_one_plus_over_x(x: f64) -> f64 {
    if x <= 1e-4 {
        -consts::EULER_MASCHERONI
            + x * (0.8224670334241132
                + x * (-0.40068563438653143
                    + x * (0.27058080842778455
                        + x * (-0.20738555102867398 + x * 0.1695571769974082))))
    } else {
        gamma::ln_gamma(1.0 + x) / x
    }
}

fn upper_gamma_scaled_small_shape(shape: f64, x: f64) -> Result<f64, BetaFuncError> {
    let log_x = x.ln();
    let log_gamma_ratio = ln_gamma_one_plus_over_x(shape);
    let difference = log_x - log_gamma_ratio;
    let scaled_difference = shape * difference;
    let mut term = -x / (shape + 1.0);
    let mut sum = term;
    let mut compensation = 0.0_f64;
    for n in 2..=128 {
        let n = f64::from(n);
        term *= (-x / n) * (shape + n - 1.0) / (shape + n);
        let corrected = term - compensation;
        let next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
        if term.abs() <= prec::F64_PREC * sum.abs() {
            let upper_gamma =
                -difference * expm1c(scaled_difference) - scaled_difference.exp() * sum;
            let result = upper_gamma * (x - scaled_difference).exp();
            return if result > 0.0 && result.is_finite() {
                Ok(result)
            } else {
                Err(BetaFuncError::ConvergenceFailed)
            };
        }
    }
    Err(BetaFuncError::ConvergenceFailed)
}

fn beta_small_b_large_a_factor(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
) -> Result<(f64, f64), BetaFuncError> {
    let bm1 = b - 1.0;
    let t = a + 0.5 * bm1;
    let lx = if y < 0.35 { (-y).ln_1p() } else { x.ln() };
    let u = -t * lx;
    let log_h = b * u.ln() - u - ln_gamma_stable(b);
    let log_prefix = log_h + ln_gamma_delta(a, b) - b * t.ln();

    let mut odd_factorials = [1.0; 30];
    let mut factorial = 1.0;
    for k in 1..=59 {
        factorial *= k as f64;
        if k >= 3 && k % 2 == 1 {
            odd_factorials[(k - 3) as usize / 2] = factorial;
        }
    }

    let mut coefficients = [0.0; 30];
    coefficients[0] = 1.0;
    let mut j = if u >= SCALED_GAMMA_MIN_X {
        upper_gamma_scaled_asymptotic(b, u)?
    } else if u > 1.0 {
        upper_gamma_scaled_continued_fraction(b, u)?
    } else if b <= 1e-4 && u <= 1.0 {
        upper_gamma_scaled_small_shape(b, u)?
    } else {
        gamma::gamma_ur(b, u) / log_h.exp()
    };
    let mut sum = j;
    let mut compensation = 0.0_f64;
    let lx2 = (0.5 * lx) * (0.5 * lx);
    let mut lx_power = 1.0;
    let t4 = 4.0 * t * t;
    let mut b_plus_2n = b;
    let mut converged = false;

    for n in 1..30 {
        let n_f64 = n as f64;
        let mut coefficient = 0.0;
        for m in 1..n {
            coefficient += (m as f64 * b - n_f64) * coefficients[n - m] / odd_factorials[m - 1];
        }
        coefficient /= n_f64;
        coefficient += bm1 / odd_factorials[n - 1];
        coefficients[n] = coefficient;

        j = (b_plus_2n * (b_plus_2n + 1.0) * j + (u + b_plus_2n + 1.0) * lx_power) / t4;
        lx_power *= lx2;
        b_plus_2n += 2.0;
        let term = coefficient * j;
        let corrected = term - compensation;
        let next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
        if term.abs() <= prec::F64_PREC * sum.abs() {
            converged = true;
            break;
        }
    }

    if converged && sum > 0.0 {
        Ok((log_prefix, sum))
    } else {
        Err(BetaFuncError::ConvergenceFailed)
    }
}

fn beta_small_b_large_a_series(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
    initial: f64,
) -> Result<f64, BetaFuncError> {
    let (log_prefix, factor) = beta_small_b_large_a_factor(a, b, x, y)?;
    let sum = initial + log_prefix.exp() * factor;
    if (0.0..=1.0).contains(&sum) {
        Ok(sum)
    } else {
        Err(BetaFuncError::ConvergenceFailed)
    }
}

fn beta_small_b_large_a_series_log(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
    initial: f64,
) -> Result<f64, BetaFuncError> {
    let (log_prefix, factor) = beta_small_b_large_a_factor(a, b, x, y)?;
    let tail = log_prefix + factor.ln();
    if initial == 0.0 {
        Ok(tail)
    } else {
        let initial = initial.ln();
        let maximum = initial.max(tail);
        Ok(maximum + (initial.min(tail) - maximum).exp().ln_1p())
    }
}

fn beta_reg_small_b_shifted_log(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
    log_beta: (f64, f64),
) -> Result<f64, BetaFuncError> {
    let steps = (10.0 - a).ceil() as usize;
    let shifted = a + steps as f64;
    let shifted_log = beta_small_b_large_a_series_log(shifted, b, x, y, 0.0)?;
    let recurrence_log = beta_a_step_log(a, b, x, steps, log_beta);
    let maximum = shifted_log.max(recurrence_log);
    Ok(maximum + (shifted_log.min(recurrence_log) - maximum).exp().ln_1p())
}

fn beta_reg_small_b_large_a(a: f64, b: f64, x: f64, y: f64) -> Result<Option<f64>, BetaFuncError> {
    if a < 10.0 || b >= 40.0 || y >= 0.3 {
        return Ok(None);
    }
    let mut steps = b.floor() as usize;
    if b == steps as f64 {
        steps -= 1;
    }
    let reduced_b = b - steps as f64;
    let initial = if steps == 0 {
        0.0
    } else {
        beta_a_step(reduced_b, a, y, steps)
    };
    beta_small_b_large_a_series(a, reduced_b, x, y, initial).map(Some)
}

fn beta_reg_small_b_large_a_log(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
) -> Result<Option<f64>, BetaFuncError> {
    if a < 10.0 || b >= 40.0 || y >= 0.3 {
        return Ok(None);
    }
    let mut steps = b.floor() as usize;
    if b == steps as f64 {
        steps -= 1;
    }
    let reduced_b = b - steps as f64;
    let initial = if steps == 0 {
        0.0
    } else {
        beta_a_step(reduced_b, a, y, steps)
    };
    beta_small_b_large_a_series_log(a, reduced_b, x, y, initial).map(Some)
}

/// Computes the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter.
///
///
/// # Panics
///
/// if `a <= 0.0` or `b <= 0.0`
pub fn beta(a: f64, b: f64) -> f64 {
    checked_beta(a, b).unwrap()
}

/// Computes the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter.
///
///
/// # Errors
///
/// if `a <= 0.0` or `b <= 0.0`
pub fn checked_beta(a: f64, b: f64) -> Result<f64, BetaFuncError> {
    checked_ln_beta(a, b).map(|x| x.exp())
}

/// Computes the lower incomplete (unregularized) beta function
/// `B(a,b,x) = int(t^(a-1)*(1-t)^(b-1),t=0..x)` for `a > 0, b > 0, 1 >= x >= 0`
/// where `a` is the first beta parameter, `b` is the second beta parameter, and
/// `x` is the upper limit of the integral
///
/// # Panics
///
/// If `a <= 0.0`, `b <= 0.0`, `x < 0.0`, or `x > 1.0`
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    checked_beta_inc(a, b, x).unwrap()
}

/// Computes the lower incomplete (unregularized) beta function
/// `B(a,b,x) = int(t^(a-1)*(1-t)^(b-1),t=0..x)` for `a > 0, b > 0, 1 >= x >= 0`
/// where `a` is the first beta parameter, `b` is the second beta parameter, and
/// `x` is the upper limit of the integral
///
/// # Errors
///
/// If `a <= 0.0`, `b <= 0.0`, `x < 0.0`, or `x > 1.0`
pub fn checked_beta_inc(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    checked_beta_reg(a, b, x).and_then(|x| checked_beta(a, b).map(|y| x * y))
}

/// Computes the regularized lower incomplete beta function
/// `I_x(a,b) = 1/Beta(a,b) * int(t^(a-1)*(1-t)^(b-1), t=0..x)`
/// `a > 0`, `b > 0`, `1 >= x >= 0` where `a` is the first beta parameter,
/// `b` is the second beta parameter, and `x` is the upper limit of the
/// integral.
///
/// # Panics
///
/// if `a <= 0.0`, `b <= 0.0`, `x < 0.0`, or `x > 1.0`
pub fn beta_reg(a: f64, b: f64, x: f64) -> f64 {
    checked_beta_reg(a, b, x).unwrap()
}

/// Computes the regularized lower incomplete beta function
/// `I_x(a,b) = 1/Beta(a,b) * int(t^(a-1)*(1-t)^(b-1), t=0..x)`
/// `a > 0`, `b > 0`, `1 >= x >= 0` where `a` is the first beta parameter,
/// `b` is the second beta parameter, and `x` is the upper limit of the
/// integral.
///
/// # Errors
///
/// if `a <= 0.0`, `b <= 0.0`, `x < 0.0`, or `x > 1.0`
pub fn checked_beta_reg(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    if a <= 0.0 {
        return Err(BetaFuncError::ANotGreaterThanZero);
    }

    if b <= 0.0 {
        return Err(BetaFuncError::BNotGreaterThanZero);
    }

    if !(0.0..=1.0).contains(&x) {
        return Err(BetaFuncError::XOutOfRange);
    }

    if x == 0.0 {
        return Ok(0.0);
    }
    if x == 1.0 {
        return Ok(1.0);
    }
    if a == b && x == 0.5 {
        return Ok(0.5);
    }
    if b == 1.0 {
        return Ok(x.powf(a));
    }
    if a == 1.0 {
        return Ok(-(b * (-x).ln_1p()).exp_m1());
    }
    let y = 1.0 - x;
    if let Some((log_result, invert)) = beta_small_shapes_series_log(a, b, x, y)? {
        let result = if invert {
            -log_result.exp_m1()
        } else {
            log_result.exp()
        };
        return if (0.0..=1.0).contains(&result) {
            Ok(result)
        } else {
            Err(BetaFuncError::ConvergenceFailed)
        };
    }
    if let Some(result) = beta_reg_asymptotic(a, b, x) {
        return Ok(result);
    }
    if a.mul_add(y, -(b * x)) >= 0.0
        && let Some(result) = beta_reg_small_b_large_a(a, b, x, y)?
    {
        return Ok(result);
    }
    if (1.0..10.0).contains(&a) && b < 1.0 && y < 0.3 {
        let result = beta_reg_small_b_shifted_log(a, b, x, y, ln_beta_accurate_parts(a, b))?.exp();
        return if (0.0..=1.0).contains(&result) {
            Ok(result)
        } else {
            Err(BetaFuncError::ConvergenceFailed)
        };
    }
    let symm_transform =
        !use_beta_power_series_before_symmetry(a, b, x) && use_beta_symmetry(a, b, x);
    let (transformed_a, transformed_b, transformed_x, transformed_y) = if symm_transform {
        (b, a, y, x)
    } else {
        (a, b, x, y)
    };
    if !use_exact_complement_continued_fraction(a, b, symm_transform)
        && let Some(tail) =
            beta_reg_small_b_large_a(transformed_a, transformed_b, transformed_x, transformed_y)?
    {
        return Ok(if symm_transform { 1.0 - tail } else { tail });
    }
    if use_beta_power_series(transformed_a, transformed_b, transformed_x) {
        let log_result = beta_power_series_log_parts(transformed_a, transformed_b, transformed_x)?;
        let result = if symm_transform {
            dd_negative_expm1(log_result)
        } else {
            (log_result.0 + log_result.1).exp()
        };
        return if (0.0..=1.0).contains(&result) {
            Ok(result)
        } else {
            Err(BetaFuncError::ConvergenceFailed)
        };
    }

    let log_power = beta_reg_log_power_parts(a, b, x);
    let power = (log_power.0 + log_power.1).exp();
    if power == 0.0 {
        return Ok(if symm_transform { 1.0 } else { 0.0 });
    }
    let fraction = beta_fraction_for_transformed_tail(
        a,
        b,
        x,
        transformed_a,
        transformed_b,
        transformed_x,
        symm_transform,
    )?;
    let accurate_fraction =
        1.0 - transformed_x == 1.0 || use_exact_complement_continued_fraction(a, b, symm_transform);
    let result = if accurate_fraction {
        let log_fraction = accurate_ln_dd(fraction);
        let log_result = dd_add(log_power, (-log_fraction.0, -log_fraction.1));
        if symm_transform {
            dd_negative_expm1(log_result)
        } else {
            dd_exp(log_result)
        }
    } else if symm_transform {
        1.0 - power / (fraction.0 + fraction.1)
    } else {
        power / (fraction.0 + fraction.1)
    };
    if (0.0..=1.0).contains(&result) {
        Ok(result)
    } else {
        Err(BetaFuncError::ConvergenceFailed)
    }
}

fn log1mexp(x: f64) -> f64 {
    if x < -core::f64::consts::LN_2 {
        (-x.exp()).ln_1p()
    } else {
        (-x.exp_m1()).ln()
    }
}

pub(crate) fn checked_ln_beta_reg(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    checked_ln_beta_reg_with_log_beta(a, b, x, None)
}

pub(crate) fn checked_ln_beta_reg_complement(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    if a <= 0.0 {
        return Err(BetaFuncError::ANotGreaterThanZero);
    }
    if b <= 0.0 {
        return Err(BetaFuncError::BNotGreaterThanZero);
    }
    if !(0.0..=1.0).contains(&x) {
        return Err(BetaFuncError::XOutOfRange);
    }
    if x == 1.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if a <= f64::EPSILON.sqrt() && b >= STIRLING_MIN && x.powf(a) > 0.5 {
        let log_cdf = checked_ln_beta_reg(a, b, x)?;
        return Ok(log1mexp(log_cdf));
    }
    if use_beta_symmetry(a, b, x) {
        let y = 1.0 - x;
        if use_beta_power_series(b, a, y) {
            return beta_power_series_log(b, a, y);
        }
    }
    let log_cdf = checked_ln_beta_reg(a, b, x)?;
    if log_cdf < -core::f64::consts::LN_2 {
        Ok(log1mexp(log_cdf))
    } else {
        checked_ln_beta_reg(b, a, 1.0 - x)
    }
}

fn checked_ln_beta_reg_with_log_beta(
    a: f64,
    b: f64,
    x: f64,
    log_beta: Option<(f64, f64)>,
) -> Result<f64, BetaFuncError> {
    if a <= 0.0 {
        return Err(BetaFuncError::ANotGreaterThanZero);
    }
    if b <= 0.0 {
        return Err(BetaFuncError::BNotGreaterThanZero);
    }
    if !(0.0..=1.0).contains(&x) {
        return Err(BetaFuncError::XOutOfRange);
    }
    if x == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if x == 1.0 {
        return Ok(0.0);
    }
    if a == b && x == 0.5 {
        return Ok(-core::f64::consts::LN_2);
    }
    if b == 1.0 {
        return Ok(a * x.ln());
    }
    if a == 1.0 {
        return Ok((-(b * (-x).ln_1p()).exp_m1()).ln());
    }
    let y = 1.0 - x;
    if let Some((log_result, invert)) =
        beta_small_shapes_series_log_with_log_beta(a, b, x, y, log_beta)?
    {
        return Ok(if invert {
            log1mexp(log_result)
        } else {
            log_result
        });
    }
    if let Some(result) = beta_reg_asymptotic(a, b, x) {
        return Ok(result.ln());
    }
    if a.mul_add(y, -(b * x)) >= 0.0
        && let Some(result) = beta_reg_small_b_large_a_log(a, b, x, y)?
    {
        return Ok(result);
    }
    if (1.0..10.0).contains(&a) && b < 1.0 && y < 0.3 {
        return beta_reg_small_b_shifted_log(a, b, x, y, ln_beta_accurate_parts(a, b));
    }
    let symm_transform =
        !use_beta_power_series_before_symmetry(a, b, x) && use_beta_symmetry(a, b, x);
    let (transformed_a, transformed_b, transformed_x, transformed_y) = if symm_transform {
        (b, a, y, x)
    } else {
        (a, b, x, y)
    };
    if !use_exact_complement_continued_fraction(a, b, symm_transform)
        && let Some(log_tail) = beta_reg_small_b_large_a_log(
            transformed_a,
            transformed_b,
            transformed_x,
            transformed_y,
        )?
    {
        return Ok(if symm_transform {
            log1mexp(log_tail)
        } else {
            log_tail
        });
    }
    if use_beta_power_series(transformed_a, transformed_b, transformed_x) {
        let log_result = beta_power_series_log_parts_with_log_beta(
            transformed_a,
            transformed_b,
            transformed_x,
            log_beta,
        )?;
        let log_result = log_result.0 + log_result.1;
        return Ok(if symm_transform {
            log1mexp(log_result)
        } else {
            log_result
        });
    }

    let log_power = if let Some(log_beta) = log_beta {
        beta_reg_log_power_parts_with_log_beta(a, b, x, log_beta)
    } else {
        beta_reg_log_power_parts(a, b, x)
    };
    if symm_transform && (log_power.0 + log_power.1).exp() == 0.0 {
        return Ok(0.0);
    }
    let fraction = beta_fraction_for_transformed_tail(
        a,
        b,
        x,
        transformed_a,
        transformed_b,
        transformed_x,
        symm_transform,
    )?;
    let smaller = a.min(b);
    let larger = a.max(b);
    let log_fraction = if fraction.1 != 0.0
        || (larger >= STIRLING_MIN && (smaller < STIRLING_MIN || smaller <= 0.25 * larger))
    {
        accurate_ln_dd(fraction)
    } else {
        (fraction.0.ln(), 0.0)
    };
    let log_result = dd_add(log_power, (-log_fraction.0, -log_fraction.1));
    let log_result = log_result.0 + log_result.1;
    if symm_transform {
        Ok(log1mexp(log_result))
    } else {
        Ok(log_result)
    }
}

fn ln_beta_stable(a: f64, b: f64) -> f64 {
    if a.min(b) <= 0.125 {
        if a.max(b) >= STIRLING_MIN {
            let result = ln_beta_accurate_parts(a, b);
            return result.0 + result.1;
        }
        if a.max(b) <= 0.125 {
            return (a + b).ln() - a.ln() - b.ln()
                + ln_gamma_one_plus_series(a)
                + ln_gamma_one_plus_series(b)
                - ln_gamma_one_plus_series(a + b);
        }
        return ln_gamma_fast_accurate(a) + ln_gamma_fast_accurate(b) - ln_gamma_stable(a + b);
    }
    if let Some(ln_beta) = imbalanced_ln_beta(a, b) {
        return ln_beta;
    }
    if a < STIRLING_MIN || b < STIRLING_MIN {
        return ln_gamma_stable(a) + ln_gamma_stable(b) - ln_gamma_stable(a + b);
    }

    let (mean, complement, log_sum, _) = beta_shape_statistics(a, b);
    a * mean.ln()
        + b * complement.ln()
        + consts::LN_SQRT_2PI
        + 0.5 * (log_sum - a.ln() - b.ln())
        + stirling_correction(a)
        + stirling_correction(b)
        - stirling_correction_log(log_sum)
}

fn lower_tail_initial(a: f64, b: f64, probability: f64, ln_beta: f64) -> (f64, f64) {
    let log_initial = (probability.ln() + a.ln() + ln_beta) / a;
    let initial = log_initial.exp();
    let initial = if initial == 0.0 {
        0.0
    } else if initial < 1.0 {
        initial
    } else {
        let (mean, _, _, _) = beta_shape_statistics(a, b);
        if mean < 1.0 {
            mean
        } else {
            f64::from_bits(1.0_f64.to_bits() - 1)
        }
    };
    (initial, log_initial)
}

fn lower_tail_initial_accurate(
    a: f64,
    probability: f64,
    log_beta: (f64, f64),
) -> (f64, (f64, f64)) {
    let mut logarithm = accurate_ln(probability);
    logarithm = dd_add(logarithm, accurate_ln(a));
    logarithm = dd_add(logarithm, log_beta);
    logarithm = dd_div_f64(logarithm, a);
    (dd_exp(logarithm), logarithm)
}

fn inverse_beta_initial(a: f64, b: f64, probability: f64, ln_beta: f64) -> (f64, f64) {
    if a > 1.0 && b > 1.0 && (probability >= 1e-4 || a.min(b) >= STIRLING_MIN) {
        let normal_tail = (-2.0 * probability.ln()).sqrt();
        let normal_quantile = normal_tail
            - (2.30753 + 0.27061 * normal_tail)
                / (1.0 + (0.99229 + 0.04481 * normal_tail) * normal_tail);
        let correction = (normal_quantile * normal_quantile - 3.0) / 6.0;
        let reciprocal_a = 1.0 / (2.0 * a - 1.0);
        let reciprocal_b = 1.0 / (2.0 * b - 1.0);
        let scale = 2.0 / (reciprocal_a + reciprocal_b);
        let w = normal_quantile * (scale + correction).sqrt() / scale
            - (reciprocal_b - reciprocal_a) * (correction + 5.0 / 6.0 - 2.0 / (3.0 * scale));
        let log_ratio = b.ln() - a.ln() + 2.0 * w;
        let initial = if log_ratio > 0.0 {
            let reciprocal = (-log_ratio).exp();
            reciprocal / (1.0 + reciprocal)
        } else {
            1.0 / (1.0 + log_ratio.exp())
        };
        if initial > 0.0 && initial < 1.0 {
            return (initial, f64::NAN);
        }
    }

    lower_tail_initial(a, b, probability, ln_beta)
}

fn inverse_beta_midpoint(lower: f64, upper: f64) -> f64 {
    let arithmetic = lower + 0.5 * (upper - lower);
    let candidate = if upper < 0.5 {
        let positive_lower = if lower == 0.0 {
            f64::from_bits(1)
        } else {
            lower
        };
        (0.5 * (positive_lower.ln() + upper.ln())).exp()
    } else if lower > 0.5 {
        let lower_complement = 1.0 - lower;
        let upper_complement = if upper == 1.0 {
            f64::from_bits(1)
        } else {
            1.0 - upper
        };
        1.0 - (0.5 * (lower_complement.ln() + upper_complement.ln())).exp()
    } else {
        arithmetic
    };
    if candidate > lower && candidate < upper {
        candidate
    } else {
        arithmetic
    }
}

fn inverse_beta_adjacent_result(lower: f64, upper: f64, lower_error: f64, upper_error: f64) -> f64 {
    if !lower_error.is_finite() {
        return upper;
    }
    let fraction = -lower_error / (upper_error - lower_error);
    if fraction < 0.5 {
        lower
    } else if fraction > 0.5 || upper.to_bits() & 1 == 0 {
        upper
    } else {
        lower
    }
}

fn inverse_beta_log_value_parts(
    a: f64,
    b: f64,
    x: f64,
    log_beta: (f64, f64),
    accurate_log_beta: Option<(f64, f64)>,
) -> Result<(f64, f64), BetaFuncError> {
    if (0.01..10.0).contains(&a) && b < 1.0 && 1.0 - x < 0.3 {
        return beta_reg_small_b_shifted_log(a, b, x, 1.0 - x, accurate_log_beta.unwrap())
            .map(|value| (value, 0.0));
    }
    if (10.0..1e15).contains(&a)
        && b < 1.0
        && 1.0 - x < 0.3
        && let Some(value) = beta_reg_small_b_large_a_log(a, b, x, 1.0 - x)?
    {
        return Ok((value, 0.0));
    }
    if use_beta_power_series(a, b, x)
        && (!use_beta_symmetry(a, b, x) || use_beta_power_series_before_symmetry(a, b, x))
    {
        beta_power_series_log_parts_with_log_beta(a, b, x, Some(log_beta))
    } else {
        checked_ln_beta_reg_with_log_beta(a, b, x, Some(log_beta)).map(|value| (value, 0.0))
    }
}

fn inverse_beta_log_tail(
    a: f64,
    b: f64,
    target: f64,
    mut current: f64,
    log_beta: (f64, f64),
    ln_beta: f64,
) -> f64 {
    const FAST_ITERATIONS: usize = 64;
    const MAX_ITERATIONS: usize = 256;

    let (log_target, log_target_correction) = accurate_ln(target);
    let mut lower = 0.0;
    let mut upper = 1.0;
    let mut lower_error = f64::NEG_INFINITY;
    let mut upper_error = -log_target - log_target_correction;
    let accurate_log_beta = if (0.01..10.0).contains(&a) && b < 1.0 {
        Some(ln_beta_accurate_parts(a, b))
    } else {
        None
    };

    for iteration in 0..MAX_ITERATIONS {
        let log_value = inverse_beta_log_value_parts(a, b, current, log_beta, accurate_log_beta)
            .unwrap_or_else(|error| {
                panic!("inv_beta_reg evaluation failed at x={current:?}: {error}")
            });
        let error_parts = dd_add(log_value, (-log_target, -log_target_correction));
        let error = error_parts.0 + error_parts.1;
        if error_parts.0 == 0.0 && error_parts.1 == 0.0 {
            return current;
        }

        if error < 0.0 {
            lower = current;
            lower_error = error;
        } else {
            upper = current;
            upper_error = error;
        }

        let midpoint = inverse_beta_midpoint(lower, upper);
        if midpoint == lower || midpoint == upper {
            return inverse_beta_adjacent_result(lower, upper, lower_error, upper_error);
        }

        let log_pdf = (a - 1.0) * current.ln() + (b - 1.0) * (-current).ln_1p() - ln_beta;
        let step = error * (log_value.0 + log_value.1 - log_pdf).exp();
        let newton = current - step;
        let next = if iteration < FAST_ITERATIONS
            && newton.is_finite()
            && ((newton > lower && newton < upper) || newton == current)
        {
            newton
        } else {
            midpoint
        };

        if next == current {
            let neighbor = if error > 0.0 {
                f64::from_bits(current.to_bits() - 1)
            } else {
                f64::from_bits(current.to_bits() + 1)
            };
            let neighbor_value =
                inverse_beta_log_value_parts(a, b, neighbor, log_beta, accurate_log_beta)
                    .unwrap_or_else(|evaluation_error| {
                        panic!("inv_beta_reg evaluation failed: {evaluation_error}")
                    });
            let neighbor_error = dd_add(neighbor_value, (-log_target, -log_target_correction));
            let neighbor_error = neighbor_error.0 + neighbor_error.1;
            if error * neighbor_error <= 0.0 {
                return if error > 0.0 {
                    inverse_beta_adjacent_result(neighbor, current, neighbor_error, error)
                } else {
                    inverse_beta_adjacent_result(current, neighbor, error, neighbor_error)
                };
            }
            current = if neighbor_error.abs() <= error.abs() {
                neighbor
            } else {
                midpoint
            };
        } else {
            current = next;
        }
    }

    panic!("inv_beta_reg did not converge for a={a}, b={b}, probability={target}")
}

fn inverse_beta_reflect(a: f64, b: f64, probability: f64, log_beta: (f64, f64)) -> bool {
    if probability <= 0.5 {
        false
    } else if a >= b {
        true
    } else {
        let midpoint_log_probability = checked_ln_beta_reg_with_log_beta(a, b, 0.5, Some(log_beta))
            .unwrap_or_else(|error| panic!("inv_beta_reg evaluation failed: {error}"));
        midpoint_log_probability < probability.ln()
    }
}

/// Computes the inverse of the regularized incomplete beta function
pub fn inv_beta_reg(a: f64, b: f64, probability: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&probability) && a > 0.0 && b > 0.0);

    if probability == 0.0 {
        return 0.0;
    }
    if probability == 1.0 {
        return 1.0;
    }
    if a == b && probability == 0.5 {
        return 0.5;
    }
    if let Some(quantile) = beta_concentrated_quantile(a, b, probability) {
        return quantile;
    }
    if b == 1.0 {
        return probability.powf(1.0 / a);
    }
    if a == 1.0 {
        return -((-probability).ln_1p() / b).exp_m1();
    }

    let log_beta = ln_beta_stable_parts(a, b);
    let flip = inverse_beta_reflect(a, b, probability, log_beta);
    let (a, b, target) = if flip {
        (b, a, 1.0 - probability)
    } else {
        (a, b, probability)
    };
    let ln_beta = log_beta.0 + log_beta.1;
    let (mut current, mut log_initial) = inverse_beta_initial(a, b, target, ln_beta);
    let smaller = a.min(b);
    let larger = a.max(b);
    if log_initial.is_finite()
        && larger >= STIRLING_MIN
        && (smaller < STIRLING_MIN || smaller <= 0.25 * larger)
    {
        let accurate_initial = lower_tail_initial_accurate(a, target, log_beta);
        if accurate_initial.0 > 0.0 && accurate_initial.0 < 1.0 {
            current = accurate_initial.0;
            log_initial = accurate_initial.1.0 + accurate_initial.1.1;
            let first_correction = ((b - 1.0).abs() / (a + 1.0)) * current;
            let remainder_ratio = (b - 2.0).abs() * current;
            if first_correction <= f64::EPSILON / 32.0 && remainder_ratio <= 0.5 {
                return if flip { 1.0 - current } else { current };
            }
        }
    }
    let min_subnormal = f64::from_bits(1);
    if current == 0.0 && log_initial < min_subnormal.ln() - core::f64::consts::LN_2 {
        return if flip { 1.0 } else { 0.0 };
    }
    let first_correction = ((b - 1.0).abs() / (a + 1.0)) * current;
    let remainder_ratio = (b - 2.0).abs() * current;
    if first_correction <= f64::EPSILON / 32.0 && remainder_ratio <= 0.5 {
        return if flip { 1.0 - current } else { current };
    }
    if current < f64::MIN_POSITIVE {
        let relative_correction = b * current / (a + 1.0);
        let relative_half_ulp = 0.5 * (min_subnormal / current);
        if relative_correction < 0.25 * relative_half_ulp {
            return if flip { 1.0 - current } else { current };
        }
    }
    let result = inverse_beta_log_tail(a, b, target, current, log_beta, ln_beta);
    if flip { 1.0 - result } else { result }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prec;
    use core::f64::consts as f64_consts;
    const MODULE_RELATIVE_ACC: f64 = 1e-14;

    fn beta_assert_relative_eq(a: f64, b: f64) {
        prec::assert_relative_eq!(
            a,
            b,
            epsilon = MODULE_EPS,
            max_relative = MODULE_RELATIVE_ACC
        );
    }

    fn beta_assert_abs_diff_eq(a: f64, b: f64) {
        prec::assert_abs_diff_eq!(a, b, epsilon = MODULE_EPS);
    }

    #[test]
    fn test_ln_beta() {
        beta_assert_relative_eq(ln_beta(0.5, 0.5), 1.144729885849400174144);
        beta_assert_relative_eq(ln_beta(1.0, 0.5), f64_consts::LN_2);
        beta_assert_relative_eq(ln_beta(2.5, 0.5), 0.163900632837673937284);
        beta_assert_relative_eq(ln_beta(0.5, 1.0), f64_consts::LN_2);
        beta_assert_relative_eq(ln_beta(1.0, 1.0), 0.0);
        beta_assert_relative_eq(ln_beta(2.5, 1.0), -0.9162907318741550651835);
        beta_assert_relative_eq(ln_beta(0.5, 2.5), 0.163900632837673937284);
        beta_assert_relative_eq(ln_beta(1.0, 2.5), -0.9162907318741550651835);
        beta_assert_relative_eq(ln_beta(2.5, 2.5), -2.608688089402107300388);
    }

    #[test]
    #[should_panic]
    fn test_ln_beta_a_lte_0() {
        ln_beta(0.0, 0.5);
    }

    #[test]
    #[should_panic]
    fn test_ln_beta_b_lte_0() {
        ln_beta(0.5, 0.0);
    }

    #[test]
    fn test_checked_ln_beta_a_lte_0() {
        assert!(checked_ln_beta(0.0, 0.5).is_err());
    }

    #[test]
    fn test_checked_ln_beta_b_lte_0() {
        assert!(checked_ln_beta(0.5, 0.0).is_err());
    }

    #[test]
    #[should_panic]
    fn test_beta_a_lte_0() {
        beta(0.0, 0.5);
    }

    #[test]
    #[should_panic]
    fn test_beta_b_lte_0() {
        beta(0.5, 0.0);
    }

    #[test]
    fn test_checked_beta_a_lte_0() {
        assert!(checked_beta(0.0, 0.5).is_err());
    }

    #[test]
    fn test_checked_beta_b_lte_0() {
        assert!(checked_beta(0.5, 0.0).is_err());
    }

    #[test]
    fn test_beta() {
        beta_assert_relative_eq(beta(0.5, 0.5), f64_consts::PI);
        beta_assert_relative_eq(beta(1.0, 0.5), 2.0);
        beta_assert_relative_eq(beta(2.5, 0.5), 1.17809724509617246442);
        beta_assert_relative_eq(beta(0.5, 1.0), 2.0);
        beta_assert_relative_eq(beta(1.0, 1.0), 1.0);
        beta_assert_relative_eq(beta(2.5, 1.0), 0.4);
        beta_assert_relative_eq(beta(0.5, 2.5), 1.17809724509617246442);
        beta_assert_relative_eq(beta(1.0, 2.5), 0.4);
        beta_assert_relative_eq(beta(2.5, 2.5), 0.073631077818510779026);
    }

    #[test]
    fn test_beta_inc() {
        beta_assert_relative_eq(beta_inc(0.5, 0.5, 0.5), f64_consts::FRAC_PI_2);
        beta_assert_relative_eq(beta_inc(0.5, 0.5, 1.0), f64_consts::PI);
        beta_assert_relative_eq(beta_inc(1.0, 0.5, 0.5), 0.5857864376269049511983);
        beta_assert_relative_eq(beta_inc(1.0, 0.5, 1.0), 2.0);
        beta_assert_relative_eq(beta_inc(2.5, 0.5, 0.5), 0.0890486225480862322117);
        beta_assert_relative_eq(beta_inc(2.5, 0.5, 1.0), 1.17809724509617246442);
        beta_assert_relative_eq(beta_inc(0.5, 1.0, 0.5), f64_consts::SQRT_2);
        beta_assert_relative_eq(beta_inc(0.5, 1.0, 1.0), 2.0);
        beta_assert_relative_eq(beta_inc(1.0, 1.0, 0.5), 0.5);
        beta_assert_relative_eq(beta_inc(1.0, 1.0, 1.0), 1.0);
        beta_assert_relative_eq(beta_inc(2.5, 1.0, 0.5), 0.0707106781186547524401);
        beta_assert_relative_eq(beta_inc(2.5, 1.0, 1.0), 0.4);
        beta_assert_relative_eq(beta_inc(0.5, 2.5, 0.5), 1.08904862254808623221);
        beta_assert_relative_eq(beta_inc(0.5, 2.5, 1.0), 1.17809724509617246442);
        beta_assert_relative_eq(beta_inc(1.0, 2.5, 0.5), 0.32928932188134524756);
        beta_assert_relative_eq(beta_inc(1.0, 2.5, 1.0), 0.4);
        beta_assert_relative_eq(beta_inc(2.5, 2.5, 0.5), 0.03681553890925538951323);
        beta_assert_relative_eq(beta_inc(2.5, 2.5, 1.0), 0.073631077818510779026);
    }

    #[test]
    #[should_panic]
    fn test_beta_inc_a_lte_0() {
        beta_inc(0.0, 1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_inc_b_lte_0() {
        beta_inc(1.0, 0.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_inc_x_lt_0() {
        beta_inc(1.0, 1.0, -1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_inc_x_gt_1() {
        beta_inc(1.0, 1.0, 2.0);
    }

    #[test]
    fn test_checked_beta_inc_a_lte_0() {
        assert!(checked_beta_inc(0.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_beta_inc_b_lte_0() {
        assert!(checked_beta_inc(1.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_beta_inc_x_lt_0() {
        assert!(checked_beta_inc(1.0, 1.0, -1.0).is_err());
    }

    #[test]
    fn test_checked_beta_inc_x_gt_1() {
        assert!(checked_beta_inc(1.0, 1.0, 2.0).is_err());
    }

    #[test]
    fn test_beta_reg() {
        beta_assert_abs_diff_eq(beta_reg(0.5, 0.5, 0.5), 0.5);
        assert_eq!(beta_reg(0.5, 0.5, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(1.0, 0.5, 0.5), 0.292893218813452475599);
        assert_eq!(beta_reg(1.0, 0.5, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(2.5, 0.5, 0.5), 0.07558681842161243795);
        assert_eq!(beta_reg(2.5, 0.5, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(0.5, 1.0, 0.5), f64_consts::FRAC_1_SQRT_2);
        assert_eq!(beta_reg(0.5, 1.0, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(1.0, 1.0, 0.5), 0.5);
        assert_eq!(beta_reg(1.0, 1.0, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(2.5, 1.0, 0.5), 0.1767766952966368811);
        assert_eq!(beta_reg(2.5, 1.0, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(0.5, 2.5, 0.5), 0.92441318157838756205);
        assert_eq!(beta_reg(0.5, 2.5, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(1.0, 2.5, 0.5), 0.8232233047033631189);
        assert_eq!(beta_reg(1.0, 2.5, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(2.5, 2.5, 0.5), 0.5);
        assert_eq!(beta_reg(2.5, 2.5, 1.0), 1.0);
    }

    #[test]
    fn test_beta_reg_large_parameters_against_reference() {
        let cases = [
            (1e6, 2e6, 0.333, 0.11032283951664962),
            (1e6, 2e6, 1.0 / 3.0, 0.5000542891707268),
            (1e6, 2e6, 0.334, 0.9928335645421132),
            (1e8, 2e8, 0.3333, 0.11033439854811466),
            (1e8, 2e8, 1.0 / 3.0, 0.5000054289165304),
            (1e8, 2e8, 0.3334, 0.992845709515461),
            (1e5, 1e5, 0.49, 1.8571347290404196e-19),
            (1e5, 1e5, 0.499, 0.18554674455755675),
            (1e5, 1e5, 0.501, 0.8144532554424433),
            (40.0, 32.0, 1e-8, 1.2676414050441584e-300),
            (32.0, 40.0, 1e-8, 1.5845516362868252e-236),
            (0.1, 1e8, 1e-8, 0.9758726562930068),
            (0.1, 1e8, 1e-9, 0.8275517592836537),
            (2.0, 1e8, 1e-8, 0.2642411213359098),
            (10.0, 1e8, 1e-7, 0.5420704043826821),
            (1e13, 9.9e14, 0.01, 0.5000000414451727),
            (
                1.098252731340299,
                1.780042655540735e17,
                5.235783704840033e-17,
                0.999881646675342,
            ),
            (
                7_627_209.761,
                11.3319,
                0.9999105965110135,
                1.6790000011611638e-274,
            ),
            (99_999.0, 11.3319, 0.9998, 0.013667998876668642),
            (100_001.0, 11.3319, 0.9998, 0.013665136770782414),
            (100_000.0, 10.0, 0.992653308338289, 1.0000000000029653e-300),
        ];

        for (a, b, x, expected) in cases {
            let actual = beta_reg(a, b, x);
            let error = (actual - expected).abs();
            let tolerance = 5e-12 * expected.max(1e-300);
            assert!(
                error <= tolerance,
                "beta_reg({a}, {b}, {x}) = {actual}, expected {expected}, error {error}"
            );
        }
    }

    #[test]
    fn test_beta_reg_extreme_ratio_central_value_against_reference() {
        let cases: [(f64, f64, f64, f64); 2] = [
            (
                1.2e7,
                1.2000000000000001e307,
                9.999999999999999e-301,
                0.50003838823874907,
            ),
            (1.2e7, 1e308, 1.2e-301, 0.50003838823881181),
        ];
        for (a, b, x, expected) in cases {
            let actual = beta_reg(a, b, x);
            assert!(
                actual.to_bits().abs_diff(expected.to_bits()) <= 1024,
                "beta_reg({a}, {b}, {x}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_beta_reg_overflowing_shape_sum() {
        let lower = f64::from_bits(0.5_f64.to_bits() - 1);
        let upper = f64::from_bits(0.5_f64.to_bits() + 1);
        assert_eq!(beta_reg(1e308, 1e308, lower), 0.0);
        assert_eq!(beta_reg(1e308, 1e308, 0.5), 0.5);
        assert_eq!(beta_reg(1e308, 1e308, upper), 1.0);
        let actual = checked_ln_beta(1e308, 1e308).unwrap();
        assert!(actual.is_finite());
        assert!((actual / 1e308 + 2.0 * core::f64::consts::LN_2).abs() <= 2e-15);
        let expected = -2.0007184997951635e301;
        let actual = checked_ln_beta(f64::MAX, 1e300).unwrap();
        assert!(((actual - expected) / expected).abs() <= 3e-10);

        let mean = f64::from_bits(0x3fe5555555555555);
        assert_eq!(beta_reg(1e308, 5e307, mean), 0.0);
        assert_eq!(
            beta_reg(1e308, 5e307, f64::from_bits(mean.to_bits() + 1)),
            1.0
        );
    }

    #[test]
    fn test_beta_reg_algorithm_boundaries_against_reference() {
        let cases = [
            (39_999_999.0, 79_999_999.0, 0.33335, 0.6507629787874431),
            (40_000_001.0, 80_000_001.0, 0.33335, 0.6507151999304125),
            (29_999_999.0, 270_000_001.0, 0.10001, 0.7182251069092127),
            (30_000_001.0, 269_999_999.0, 0.10001, 0.7180951316317142),
            (1e8, 2e8, 0.33328635138267637, 0.042150859881784875),
            (1e8, 2e8, 0.33328603712606697, 0.04112293252416181),
        ];

        for (a, b, x, expected) in cases {
            let actual = beta_reg(a, b, x);
            let relative_error = ((actual - expected) / expected).abs();
            assert!(
                relative_error <= 2e-12,
                "beta_reg({a}, {b}, {x}) = {actual}, expected {expected}, relative error {relative_error}"
            );
        }
    }

    #[test]
    fn test_beta_reg_large_a_small_b_subnormal() {
        let cases = [
            (
                1e18,
                39.9,
                f64::from_bits(0x3feffffffffffff8),
                f64::from_bits(0x1520b9),
            ),
            (1e8, 0.9, 0.99999284, f64::from_bits(0xfce148c723)),
        ];
        for (a, b, x, expected) in cases {
            let actual = beta_reg(a, b, x);
            assert!(
                actual.to_bits().abs_diff(expected.to_bits()) <= 4,
                "beta_reg({a}, {b}, {x}) = {actual:e} ({:#x}), expected {expected:e} ({:#x})",
                actual.to_bits(),
                expected.to_bits()
            );
        }
    }

    #[test]
    fn test_beta_reg_large_a_tiny_b_rounded_complement() {
        let x = f64::from_bits(1.0_f64.to_bits() - 1);
        let cases = [
            (
                1.7492718718060828e16,
                1.7529350052864036e-11,
                f64::from_bits(0x3d7057be8b9ff83b),
            ),
            (
                2.6496319847741348e16,
                3.8997923472821135e-12,
                f64::from_bits(0x3d2edb1e5cecbc3f),
            ),
            (
                1.3443603650606364e16,
                3.8682302848162155e-10,
                f64::from_bits(0x3dc581e85bf535df),
            ),
            (
                1.4398454548018444e16,
                1.1381500822684144e-10,
                f64::from_bits(0x3da5a5b386b28adf),
            ),
            (
                9_288_475_808_954_264.0,
                5.299156768316511e-9,
                f64::from_bits(0x3e12f55b03b79471),
            ),
            (
                1.6977806187270128e16,
                5.491909396055591e-12,
                f64::from_bits(0x3d562f56c473937b),
            ),
        ];
        for (a, b, expected) in cases {
            let actual = beta_reg(a, b, x);
            assert!(
                actual.to_bits().abs_diff(expected.to_bits()) <= 64,
                "beta_reg({a}, {b}, {x}) = {actual:e}, expected {expected:e}"
            );
        }
    }

    #[test]
    fn test_beta_reg_small_shape_upper_gamma_against_reference() {
        let cases = [
            (
                112_176_097_488.593_9,
                1.3959752253898728e-12,
                f64::from_bits(0x3fefffffffff851d),
                [
                    9.999569151432288e-13,
                    9.999869047052781e-13,
                    1.000016895594139e-12,
                ],
            ),
            (
                238_641_107_383.443_27,
                1.799146819367202e-12,
                f64::from_bits(0x3fefffffffffb5cc),
                [
                    9.999141915967447e-13,
                    9.999714463464230e-13,
                    1.000028705627258e-12,
                ],
            ),
            (
                246.932962952654,
                1.1953991131275682e-12,
                f64::from_bits(0x3feffffee33a9e66),
                [
                    9.999999999706979e-12,
                    9.999999999957152e-12,
                    1.000000000020733e-11,
                ],
            ),
        ];
        for (a, b, x, expected) in cases {
            for (offset, expected) in [-1_i64, 0, 1].into_iter().zip(expected) {
                let x = f64::from_bits(x.to_bits().wrapping_add_signed(offset));
                let actual = beta_reg(a, b, x);
                let relative_error = ((actual - expected) / expected).abs();
                assert!(
                    relative_error <= 5e-13,
                    "beta_reg({a}, {b}, {x}) = {actual:e}, expected {expected:e}, relative error {relative_error}"
                );
            }
        }
    }

    #[test]
    fn test_ln_beta_reg_tiny_shape_scaled_gamma_against_reference() {
        let cases = [
            (
                f64::from_bits(0x3feffffbce423b02),
                [
                    f64::from_bits(0xc085ae5914154dec),
                    f64::from_bits(0xc085ae59141548a5),
                    f64::from_bits(0xc085ae591415435d),
                ],
            ),
            (
                f64::from_bits(0x3fefffeb0750a667),
                [
                    f64::from_bits(0xc085f9547c11ffd4),
                    f64::from_bits(0xc085f9547c11fbaa),
                    f64::from_bits(0xc085f9547c11f77f),
                ],
            ),
            (
                f64::from_bits(0x3fefff7be22e5816),
                [
                    f64::from_bits(0xc087af793037cd6d),
                    f64::from_bits(0xc087af793037c98d),
                    f64::from_bits(0xc087af793037c5ad),
                ],
            ),
        ];
        for (x, expected) in cases {
            for (offset, expected) in [-1_i64, 0, 1].into_iter().zip(expected) {
                let x = f64::from_bits(x.to_bits().wrapping_add_signed(offset));
                let actual = checked_ln_beta_reg(1e6, 1e-300, x).unwrap();
                assert!((actual - expected).abs() <= 2e-13);
            }
        }
    }

    #[test]
    fn test_ln_beta_reg_power_series_is_locally_monotone() {
        let cases = [
            (
                9.11327743985456,
                133_525_174_076_797.34,
                f64::from_bits(0x3cf3d7e149ac36dd),
            ),
            (
                6.078046923216118,
                31_131_628_187_944.344,
                f64::from_bits(0x3cf592225b607c93),
            ),
        ];
        for (a, b, root) in cases {
            let mut previous = f64::NEG_INFINITY;
            for offset in -100_i64..=100 {
                let x = f64::from_bits(root.to_bits().wrapping_add_signed(offset));
                let value = checked_ln_beta_reg(a, b, x).unwrap();
                assert!(value >= previous, "a={a}, b={b}, x={x}");
                previous = value;
            }
        }
    }

    #[test]
    fn test_beta_reg_power_series_is_locally_monotone() {
        let cases: [(f64, f64, f64); 7] = [
            (
                0.47937889777569664,
                390_713_368_494_940.25,
                5.842150555453333e-16,
            ),
            (
                0.20713927131052443,
                1_264_447_072_006_281.8,
                7.355559632987759e-17,
            ),
            (
                0.5883286844875396,
                53_930_034_336_347.77,
                2.7619798816617607e-15,
            ),
            (
                0.3047929367901273,
                258_195_370_359_324.8,
                1.5384576649827977e-15,
            ),
            (
                0.21280081734067854,
                54_626_561.16286868,
                4.363878090733803e-9,
            ),
            (
                42.51394493556042,
                2_256_890_178_438.929,
                1.0526514336858459e-13,
            ),
            (
                77.54913939933753,
                14_481_621_713.827797,
                2.8605493321691776e-11,
            ),
        ];
        for (a, b, center) in cases {
            let mut previous = 0.0;
            for offset in -64_i64..=64 {
                let x = f64::from_bits(center.to_bits().wrapping_add_signed(offset));
                let value = beta_reg(a, b, x);
                assert!(value >= previous, "a={a}, b={b}, x={x}");
                previous = value;
            }
        }
    }

    #[test]
    fn test_beta_reg_power_series_subnormal_result_against_reference() {
        let actual = beta_reg(
            147.13149557601173,
            1.6465152935404156e16,
            f64::from_bits(0x3c78ef1d912aaa46),
        );
        assert_eq!(actual.to_bits(), 4);
    }

    #[test]
    fn test_beta_reg_power_series_boundary_against_reference() {
        let (log_beta, log_beta_error) = ln_beta_accurate_parts(10.0, 32.0);
        assert_eq!(log_beta.to_bits(), 0xc03723e193251f2a);
        assert!((log_beta_error - f64::from_bits(0xbcd496eeab49e82c)).abs() <= 2e-19);
        let cases = [
            (0x3fcfffffffffff7f, 0x3fe30d694d7fb0f1),
            (0x3fcfffffffffff80, 0x3fe30d694d7fb0f2),
            (0x3fcfffffffffff81, 0x3fe30d694d7fb0f4),
            (0x3fcfffffffffff82, 0x3fe30d694d7fb0f5),
        ];
        let mut previous = 0;
        for (x, expected) in cases {
            let actual = beta_reg(10.0, 32.0, f64::from_bits(x)).to_bits();
            assert!(
                actual.abs_diff(expected) <= 2,
                "x={x:#018x}, actual={actual:#018x}, expected={expected:#018x}"
            );
            assert!(
                actual > previous,
                "x={x:#018x}, actual={actual:#018x}, previous={previous:#018x}"
            );
            previous = actual;
        }
    }

    #[test]
    fn test_beta_reg_near_one_moderate_shapes_converges() {
        let x = f64::from_bits(1.0_f64.to_bits() - 1);
        for (a, b) in [(39.9, 40.0), (40.0, 40.0), (40.0, 41.0)] {
            let actual = checked_beta_reg(a, b, x).unwrap();
            assert!(
                (0.0..=1.0).contains(&actual),
                "a={a}, b={b}, actual={actual:?}"
            );
        }
    }

    #[test]
    fn test_beta_reg_near_one_uses_convergent_power_series() {
        let x = f64::from_bits(1.0_f64.to_bits() - 1);
        let cases = [
            (217348.9453342118, 7.083729216298346e17),
            (74.50754210941346, 4.6813710928374765e17),
            (13.940004463756644, 5.294575065065153e17),
        ];
        for (a, b) in cases {
            let actual = checked_beta_reg(a, b, x).unwrap();
            assert!(
                (0.0..=1.0).contains(&actual),
                "a={a}, b={b}, actual={actual:?}"
            );
        }
    }

    #[test]
    fn test_beta_reg_tiny_first_shape_remains_monotone_below_split() {
        let a = 2.1856409177373306e-11;
        let b = 18.619031676940928;
        let references = [
            (0x3ea669742f6d91e9_u64, 0x3fefffffffdfb936_u64),
            (0x3fa7d0724ba189c0_u64, 0x3fefffffffff2a9e_u64),
        ];
        let mut previous = 0_u64;
        for (x, expected) in references {
            let actual = checked_beta_reg(a, b, f64::from_bits(x)).unwrap().to_bits();
            assert!(
                actual.abs_diff(expected) <= 4,
                "x={x:#018x}, actual={actual:#018x}, expected={expected:#018x}"
            );
            assert!(actual >= previous);
            previous = actual;
        }
    }

    #[test]
    fn test_beta_reg_exact_complement_fraction_against_reference() {
        let center = 0x3ee7118258b21dd3_u64;
        let references = [
            (-128_i64, 0x3fe51a846b074d53_u64),
            (-64, 0x3fe51a846b074dbd),
            (-1, 0x3fe51a846b074e25),
            (0, 0x3fe51a846b074e27),
            (1, 0x3fe51a846b074e29),
            (64, 0x3fe51a846b074e91),
            (128, 0x3fe51a846b074efb),
        ];
        for (offset, expected) in references {
            let x = f64::from_bits(center.wrapping_add_signed(offset));
            let actual = checked_beta_reg(10.0, 1e6, x).unwrap().to_bits();
            assert!(
                actual.abs_diff(expected) <= 3,
                "offset={offset}, actual={actual:#018x}, expected={expected:#018x}"
            );
        }
        let mut previous = 0.0;
        for bits in center - 128..=center + 128 {
            let actual = checked_beta_reg(10.0, 1e6, f64::from_bits(bits)).unwrap();
            assert!(
                actual >= previous,
                "bits={bits:#018x}, previous={previous:?}, actual={actual:?}"
            );
            previous = actual;
        }
    }

    #[test]
    fn test_beta_reg_continued_fraction_adjacent_reference() {
        let a = 1833.469197457969;
        let b = 648975.2550258434;
        let cases = [
            (0x3f63feb8f2cd8c97, 0x3e112e0be826bc4b, 0xc034b927f32c0140),
            (0x3f63feb8f2cd8c98, 0x3e112e0be826bd23, 0xc034b927f32c0133),
        ];
        let mut previous = 0;
        for (x, expected, expected_log) in cases {
            let x = f64::from_bits(x);
            assert_eq!(
                checked_ln_beta_reg(a, b, x).unwrap().to_bits(),
                expected_log
            );
            let actual = beta_reg(a, b, x).to_bits();
            let log_power = beta_reg_log_power_parts(a, b, x);
            let fraction = beta_continued_fraction(a, b, x).unwrap();
            let direct = ((log_power.0 + log_power.1).exp() / fraction).to_bits();
            assert!(
                actual.abs_diff(expected) <= 2,
                "actual={actual:#018x}, direct={direct:#018x}, expected={expected:#018x}"
            );
            assert!(actual > previous);
            previous = actual;
        }
    }

    #[test]
    fn test_beta_reg_accuracy_gaps_against_500_digit_references() {
        let cases = [
            (
                0.8144818117006096,
                1.250857626649459e-12,
                0.9669920517519052,
                0x3d94af09e6a6b751_u64,
            ),
            (
                0.2623971057030866,
                5.23256841817563e-12,
                0.9924817752047999,
                0x3dc7f760fcea90cd,
            ),
            (
                25.32628846940565,
                3.1028101710805442,
                0.9276950604606229,
                0x3fe69562e02877e6,
            ),
        ];
        for (a, b, x, expected) in cases {
            let actual = beta_reg(a, b, x).to_bits();
            assert!(
                actual.abs_diff(expected) <= 4,
                "a={a:?}, b={b:?}, x={x:?}, actual={actual:#018x}, expected={expected:#018x}"
            );
        }
    }

    #[test]
    fn test_inv_beta_reg_typical_against_500_digit_reference() {
        let actual = inv_beta_reg(2.0, 5.0, 0.3).to_bits();
        let expected = 0x3fc745560dce9cd1_u64;
        assert!(
            actual.abs_diff(expected) <= 2,
            "actual={actual:#018x}, expected={expected:#018x}"
        );
    }

    #[test]
    fn test_beta_reg_tiny_x_large_b_against_reference() {
        let cases: [(f64, f64, f64, u64); 2] = [
            (100.0, 1e308, 1.01e-306, 0x3fe1b153914c2fe1_u64),
            (1e6, 1e308, 1.000001e-302, 0x3fe0045b85d90000_u64),
        ];
        for (a, b, center, expected) in cases {
            let center_bits = center.to_bits();
            let mut previous = 0.0;
            for bits in center_bits - 128..=center_bits + 128 {
                let actual = checked_beta_reg(a, b, f64::from_bits(bits)).unwrap();
                assert!(
                    actual >= previous,
                    "a={a}, b={b}, bits={bits:#018x}, previous={previous:?}, actual={actual:?}"
                );
                previous = actual;
            }
            let actual = checked_beta_reg(a, b, center).unwrap().to_bits();
            assert!(
                actual.abs_diff(expected) <= 4,
                "a={a}, b={b}, actual={actual:#018x}, expected={expected:#018x}"
            );
        }
    }

    #[test]
    fn test_beta_reg_tiny_x_continued_fraction_singularity() {
        let references = [
            (0x3c9d1c7c0f1fd2c9_u64, 0x3fe1b153914c2fde_u64),
            (0x3c9d1c7c0f1fd2ca_u64, 0x3fe1b153914c2fe2_u64),
            (0x3c9d1c7c0f1fd2cb_u64, 0x3fe1b153914c2fe7_u64),
        ];
        let mut previous = 0_u64;
        for (x, expected) in references {
            let actual = checked_beta_reg(100.0, 1e18, f64::from_bits(x))
                .unwrap()
                .to_bits();
            assert!(
                actual.abs_diff(expected) <= 8,
                "x={x:#018x}, actual={actual:#018x}, expected={expected:#018x}"
            );
            assert!(actual >= previous);
            previous = actual;
        }
    }

    #[test]
    fn test_beta_reg_tiny_x_does_not_lose_complement() {
        let a = 40.0;
        let b = 1e18;
        let center = 0x3c87a28834d566b4_u64;
        let mut previous = 0.0;
        for bits in center - 128..=center + 128 {
            let actual = checked_beta_reg(a, b, f64::from_bits(bits)).unwrap();
            assert!(
                actual >= previous,
                "bits={bits:#018x}, previous={previous:?}, actual={actual:?}"
            );
            previous = actual;
        }
        let actual = checked_beta_reg(a, b, f64::from_bits(center)).unwrap();
        assert_eq!(actual.to_bits(), 0x3fe2a783c7380c04);
    }

    #[test]
    fn test_beta_reg_power_series_tiny_shape_boundary() {
        let a = f64::from_bits(0x00000000000007e8);
        let b = f64::from_bits(0x4040000000000000);
        let x = f64::from_bits(0x01556e1fc2f8f359);
        assert!(beta_power_series_log_parts(a, b, x).is_ok());
        for offset in -3_i64..=3 {
            let x = f64::from_bits(x.to_bits().wrapping_add_signed(offset));
            assert_eq!(checked_beta_reg(a, b, x).unwrap(), 1.0);
            assert_eq!(
                checked_ln_beta_reg(a, b, x).unwrap().to_bits(),
                0x8000000000155101
            );
        }
    }

    #[test]
    fn test_beta_reg_power_series_tiny_shape_is_locally_monotone() {
        let a = f64::from_bits(0x3d719799812dea11);
        let b = f64::from_bits(0x43abc16d674ec800);
        let x = f64::from_bits(0x3c32725dd1d243ac);
        for offset in -2_i64..=3 {
            let x = f64::from_bits(x.to_bits().wrapping_add_signed(offset));
            assert_eq!(beta_reg(a, b, x).to_bits(), 0x3feffffffffff848);
        }
    }

    #[test]
    fn test_accurate_ln_against_multiprecision_reference() {
        let cases = [
            (0x0000000000000001, 0xc0874385446d71c3, 0xbd28e569fa8ee781),
            (0x0010000000000000, 0xc086232bdd7abcd2, 0xbd1eef3fec1be37f),
            (0x39b0000000000000, 0xc051542457337d43, 0x3cde3948c376279d),
            (0x3fe8000000000000, 0xbfd269621134db92, 0xbc7e0efadd9db02b),
            (0x3ff6a09e667f3bcc, 0x3fd62e42fefa39ee, 0xbc78d6e518e495a3),
            (0x3ff6a09e667f3bcd, 0x3fd62e42fefa39f0, 0x3c7c2e0e1b1548c2),
            (0x3ff6a09e667f3bce, 0x3fd62e42fefa39f3, 0x3c7133014f0f271f),
            (0x3ff8000000000000, 0x3fd9f323ecbf984c, 0xbc4a92e513217f5c),
            (0x4000000000000000, 0x3fe62e42fefa39ef, 0x3c7abc9e3b39803f),
            (0x4630000000000000, 0x4051542457337d43, 0xbcde3948c376279d),
            (0x7fefffffffffffff, 0x40862e42fefa39ef, 0x3d1a9c9e3b39803f),
        ];
        for (input, expected_high, expected_low) in cases {
            let (high, low) = accurate_ln(f64::from_bits(input));
            let expected_low = f64::from_bits(expected_low);
            let magnitude = expected_low.abs();
            let spacing = f64::from_bits(magnitude.to_bits() + 1) - magnitude;
            assert_eq!(high.to_bits(), expected_high);
            assert!(
                (low - expected_low).abs() <= 8.0 * spacing,
                "input={input:#018x}, low={low:?}, expected={expected_low:?}"
            );
        }
    }

    #[test]
    fn test_beta_reg_bgrat_lower_shape_boundary() {
        let cases = [
            (31.999, 0.5, 0.9, f64::from_bits(0x3f83d8d11db5fecb)),
            (32.001, 0.5, 0.9, f64::from_bits(0x3f83d79daec1916d)),
        ];
        for (a, b, x, expected) in cases {
            let actual = beta_reg(a, b, x);
            let relative_error = ((actual - expected) / expected).abs();
            assert!(
                relative_error <= 1e-12,
                "beta_reg({a}, {b}, {x}) = {actual}, expected {expected}, relative error {relative_error}"
            );
        }
    }

    #[test]
    fn test_beta_reg_scaled_gamma_boundary_against_reference() {
        let cases = [
            (100_000.0, 10.1, 0.9996800497549934, 1.904358612390508e-6),
            (100_000.0, 10.1, 0.9996200704814975, 2.132915725768903e-8),
            (100_000.0, 10.1, 0.9996000781900461, 4.537230484132134e-9),
            (1e8, 0.1, 0.9999993610002013, 4.358202373741317e-31),
            (1e8, 0.1, 0.999999360000202, 3.9380016482795125e-31),
            (1e8, 0.9, 0.9999928600254862, 3.9763309919351194e-311),
            (1e8, 0.9, 0.9999928400256292, 5.37987584721e-312),
        ];
        for (a, b, x, expected) in cases {
            let actual = beta_reg(a, b, x);
            let relative_error = ((actual - expected) / expected).abs();
            assert!(
                relative_error <= 5e-12,
                "beta_reg({a}, {b}, {x}) = {actual:e}, expected {expected:e}, relative error {relative_error}"
            );
        }
    }

    #[test]
    fn test_beta_reg_small_shapes_stays_in_range() {
        let cases = [
            (
                0.1350095402068847,
                2.522023373459552e-11,
                0.858047569045879,
                2.2760966295231215e-10,
            ),
            (
                1.6182184909371272e-12,
                0.8611154417262772,
                0.2090095742796264,
                0.9999999999971043,
            ),
        ];
        for (a, b, x, expected) in cases {
            let actual = beta_reg(a, b, x);
            assert!((0.0..=1.0).contains(&actual));
            assert!(
                (actual - expected).abs() <= 5e-15 * expected.max(1e-10),
                "beta_reg({a}, {b}, {x}) = {actual}, expected {expected}"
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_beta_reg_a_lte_0() {
        beta_reg(0.0, 1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_reg_b_lte_0() {
        beta_reg(1.0, 0.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_reg_x_lt_0() {
        beta_reg(1.0, 1.0, -1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_reg_x_gt_1() {
        beta_reg(1.0, 1.0, 2.0);
    }

    #[test]
    fn test_checked_beta_reg_a_lte_0() {
        assert!(checked_beta_reg(0.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_beta_reg_b_lte_0() {
        assert!(checked_beta_reg(1.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_beta_reg_x_lt_0() {
        assert!(checked_beta_reg(1.0, 1.0, -1.0).is_err());
    }

    #[test]
    fn test_checked_beta_reg_x_gt_1() {
        assert!(checked_beta_reg(1.0, 1.0, 2.0).is_err());
    }

    #[test]
    fn test_inv_beta_reg_extreme_probability_does_not_panic() {
        let actual = inv_beta_reg(200.0, 2.0, 1e-165);
        let expected = 0.14582246504394993;
        let relative_error = ((actual - expected) / expected).abs();
        assert!(
            relative_error <= 5e-13,
            "actual {actual}, expected {expected}"
        );
    }

    #[test]
    fn test_inv_beta_reg_extreme_probability_terminates() {
        let actual = inv_beta_reg(200.0, 2.0, 1e-60);
        let expected = 0.4897050363600545;
        let relative_error = ((actual - expected) / expected).abs();
        assert!(
            relative_error <= 5e-13,
            "actual {actual}, expected {expected}"
        );
    }

    #[test]
    fn test_inv_beta_reg_small_shape_lower_tail() {
        let cases = [
            (1e-33, 0.0),
            (1e-32, f64::from_bits(2)),
            (1e-31, 1.215703604971242e-313),
            (1e-30, 1.2157036049544172e-303),
            (1e-20, 1.2157036049544e-203),
            (1e-10, 1.2157036049543856e-103),
            (1e-4, 1.2157036049543764e-43),
            (1e-2, 1.215703604954373e-23),
        ];
        let mut previous = 0.0;

        for (probability, expected) in cases {
            let actual = inv_beta_reg(0.1, 500.0, probability);
            if expected == 0.0 {
                assert_eq!(actual, expected);
                continue;
            }
            let relative_error = ((actual - expected) / expected).abs();
            assert!(
                relative_error <= 5e-14,
                "inv_beta_reg(0.1, 500, {probability}) = {actual}, expected {expected}, relative error {relative_error}"
            );
            assert!(actual >= previous);
            previous = actual;
        }
    }

    #[test]
    fn test_inv_beta_reg_small_shape_rounds_extreme_tail() {
        let cases = [
            (1e-30, 0x0010aad919ea62cfa),
            (1e-31, 0x00000005baa38454),
            (1e-32, 0x0000000000000002),
        ];
        for (probability, expected) in cases {
            assert_eq!(inv_beta_reg(0.1, 500.0, probability).to_bits(), expected);
        }
    }

    #[test]
    fn test_inv_beta_reg_early_tail_correction_against_reference() {
        assert_eq!(
            inv_beta_reg(10.0, 1e18, f64::from_bits(0x206b45a31ae6c90e),).to_bits(),
            0x392f275e33972f0c
        );
    }

    #[test]
    fn test_inv_beta_reg_large_a_tiny_b_lower_tail() {
        let cases = [
            (
                27.229198855436444,
                3.192251825919222e-12,
                1e-12,
                0x3fef0fdff94fb881,
            ),
            (
                10.741694769633645,
                2.057645959850482e-10,
                5e-9,
                0x3fefffffffffca0f,
            ),
            (
                3.791228906881053,
                3.2160853621997853e-9,
                5e-9,
                0x3feeb7bc46a5108f,
            ),
            (
                0.07111267420172858,
                2.459402818189203e-11,
                1e-9,
                0x3fefffffffffa790,
            ),
            (
                0.0715388852036888,
                3.187243980970482e-9,
                1e-7,
                0x3feffffff2a24e82,
            ),
        ];
        for (a, b, probability, expected) in cases {
            let actual = inv_beta_reg(a, b, probability).to_bits();
            assert!(
                actual.abs_diff(expected) <= 2,
                "a={a}, b={b}, actual={actual:#x}, expected={expected:#x}"
            );
        }
    }

    #[test]
    fn test_beta_reg_moderate_a_tiny_b_against_reference() {
        let actual = beta_reg(
            6.333131463399467,
            1.3323977213610329e-11,
            0.9137396220685055,
        )
        .to_bits();
        let expected = 0x3d9ef22640629504_u64;
        assert!(
            actual.abs_diff(expected) <= 4,
            "actual={actual:#018x}, expected={expected:#018x}"
        );
    }

    #[test]
    fn test_beta_reg_small_shapes_near_one_against_reference() {
        let actual = checked_beta_reg(0.8593272045160161, 0.9835139781033098, 0.9999999999999999)
            .unwrap()
            .to_bits();
        assert!(actual.abs_diff(0x3feffffffffffffe) <= 1);
    }

    #[test]
    fn test_ln_beta_accurate_parts_reference() {
        let cases = [
            (0.1, 32.0, 0x3ffe85545aa95cd9, 0xbc8fef9442e0fba4),
            (0.3, 1000.0, 0xbfef3edcaae7008a, 0xbc8237c135557682),
            (10.0, 32.0, 0xc03723e193251f2a, 0xbcd496eeab49e82c),
        ];
        for (a, b, high, low) in cases {
            let actual = ln_beta_accurate_parts(a, b);
            assert_eq!(actual.0.to_bits(), high);
            let expected = f64::from_bits(low);
            let high_value = f64::from_bits(high).abs();
            let spacing = f64::from_bits(high_value.to_bits() + 1) - high_value;
            assert!(
                (actual.1 - expected).abs() <= 0.01 * spacing,
                "a={a}, b={b}, actual={:?}, expected={expected:?}",
                actual.1
            );
        }
        let gamma = ln_gamma_accurate_parts(0.1);
        assert_eq!(gamma.0.to_bits(), 0x4002058e35f3deee);
        assert!((gamma.1 - f64::from_bits(0xbc97ad885b23066b)).abs() <= 5e-19);
        let delta = ln_gamma_delta_parts(32.0, 0.1);
        assert_eq!(delta.0.to_bits(), 0x3fd6172044f9840c);
        assert!((delta.1 - f64::from_bits(0xbc7ed6f8e6ca2265)).abs() <= 5e-19);
    }

    #[test]
    fn test_inv_beta_reg_regular_shape_lower_tail() {
        let cases = [
            (1e-300, 7.053456158585983e-153),
            (1e-100, 7.053456158585983e-53),
            (1e-40, 7.053456158585983e-23),
            (1e-30, 7.053456158585999e-18),
            (1e-20, 7.053456158916007e-13),
        ];
        let mut previous = 0.0;

        for (probability, expected) in cases {
            let actual = inv_beta_reg(2.0, 200.0, probability);
            let relative_error = ((actual - expected) / expected).abs();
            assert!(
                relative_error <= 5e-12,
                "inv_beta_reg(2, 200, {probability}) = {actual}, expected {expected}, relative error {relative_error}"
            );
            assert!(actual > previous);
            previous = actual;
        }
    }

    #[test]
    fn test_inv_beta_reg_large_parameters() {
        let cases = [(0.1, 0.3332984541555588), (0.9, 0.3333682129869408)];

        for (probability, expected) in cases {
            let actual = inv_beta_reg(1e8, 2e8, probability);
            let relative_error = ((actual - expected) / expected).abs();
            assert!(
                relative_error <= 5e-12,
                "inv_beta_reg(1e8, 2e8, {probability}) = {actual}, expected {expected}, relative error {relative_error}"
            );
        }
    }

    #[test]
    fn test_inv_beta_reg_overflowing_shape_sum() {
        for shape in [1e307, 1e308] {
            assert_eq!(inv_beta_reg(shape, shape, 0.1), 0.5);
            assert_eq!(inv_beta_reg(shape, shape, 0.9), 0.5);
        }
        let expected = f64::from_bits(0x3fe5555555555555);
        for probability in [0.1, 0.5, 0.9] {
            assert_eq!(inv_beta_reg(1e308, 5e307, probability), expected);
        }
    }

    #[test]
    fn test_inv_beta_reg_min_subnormal_large_a_tiny_b() {
        let cases = [
            (
                1.418970410722184e16,
                0.0001029663852090984,
                f64::from_bits(0x3feffffffffffe31),
            ),
            (
                4.674866848491979e16,
                1.8053488701439817e-11,
                f64::from_bits(0x3fefffffffffff77),
            ),
            (
                3.2111418342313892e16,
                0.004499324538510611,
                f64::from_bits(0x3fefffffffffff33),
            ),
            (
                3.117388966777583e17,
                0.00105319319351692,
                f64::from_bits(0x3fefffffffffffeb),
            ),
            (
                9.629243664883278e17,
                3.208469262232818e-5,
                f64::from_bits(0x3feffffffffffff9),
            ),
            (
                7.351984375091425e17,
                2.6812348495943197e-11,
                f64::from_bits(0x3feffffffffffff7),
            ),
            (
                1.7012222411445178e17,
                7.129120396546662e-6,
                f64::from_bits(0x3fefffffffffffda),
            ),
            (
                1.9543788953358486e17,
                1.1304448170316649e-12,
                f64::from_bits(0x3fefffffffffffdf),
            ),
            (
                9.996829742803416e17,
                1.410942501012109e-8,
                f64::from_bits(0x3feffffffffffffa),
            ),
        ];
        for (a, b, expected) in cases {
            let actual = inv_beta_reg(a, b, f64::from_bits(1));
            assert_eq!(actual, expected, "a={a}, b={b}");
        }
    }

    #[test]
    fn test_inv_beta_reg_small_shape_upper_gamma() {
        let cases = [
            (
                112_176_097_488.593_9,
                1.3959752253898728e-12,
                1e-12,
                f64::from_bits(0x3fefffffffff851d),
            ),
            (
                238_641_107_383.443_27,
                1.799146819367202e-12,
                1e-12,
                f64::from_bits(0x3fefffffffffb5cc),
            ),
            (
                246.932962952654,
                1.1953991131275682e-12,
                1e-11,
                f64::from_bits(0x3feffffee33a9e66),
            ),
        ];
        for (a, b, probability, expected) in cases {
            assert_eq!(inv_beta_reg(a, b, probability), expected);
        }
    }

    #[test]
    fn test_inv_beta_reg_large_a_tiny_b_is_monotone() {
        let cases = [
            (5.034263241208714e17, 1.8917307295846354e-5),
            (7.663354755004902e17, 0.06629881964843289),
            (9.703110430017175e17, 1.3592520602121614e-6),
            (7.633216846220836e17, 0.04203941489807821),
            (9.846275348488209e17, 7.919461066109182e-7),
            (8.324653375999025e17, 5.050727538603147e-11),
            (6.519274800253329e17, 1.3080952792915084e-9),
            (9.600975622510844e17, 3.1549066745793863e-7),
            (5.0359005294126995e17, 4.282989132250602e-6),
            (8.523009112110578e17, 2.1697803811832315e-7),
        ];
        for (a, b) in cases {
            let lower = inv_beta_reg(a, b, 1e-310);
            let upper = inv_beta_reg(a, b, 1e-300);
            assert!(lower <= upper, "a={a}, b={b}, lower={lower}, upper={upper}");
        }
    }

    #[test]
    fn test_inv_beta_reg_log_solver_boundary_is_monotone() {
        let probability = 1e-8_f64;
        let probabilities = [
            f64::from_bits(probability.to_bits() - 1),
            probability,
            f64::from_bits(probability.to_bits() + 1),
        ];
        let cases = [
            (
                2.0,
                200.0,
                [
                    f64::from_bits(0x3ea7ab27fd13660a),
                    f64::from_bits(0x3ea7ab27fd13660b),
                    f64::from_bits(0x3ea7ab27fd13660b),
                ],
            ),
            (
                0.1,
                500.0,
                [
                    f64::from_bits(0x2eb79df9fcc6b8b8),
                    f64::from_bits(0x2eb79df9fcc6b8c3),
                    f64::from_bits(0x2eb79df9fcc6b8ce),
                ],
            ),
            (3.508179849994976e17, 0.8360747930277879, [1.0; 3]),
        ];
        for (a, b, expected) in cases {
            let actual = probabilities.map(|p| inv_beta_reg(a, b, p));
            assert!(
                actual[0] <= actual[1] && actual[1] <= actual[2],
                "a={a}, b={b}, actual={actual:?}"
            );
            for ((value, reference), probability) in
                actual.into_iter().zip(expected).zip(probabilities)
            {
                let ulp_error = value.to_bits().abs_diff(reference.to_bits());
                assert!(
                    ulp_error <= 256,
                    "a={a}, b={b}, probability={probability}, value={value}, reference={reference}, ulp_error={ulp_error}"
                );
                let quantile_relative_error = ((value - reference) / reference).abs();
                assert!(
                    quantile_relative_error <= 4e-14,
                    "a={a}, b={b}, probability={probability}, value={value}, reference={reference}, quantile_relative_error={quantile_relative_error}"
                );
                if value > 0.0 && value < 1.0 {
                    let relative_error =
                        ((beta_reg(a, b, value) - probability) / probability).abs();
                    assert!(
                        relative_error <= 1e-14,
                        "a={a}, b={b}, probability={probability}, value={value}, relative_error={relative_error}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_inv_beta_reg_adjacent_probability_is_monotone() {
        let probability = 1e-8_f64;
        let probabilities = [
            f64::from_bits(probability.to_bits() - 1),
            probability,
            f64::from_bits(probability.to_bits() + 1),
        ];
        let cases = [
            (
                9.11327743985456,
                133_525_174_076_797.34,
                f64::from_bits(0x3cf3d7e149ac36dd),
            ),
            (
                6.078046923216118,
                31_131_628_187_944.344,
                f64::from_bits(0x3cf592225b607c93),
            ),
        ];
        for (a, b, expected) in cases {
            let actual = probabilities.map(|p| inv_beta_reg(a, b, p));
            assert!(
                actual[0] <= actual[1] && actual[1] <= actual[2],
                "a={a}, b={b}, actual={actual:?}"
            );
            for value in actual {
                assert!(value.to_bits().abs_diff(expected.to_bits()) <= 256);
            }
        }
    }

    #[test]
    fn test_inv_beta_reg_upper_adjacent_probability_is_monotone() {
        let cases = [
            (
                100.0,
                1e6,
                [0x3feffffffffffff9, 0x3feffffffffffffa],
                [0x3f2a6e8528d3e729, 0x3f2a78942066b3b0],
            ),
            (
                1000.0,
                1e6,
                [0x3feffffffffffff7, 0x3feffffffffffff8],
                [0x3f54d1ec0e95e0f5, 0x3f54d42ffc3c17aa],
            ),
            (
                1000.0,
                1e6,
                [0x3feffffffffffffb, 0x3feffffffffffffc],
                [0x3f54dd318598d8ed, 0x3f54e1735a4b5c03],
            ),
            (
                1000.0,
                1e6,
                [0x3feffffffffffffd, 0x3feffffffffffffe],
                [0x3f54e6ebec74e0ca, 0x3f54ee997db90e85],
            ),
            (
                1000.0,
                1e8,
                [0x3feffffffffffff3, 0x3feffffffffffff4],
                [0x3eeaa4df95604c33, 0x3eeaa6db7106f8eb],
            ),
        ];
        for (a, b, probability_bits, expected_bits) in cases {
            let actual = probability_bits.map(|bits| inv_beta_reg(a, b, f64::from_bits(bits)));
            assert!(actual[0] <= actual[1]);
            for (value, expected) in actual.into_iter().zip(expected_bits.map(f64::from_bits)) {
                let ulp_error = value.to_bits().abs_diff(expected.to_bits());
                assert!(
                    ulp_error <= 512,
                    "a={a}, b={b}, value={value}, expected={expected}, ulp_error={ulp_error}"
                );
            }
        }
    }

    #[test]
    fn test_inv_beta_reg_orientation_preserves_tiny_quantiles() {
        let cases = [
            (0.49, f64::from_bits(0x083429b7deb4de35)),
            (0.5, f64::from_bits(0x0a0650cbd0bac729)),
            (0.51, f64::from_bits(0x0bd08de62d4b3d17)),
            (0.9, f64::from_bits(0x3f064452047719b0)),
            (0.99, 1.0),
        ];
        let mut previous = 0.0;
        for (probability, expected) in cases {
            let actual = inv_beta_reg(0.001, 0.01, probability);
            assert!(actual >= previous);
            if expected == 1.0 {
                assert_eq!(actual, expected);
            } else {
                assert!(((actual - expected) / expected).abs() <= 1e-12);
            }
            previous = actual;
        }
        let actual = inv_beta_reg(0.01, 1e8, 0.51);
        let expected = f64::from_bits(0x38260460ad60f7d3);
        assert!(((actual - expected) / expected).abs() <= 1e-12);
    }

    #[test]
    fn test_inv_beta_reg_concentrated_quantiles_round_correctly() {
        let cases = [
            (
                5.6337457945398355e35,
                3.4148653071385907e36,
                0.1,
                f64::from_bits(0x3fc2206894075924),
            ),
            (
                5.6337457945398355e35,
                3.4148653071385907e36,
                0.9,
                f64::from_bits(0x3fc2206894075924),
            ),
            (
                7.778370008599511e35,
                3.99094171205976e36,
                f64::from_bits(1),
                f64::from_bits(0x3fc4e0cc7f8ea39f),
            ),
            (
                7.778370008599511e35,
                3.99094171205976e36,
                0.1,
                f64::from_bits(0x3fc4e0cc7f8ea3a0),
            ),
            (
                7.778370008599511e35,
                3.99094171205976e36,
                0.9,
                f64::from_bits(0x3fc4e0cc7f8ea3a0),
            ),
        ];
        for (a, b, probability, expected) in cases {
            assert_eq!(inv_beta_reg(a, b, probability), expected);
        }
    }

    #[test]
    fn test_inv_beta_reg_extreme_tail_balanced_shapes() {
        let cases = [
            (f64::from_bits(1), 0.1384383837250825),
            (1e-300, 0.14764444133469024),
        ];
        for (probability, expected) in cases {
            let actual = inv_beta_reg(1000.0, 1000.0, probability);
            let relative_error = ((actual - expected) / expected).abs();
            assert!(
                relative_error <= 5e-13,
                "probability {probability}, actual {actual}, expected {expected}, relative error {relative_error}"
            );
        }
    }

    #[test]
    fn test_inv_beta_reg_extreme_tail_imbalanced_shapes() {
        let cases = [
            (200.0, 2.0, 1e-192, 0.10683857283574616),
            (1000.0, 2.0, f64::from_bits(1), 0.47203081850113066),
            (1000.0, 2.0, 1e-303, 0.49464719057284383),
            (1000.0, 2.0, 1e-200, 0.627230829476228),
            (1000.0, 2.0, 1e-100, 0.7900887907081466),
            (1000.0, 10.0, f64::from_bits(1), 0.454569346824437),
            (1000.0, 10.0, 1e-303, 0.47650393899531424),
            (1000.0, 10.0, 1e-200, 0.6055787273511661),
            (1000.0, 10.0, 1e-100, 0.7659557362087095),
            (1000.0, 100.0, f64::from_bits(1), 0.356892489498544),
            (1000.0, 100.0, 1e-303, 0.3750351205470552),
            (1000.0, 100.0, 1e-200, 0.48455098775995836),
            (1000.0, 100.0, 1e-100, 0.6303764215497716),
            (7_627_209.761, 11.3319, 1.679e-274, 0.9999105965110135),
        ];
        for (a, b, probability, expected) in cases {
            let actual = inv_beta_reg(a, b, probability);
            let relative_error = ((actual - expected) / expected).abs();
            assert!(
                relative_error <= 5e-13,
                "inv_beta_reg({a}, {b}, {probability}) = {actual}, expected {expected}, relative error {relative_error}"
            );
        }
    }

    #[test]
    fn test_inv_beta_reg_subnormal_power_series_boundary() {
        let a = f64::from_bits(0x4024000000000000);
        let b = f64::from_bits(0x7e37e43c8800759c);
        let probability = f64::from_bits(0x2df5ed8667733d64);
        for offset in -2_i64..=2 {
            let probability = f64::from_bits(probability.to_bits().wrapping_add_signed(offset));
            assert_eq!(
                inv_beta_reg(a, b, probability).to_bits(),
                0x000730d67819e860,
                "offset={offset}"
            );
        }
    }

    #[test]
    fn test_error_is_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<BetaFuncError>();
    }
}
