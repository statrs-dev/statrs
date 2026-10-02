use super::{
    BetaFuncError, beta_continued_fraction, beta_gamma_limit_applicable, beta_reg_use_complement,
    checked_beta_reg, gamma_upper_tail_with_beta_correction,
    large_params::{self, LogPrefactor},
    ln_beta,
};
use crate::function::gamma;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

const MAX_ITERATIONS: usize = 128;

#[derive(Clone, Copy)]
struct Point {
    log_x: f64,
    x: f64,
    log_cdf: f64,
}

fn log_regularized_beta(a: f64, b: f64, x: f64, log_beta: f64) -> Result<f64, BetaFuncError> {
    if x == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if x == 1.0 {
        return Ok(0.0);
    }

    if beta_reg_use_complement(a, b, x) {
        return checked_beta_reg(a, b, x).map(f64::ln);
    }

    let fraction = beta_continued_fraction(a, b, x)?;
    let log_prefactor = match large_params::log_prefactor(a, b, x) {
        Some(LogPrefactor::Value(parts)) => parts.0 + parts.1,
        Some(LogPrefactor::Underflow) => f64::NEG_INFINITY,
        None => a * x.ln() + b * (-x).ln_1p() - log_beta,
    };
    Ok(log_prefactor + (fraction / a).ln())
}

fn closer_point(lower: Point, upper: Point, log_probability: f64) -> f64 {
    let lower_error = (lower.log_cdf - log_probability).abs();
    let upper_error = (upper.log_cdf - log_probability).abs();
    if lower_error <= upper_error {
        lower.x
    } else {
        upper.x
    }
}

fn representable_midpoint(lower: f64, upper: f64) -> Option<f64> {
    let lower_bits = lower.to_bits();
    let distance = upper.to_bits() - lower_bits;
    (distance > 1).then(|| f64::from_bits(lower_bits + distance / 2))
}

fn closest_representable(
    a: f64,
    b: f64,
    log_beta: f64,
    lower: Point,
    upper: Point,
    log_probability: f64,
) -> Result<f64, BetaFuncError> {
    let mut best = lower.x;
    let mut best_error = (lower.log_cdf - log_probability).abs();
    for bits in lower.x.to_bits() + 1..=upper.x.to_bits() {
        let x = f64::from_bits(bits);
        let error = (log_regularized_beta(a, b, x, log_beta)? - log_probability).abs();
        if error < best_error {
            best = x;
            best_error = error;
        }
    }
    Ok(best)
}

pub(super) fn inverse_log_beta(a: f64, b: f64) -> f64 {
    if b == 1.0 {
        return -a.ln();
    }
    if b == 2.0 {
        return -a.ln() - (a + 1.0).ln();
    }
    if a == 1.0 {
        return -b.ln();
    }
    if a == 2.0 {
        return -b.ln() - (b + 1.0).ln();
    }
    if a.min(b) >= 10.0 {
        let scale = a.max(b);
        let scaled_a = a / scale;
        let scaled_b = b / scale;
        let x = scaled_a / (scaled_a + scaled_b);
        if let Some(LogPrefactor::Value(prefactor)) = large_params::log_prefactor(a, b, x) {
            return a * x.ln() + b * (-x).ln_1p() - prefactor.0 - prefactor.1;
        }
    }
    let smaller = a.min(b);
    let larger = a.max(b);
    if larger >= 10.0 {
        let ratio = smaller / larger;
        let gamma_difference = -smaller * larger.ln() - (larger + smaller - 0.5) * ratio.ln_1p()
            + smaller
            + large_params::stirling_correction(larger)
            - large_params::stirling_correction(larger + smaller);
        return gamma::ln_gamma(smaller) + gamma_difference;
    }
    ln_beta(a, b)
}

fn solve_lower_tail(a: f64, b: f64, probability: f64) -> Result<f64, BetaFuncError> {
    let log_probability = probability.ln();
    let log_beta = inverse_log_beta(a, b);
    let smallest = f64::from_bits(1);
    let smallest_log = smallest.ln();
    let smallest_log_cdf = log_regularized_beta(a, b, smallest, log_beta)?;

    if smallest_log_cdf > log_probability {
        return Ok(0.0);
    }
    if smallest_log_cdf == log_probability {
        return Ok(smallest);
    }

    let mut lower = Point {
        log_x: smallest_log,
        x: smallest,
        log_cdf: smallest_log_cdf,
    };
    let mut upper = Point {
        log_x: 0.0,
        x: 1.0,
        log_cdf: 0.0,
    };
    let estimate = (log_probability + a.ln() + log_beta) / a;
    let mut log_x = estimate.clamp(lower.log_x, upper.log_x);
    if log_x == lower.log_x || log_x == upper.log_x {
        log_x = lower.log_x + 0.5 * (upper.log_x - lower.log_x);
    }
    let mut x = log_x.exp();

    for _ in 0..MAX_ITERATIONS {
        let log_x = x.ln();
        let log_cdf = log_regularized_beta(a, b, x, log_beta)?;
        let current = Point { log_x, x, log_cdf };

        if log_cdf < log_probability {
            lower = current;
        } else if log_cdf > log_probability {
            upper = current;
        } else {
            return Ok(x);
        }

        if upper.x.to_bits() - lower.x.to_bits() <= 8 {
            return closest_representable(a, b, log_beta, lower, upper, log_probability);
        }
        let log_tolerance = 32.0 * f64::EPSILON * log_probability.abs().max(1.0);
        if (lower.log_cdf - log_probability).abs() <= log_tolerance
            && (upper.log_cdf - log_probability).abs() <= log_tolerance
        {
            return Ok(closer_point(lower, upper, log_probability));
        }

        let log_density = (a - 1.0) * log_x + (b - 1.0) * (-x).ln_1p() - log_beta;
        let slope = (log_x + log_density - log_cdf).exp();
        let newton = log_x - (log_cdf - log_probability) / slope;
        let midpoint = lower.log_x + 0.5 * (upper.log_x - lower.log_x);
        let next = if newton.is_finite() && newton > lower.log_x && newton < upper.log_x {
            newton
        } else {
            midpoint
        };

        let next_x = next.exp();
        x = if next_x <= lower.x || next_x >= upper.x {
            let Some(value) = representable_midpoint(lower.x, upper.x) else {
                return closest_representable(a, b, log_beta, lower, upper, log_probability);
            };
            value
        } else {
            next_x
        };
    }

    Err(BetaFuncError::ConvergenceFailed)
}

fn solve_gamma_upper_tail(a: f64, b: f64, probability: f64) -> Result<f64, BetaFuncError> {
    let log_tail = (-probability).ln_1p();
    let mut lower = 0.0;
    let mut upper = 1.0 / b;
    while gamma_upper_tail_with_beta_correction(a, b, upper)?.ln() > log_tail {
        lower = upper;
        upper *= 2.0;
    }

    while let Some(midpoint) = representable_midpoint(lower, upper) {
        let log_survival = gamma_upper_tail_with_beta_correction(a, b, midpoint)?.ln();
        if log_survival > log_tail {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }

    let lower_error = (gamma_upper_tail_with_beta_correction(a, b, lower)?.ln() - log_tail).abs();
    let upper_error = (gamma_upper_tail_with_beta_correction(a, b, upper)?.ln() - log_tail).abs();
    Ok(if lower_error <= upper_error {
        lower
    } else {
        upper
    })
}

pub(super) fn try_inv_beta_reg(a: f64, b: f64, probability: f64) -> Result<f64, BetaFuncError> {
    if !a.is_finite() || a <= 0.0 {
        return Err(BetaFuncError::ANotGreaterThanZero);
    }
    if !b.is_finite() || b <= 0.0 {
        return Err(BetaFuncError::BNotGreaterThanZero);
    }
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(BetaFuncError::XOutOfRange);
    }

    if probability == 0.0 {
        return Ok(0.0);
    }
    if probability == 1.0 {
        return Ok(1.0);
    }
    if probability == 0.5 && a == b {
        return Ok(0.5);
    }
    if probability == 0.5 && a.min(b) >= 1e16 {
        let scale = a.max(b);
        let scaled_a = a / scale;
        let scaled_b = b / scale;
        let sum = scaled_a + scaled_b;
        let mean = scaled_a / sum;
        let median_correction = ((scaled_a - scaled_b) / (3.0 * sum * sum)) / scale;
        return Ok(mean + median_correction);
    }
    if a == 1.0 {
        return Ok(-((-probability).ln_1p() / b).exp_m1());
    }
    if b == 1.0 {
        return Ok((probability.ln() / a).exp());
    }
    if probability <= 0.5 {
        let midpoint_below_one = f64::EPSILON / 4.0;
        if b < probability * a * midpoint_below_one {
            return Ok(1.0);
        }
        let reflected_probability = 1.0 - probability;
        if b < probability * a * f64::EPSILON.sqrt() {
            if reflected_probability == 1.0 {
                return Ok(1.0);
            }
            let value = if beta_gamma_limit_applicable(b, a) {
                solve_gamma_upper_tail(b, a, reflected_probability)?
            } else {
                solve_lower_tail(b, a, reflected_probability)?
            };
            return Ok(1.0 - value);
        }
        return solve_lower_tail(a, b, probability);
    }
    if a < b * f64::EPSILON.sqrt() {
        if beta_gamma_limit_applicable(a, b) {
            return solve_gamma_upper_tail(a, b, probability);
        }
        return solve_lower_tail(a, b, probability);
    }

    solve_lower_tail(b, a, 1.0 - probability).map(|value| 1.0 - value)
}

pub(super) fn inv_beta_reg(a: f64, b: f64, probability: f64) -> f64 {
    try_inv_beta_reg(a, b, probability)
        .unwrap_or_else(|error| panic!("inv_beta_reg failed: {error}"))
}
