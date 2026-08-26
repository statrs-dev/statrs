use super::{
    BetaFuncError, beta_continued_fraction, checked_beta_reg,
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

    let symmetry_split = (a + 1.0) / (a + b + 2.0);
    if x >= symmetry_split {
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

fn inverse_log_beta(a: f64, b: f64) -> f64 {
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

fn solve_lower_tail(a: f64, b: f64, probability: f64) -> f64 {
    let log_probability = probability.ln();
    let log_beta = inverse_log_beta(a, b);
    let smallest = f64::from_bits(1);
    let smallest_log = smallest.ln();
    let smallest_log_cdf = log_regularized_beta(a, b, smallest, log_beta)
        .unwrap_or_else(|error| panic!("inv_beta_reg evaluation failed: {error}"));

    if smallest_log_cdf > log_probability {
        return 0.0;
    }
    if smallest_log_cdf == log_probability {
        return smallest;
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

    for _ in 0..MAX_ITERATIONS {
        let x = log_x.exp();
        let log_cdf = log_regularized_beta(a, b, x, log_beta)
            .unwrap_or_else(|error| panic!("inv_beta_reg evaluation failed: {error}"));
        let current = Point { log_x, x, log_cdf };

        if log_cdf < log_probability {
            lower = current;
        } else if log_cdf > log_probability {
            upper = current;
        } else {
            return x;
        }

        if upper.x.to_bits() - lower.x.to_bits() <= 8 {
            return closest_representable(a, b, log_beta, lower, upper, log_probability)
                .unwrap_or_else(|error| panic!("inv_beta_reg evaluation failed: {error}"));
        }
        let log_tolerance = 32.0 * f64::EPSILON * log_probability.abs().max(1.0);
        if (lower.log_cdf - log_probability).abs() <= log_tolerance
            && (upper.log_cdf - log_probability).abs() <= log_tolerance
        {
            return closer_point(lower, upper, log_probability);
        }

        let log_density = (a - 1.0) * log_x + (b - 1.0) * (-x).ln_1p() - log_beta;
        let slope = (log_x + log_density - log_cdf).exp();
        let newton = log_x - (log_cdf - log_probability) / slope;
        let midpoint = lower.log_x + 0.5 * (upper.log_x - lower.log_x);
        let mut next = if newton.is_finite() && newton > lower.log_x && newton < upper.log_x {
            newton
        } else {
            midpoint
        };

        if next.exp() == x {
            next = midpoint;
        }
        log_x = next;
    }

    panic!(
        "inv_beta_reg did not converge for a={a}, b={b}, probability={probability}, lower={:?}, upper={:?}, lower_log_cdf={:?}, upper_log_cdf={:?}",
        lower.x, upper.x, lower.log_cdf, upper.log_cdf
    )
}

pub(super) fn inv_beta_reg(a: f64, b: f64, probability: f64) -> f64 {
    assert!(a.is_finite() && a > 0.0, "a must be finite and positive");
    assert!(b.is_finite() && b > 0.0, "b must be finite and positive");
    assert!(
        probability.is_finite() && (0.0..=1.0).contains(&probability),
        "probability must be finite and in [0, 1]"
    );

    if probability == 0.0 {
        return 0.0;
    }
    if probability == 1.0 {
        return 1.0;
    }
    if probability == 0.5 && a == b {
        return 0.5;
    }
    if probability <= 0.5 {
        return solve_lower_tail(a, b, probability);
    }

    1.0 - solve_lower_tail(b, a, 1.0 - probability)
}
