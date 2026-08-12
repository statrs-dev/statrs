use super::super::*;

pub(super) fn inverse_beta_midpoint(lower: f64, upper: f64) -> f64 {
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

pub(super) fn inverse_beta_adjacent_result(
    lower: f64,
    upper: f64,
    lower_error: f64,
    upper_error: f64,
) -> f64 {
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

pub(super) fn inverse_beta_log_value_parts(
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

pub(super) fn inverse_beta_log_tail(
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
        Some(ln_beta_inverse_accurate_parts(a, b))
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

pub(super) fn inverse_beta_reflect(a: f64, b: f64, probability: f64, log_beta: (f64, f64)) -> bool {
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
