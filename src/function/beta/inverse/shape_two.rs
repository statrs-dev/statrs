//! Shape-two identities follow from DLMF 8.17.7--8.17.8 and symmetry 8.17.4.

use super::super::*;
use super::inverse_beta_adjacent_result;
use value::{direct_cdf_and_pdf, fast_cdf_and_pdf, log_cdf, log_cdf_parts};

mod value;

fn error_parts(a: f64, b: f64, x: f64, probability: f64) -> ((f64, f64), Option<f64>) {
    if let Some((cdf, pdf)) = direct_cdf_and_pdf(a, b, x) {
        return (dd_add(cdf, (-probability, 0.0)), Some(pdf));
    }
    let log_target = accurate_ln(probability);
    (
        dd_add(log_cdf_parts(a, b, x), (-log_target.0, -log_target.1)),
        None,
    )
}

fn adjacent_result(a: f64, b: f64, probability: f64, mut current: f64) -> f64 {
    let mut lower = 0.0;
    let mut upper = 1.0;
    let mut lower_error = f64::NEG_INFINITY;
    let mut upper_error = f64::INFINITY;
    for _ in 0..64 {
        if current == 0.0 || current == 1.0 {
            return current;
        }
        let (current_error, pdf) = error_parts(a, b, current, probability);
        let error = current_error.0 + current_error.1;
        if error < 0.0 {
            lower = current;
            lower_error = error;
        } else {
            upper = current;
            upper_error = error;
        }
        if upper.to_bits().abs_diff(lower.to_bits()) == 1 {
            return inverse_beta_adjacent_result(lower, upper, lower_error, upper_error);
        }
        let step = if let Some(pdf) = pdf {
            error / pdf
        } else {
            let log_target = accurate_ln(probability);
            let log_pdf = if b == 2.0 {
                (a - 1.0).mul_add(current.ln(), (a * (a + 1.0)).ln() + (-current).ln_1p())
            } else {
                (b - 1.0).mul_add((-current).ln_1p(), (b * (b + 1.0)).ln() + current.ln())
            };
            error * ((log_target.0 + log_target.1) - log_pdf).exp()
        };
        let candidate = current - step;
        if candidate == current {
            let neighbor = if error > 0.0 {
                f64::from_bits(current.to_bits() - 1)
            } else {
                f64::from_bits(current.to_bits() + 1)
            };
            let (neighbor_error, _) = error_parts(a, b, neighbor, probability);
            let neighbor_error = neighbor_error.0 + neighbor_error.1;
            if error * neighbor_error <= 0.0 {
                return if error > 0.0 {
                    inverse_beta_adjacent_result(neighbor, current, neighbor_error, error)
                } else {
                    inverse_beta_adjacent_result(current, neighbor, error, neighbor_error)
                };
            }
            current = neighbor;
            continue;
        }
        let next = if candidate.is_finite() && candidate > lower && candidate < upper {
            candidate
        } else {
            lower + 0.5 * (upper - lower)
        };
        if next == current {
            let neighbor = if error > 0.0 {
                f64::from_bits(current.to_bits() - 1)
            } else {
                f64::from_bits(current.to_bits() + 1)
            };
            let (neighbor_error, _) = error_parts(a, b, neighbor, probability);
            let neighbor_error = neighbor_error.0 + neighbor_error.1;
            if error * neighbor_error <= 0.0 {
                return if error > 0.0 {
                    inverse_beta_adjacent_result(neighbor, current, neighbor_error, error)
                } else {
                    inverse_beta_adjacent_result(current, neighbor, error, neighbor_error)
                };
            }
            current = lower + 0.5 * (upper - lower);
        } else {
            current = next;
        }
    }
    panic!(
        "shape-two inverse did not resolve adjacent values for a={a}, b={b}, probability={probability}"
    )
}

pub(super) fn inverse_beta_shape_two(a: f64, b: f64, probability: f64) -> f64 {
    let mut current = if b == 2.0 {
        ((probability.ln() - (a + 1.0).ln()) / a).exp()
    } else {
        (0.5 * (probability.ln() + core::f64::consts::LN_2 - (b * (b + 1.0)).ln())).exp()
    };
    current = current.clamp(f64::from_bits(1), f64::from_bits(1.0_f64.to_bits() - 1));
    let mut lower = 0.0;
    let mut upper = 1.0;
    for _ in 0..128 {
        if let Some((cdf, pdf)) = fast_cdf_and_pdf(a, b, current) {
            let error = cdf - probability;
            if error == 0.0 {
                return adjacent_result(a, b, probability, current);
            }
            if error < 0.0 {
                lower = current;
            } else {
                upper = current;
            }
            if upper.to_bits().abs_diff(lower.to_bits()) == 1 {
                return adjacent_result(a, b, probability, current);
            }
            let step = error / pdf;
            let pdf_ratio = (a - 1.0) / current - (b - 1.0) / (1.0 - current);
            let denominator = 1.0 - 0.5 * step * pdf_ratio;
            let candidate = current - step / denominator;
            if candidate == current {
                return adjacent_result(a, b, probability, current);
            }
            let next = if denominator > 0.0 && candidate > lower && candidate < upper {
                candidate
            } else {
                lower + 0.5 * (upper - lower)
            };
            if next == current {
                return adjacent_result(a, b, probability, current);
            }
            current = next;
            continue;
        }
        let target = accurate_ln(probability);
        let log_value = log_cdf(a, b, current);
        let error = log_value - (target.0 + target.1);
        if error == 0.0 {
            return adjacent_result(a, b, probability, current);
        }
        if error < 0.0 {
            lower = current;
        } else {
            upper = current;
        }
        if upper.to_bits().abs_diff(lower.to_bits()) == 1 {
            return adjacent_result(a, b, probability, current);
        }
        let log_pdf = if b == 2.0 {
            (a - 1.0).mul_add(current.ln(), (a * (a + 1.0)).ln() + (-current).ln_1p())
        } else {
            (b - 1.0).mul_add((-current).ln_1p(), (b * (b + 1.0)).ln() + current.ln())
        };
        let inverse_derivative = (log_value - log_pdf).exp();
        let step = error * inverse_derivative;
        let log_pdf_derivative = (a - 1.0) / current - (b - 1.0) / (1.0 - current);
        let denominator = 1.0 - 0.5 * step * (log_pdf_derivative - 1.0 / inverse_derivative);
        let candidate = current - step / denominator;
        if candidate == current {
            return adjacent_result(a, b, probability, current);
        }
        let next = if denominator > 0.0 && candidate > lower && candidate < upper {
            candidate
        } else {
            lower + 0.5 * (upper - lower)
        };
        if next == current {
            return adjacent_result(a, b, probability, current);
        }
        current = next;
    }
    panic!("shape-two inverse did not converge for a={a}, b={b}, probability={probability}")
}

#[cfg(test)]
mod tests;
