//! Shape-two identities follow from DLMF 8.17.7--8.17.8 and symmetry 8.17.4.

use super::super::*;
use adjacent::adjacent_result;
use value::{fast_cdf_and_pdf, log_cdf};

mod adjacent;
mod value;

#[cfg(test)]
use value::log_cdf_parts;

pub(super) fn inverse_beta_shape_two(a: f64, b: f64, probability: f64) -> f64 {
    let mut current = if b == 2.0 {
        ((probability.ln() - (a + 1.0).ln()) / a).exp()
    } else {
        (0.5 * (probability.ln() + core::f64::consts::LN_2 - b.ln() - (b + 1.0).ln())).exp()
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
            (a - 1.0).mul_add(current.ln(), a.ln() + (a + 1.0).ln() + (-current).ln_1p())
        } else {
            (b - 1.0).mul_add((-current).ln_1p(), b.ln() + (b + 1.0).ln() + current.ln())
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
