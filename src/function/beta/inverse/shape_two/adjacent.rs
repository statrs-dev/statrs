use super::super::super::*;
use super::super::inverse_beta_adjacent_result;
use super::value::{direct_cdf_and_pdf, log_cdf_parts};

#[derive(Copy, Clone, PartialEq, Eq)]
enum ErrorScale {
    Series,
    Tail,
    Power,
    Log,
}

#[derive(Copy, Clone)]
struct Endpoint {
    value: f64,
    error: f64,
    scale: ErrorScale,
}

#[derive(Copy, Clone)]
struct Evaluation {
    endpoint: Endpoint,
    pdf: Option<f64>,
}

fn evaluate(a: f64, b: f64, value: f64, probability: f64, log_target: (f64, f64)) -> Evaluation {
    let (error, pdf, scale) = if let Some((cdf, pdf)) = direct_cdf_and_pdf(a, b, value) {
        let scale = if b == 2.0 {
            ErrorScale::Power
        } else if value < 0.5 && b * value < 0.5 {
            ErrorScale::Series
        } else {
            ErrorScale::Tail
        };
        (dd_add(cdf, (-probability, 0.0)), Some(pdf), scale)
    } else {
        (
            dd_add(log_cdf_parts(a, b, value), (-log_target.0, -log_target.1)),
            None,
            ErrorScale::Log,
        )
    };
    Evaluation {
        endpoint: Endpoint {
            value,
            error: error.0 + error.1,
            scale,
        },
        pdf,
    }
}

fn pair_result(
    a: f64,
    b: f64,
    mut lower: Endpoint,
    mut upper: Endpoint,
    log_target: (f64, f64),
) -> f64 {
    if lower.scale != upper.scale {
        for endpoint in [&mut lower, &mut upper] {
            let error = dd_add(
                log_cdf_parts(a, b, endpoint.value),
                (-log_target.0, -log_target.1),
            );
            endpoint.error = error.0 + error.1;
        }
    }
    inverse_beta_adjacent_result(lower.value, upper.value, lower.error, upper.error)
}

fn neighboring_result(
    a: f64,
    b: f64,
    probability: f64,
    current: Endpoint,
    neighbor: f64,
    log_target: (f64, f64),
) -> Option<f64> {
    let neighbor = evaluate(a, b, neighbor, probability, log_target).endpoint;
    if current.error * neighbor.error > 0.0 {
        return None;
    }
    let (lower, upper) = if neighbor.value < current.value {
        (neighbor, current)
    } else {
        (current, neighbor)
    };
    Some(pair_result(a, b, lower, upper, log_target))
}

pub(super) fn adjacent_result(a: f64, b: f64, probability: f64, mut current: f64) -> f64 {
    let log_target = accurate_ln(probability);
    let mut lower = Endpoint {
        value: 0.0,
        error: f64::NEG_INFINITY,
        scale: ErrorScale::Log,
    };
    let mut upper = Endpoint {
        value: 1.0,
        error: -log_target.0 - log_target.1,
        scale: ErrorScale::Log,
    };
    for _ in 0..64 {
        if current == 0.0 || current == 1.0 {
            return current;
        }
        let evaluation = evaluate(a, b, current, probability, log_target);
        let endpoint = evaluation.endpoint;
        if endpoint.error < 0.0 {
            lower = endpoint;
        } else {
            upper = endpoint;
        }
        if upper.value.to_bits().abs_diff(lower.value.to_bits()) == 1 {
            return pair_result(a, b, lower, upper, log_target);
        }
        let step = if let Some(pdf) = evaluation.pdf {
            endpoint.error / pdf
        } else {
            let log_pdf = if b == 2.0 {
                (a - 1.0).mul_add(current.ln(), a.ln() + (a + 1.0).ln() + (-current).ln_1p())
            } else {
                (b - 1.0).mul_add((-current).ln_1p(), b.ln() + (b + 1.0).ln() + current.ln())
            };
            endpoint.error * ((log_target.0 + log_target.1) - log_pdf).exp()
        };
        let candidate = current - step;
        if candidate == current {
            let neighbor = f64::from_bits(if endpoint.error > 0.0 {
                current.to_bits() - 1
            } else {
                current.to_bits() + 1
            });
            if let Some(result) =
                neighboring_result(a, b, probability, endpoint, neighbor, log_target)
            {
                return result;
            }
            current = neighbor;
            continue;
        }
        let next = if candidate.is_finite() && candidate > lower.value && candidate < upper.value {
            candidate
        } else {
            lower.value + 0.5 * (upper.value - lower.value)
        };
        if next == current {
            let neighbor = f64::from_bits(if endpoint.error > 0.0 {
                current.to_bits() - 1
            } else {
                current.to_bits() + 1
            });
            if let Some(result) =
                neighboring_result(a, b, probability, endpoint, neighbor, log_target)
            {
                return result;
            }
            current = lower.value + 0.5 * (upper.value - lower.value);
        } else {
            current = next;
        }
    }
    panic!(
        "shape-two inverse did not resolve adjacent values for a={a}, b={b}, probability={probability}"
    )
}
