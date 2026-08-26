use super::double_double::{add, divide, exp, multiply, two_sum};
use super::large_params::log_ratio;
use crate::consts;
use crate::function::erf;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

const MIN_SUM: f64 = 1e5;
const MIN_SHAPE: f64 = 1e4;
const MAX_DEVIANCE: f64 = 9.0;
const SERIES_DEGREE: usize = 6;

fn series_multiply(
    left: &[f64; SERIES_DEGREE + 1],
    right: &[f64; SERIES_DEGREE + 1],
) -> [f64; SERIES_DEGREE + 1] {
    let mut result = [0.0; SERIES_DEGREE + 1];
    for index in 0..=SERIES_DEGREE {
        for offset in 0..=index {
            result[index] = left[offset].mul_add(right[index - offset], result[index]);
        }
    }
    result
}

fn series_reciprocal(value: &[f64; SERIES_DEGREE + 1]) -> [f64; SERIES_DEGREE + 1] {
    let mut result = [0.0; SERIES_DEGREE + 1];
    result[0] = 1.0 / value[0];
    for index in 1..=SERIES_DEGREE {
        let mut sum = 0.0;
        for offset in 1..=index {
            sum = value[offset].mul_add(result[index - offset], sum);
        }
        result[index] = -sum / value[0];
    }
    result
}

fn series_sqrt(value: &[f64; SERIES_DEGREE + 1]) -> [f64; SERIES_DEGREE + 1] {
    let mut result = [0.0; SERIES_DEGREE + 1];
    result[0] = value[0].sqrt();
    for index in 1..=SERIES_DEGREE {
        let mut sum = 0.0;
        for offset in 1..index {
            sum = result[offset].mul_add(result[index - offset], sum);
        }
        result[index] = (value[index] - sum) / (2.0 * result[0]);
    }
    result
}

fn evaluate(coefficients: &[f64], argument: f64) -> f64 {
    coefficients.iter().rev().fold(0.0, |value, coefficient| {
        value.mul_add(argument, *coefficient)
    })
}

fn coefficients(mean: f64, delta: f64) -> (f64, f64) {
    let complement = 1.0 - mean;
    let variance_root = (mean * complement).sqrt();
    let mut deviance = [0.0; SERIES_DEGREE + 1];
    let mut mean_power = mean;
    let mut complement_power = complement;
    for (index, coefficient) in deviance.iter_mut().enumerate() {
        let order = index + 2;
        let sign = if order & 1 == 0 { 1.0 } else { -1.0 };
        *coefficient =
            2.0 * mean * complement * (sign / mean_power + 1.0 / complement_power) / order as f64;
        mean_power *= mean;
        complement_power *= complement;
    }

    let eta_over_delta = series_sqrt(&deviance);
    let delta_over_eta = series_reciprocal(&eta_over_delta);
    let mut c0_coefficients = [0.0; SERIES_DEGREE + 1];
    c0_coefficients[..SERIES_DEGREE].copy_from_slice(&delta_over_eta[1..(SERIES_DEGREE + 1)]);
    for coefficient in &mut c0_coefficients {
        *coefficient *= variance_root;
    }

    let mut eta_derivative = [0.0; SERIES_DEGREE + 1];
    for index in 0..=SERIES_DEGREE {
        eta_derivative[index] = (index + 1) as f64 * eta_over_delta[index];
    }
    let mut inverse_eta_derivative = series_reciprocal(&eta_derivative);
    for coefficient in &mut inverse_eta_derivative {
        *coefficient *= variance_root;
    }
    let mut c0_delta_derivative = [0.0; SERIES_DEGREE + 1];
    for index in 0..SERIES_DEGREE {
        c0_delta_derivative[index] = (index + 1) as f64 * c0_coefficients[index + 1];
    }
    let c0_eta_derivative = series_multiply(&c0_delta_derivative, &inverse_eta_derivative);
    let mut divided_numerator = [0.0; SERIES_DEGREE + 1];
    for index in 0..SERIES_DEGREE {
        divided_numerator[index] = -c0_eta_derivative[index + 1];
    }
    let mut c1_coefficients = series_multiply(&delta_over_eta, &divided_numerator);
    let stirling = (1.0 - 1.0 / mean - 1.0 / complement) / 12.0;
    for index in 0..=SERIES_DEGREE {
        c1_coefficients[index] =
            variance_root.mul_add(c1_coefficients[index], -stirling * c0_coefficients[index]);
    }

    (
        evaluate(&c0_coefficients, delta),
        evaluate(&c1_coefficients, delta),
    )
}

fn mean_and_delta(a: f64, b: f64, x: f64) -> (f64, (f64, f64)) {
    let scale = a.max(b);
    let scaled_a = (a / scale, (-a / scale).mul_add(scale, a) / scale);
    let scaled_b = (b / scale, (-b / scale).mul_add(scale, b) / scale);
    let scaled_sum = add(scaled_a, scaled_b);
    let mean = divide(scaled_a, scaled_sum);
    let numerator = add(multiply((x, 0.0), scaled_sum), (-scaled_a.0, -scaled_a.1));
    (mean.0 + mean.1, divide(numerator, scaled_sum))
}

fn normal_tail(argument: (f64, f64)) -> f64 {
    let (value, error) = two_sum(argument.0, argument.1);
    let tail = 0.5 * erf::erfc(value);
    let derivative = -(-value * value).exp() / core::f64::consts::PI.sqrt();
    let corrected = add((tail, 0.0), (derivative * error, 0.0));
    corrected.0 + corrected.1
}

pub(super) fn beta_reg_temme(a: f64, b: f64, x: f64) -> Option<f64> {
    if a < MIN_SUM && b < MIN_SUM - a {
        return None;
    }
    let scale = a.max(b);
    let scaled_a = a / scale;
    let scaled_b = b / scale;
    let scaled_sum = scaled_a + scaled_b;
    let mean = scaled_a / scaled_sum;
    let complement = scaled_b / scaled_sum;
    if mean.min(complement) < 0.01 || (mean.min(complement) < 0.1 && a.min(b) < MIN_SHAPE) {
        return None;
    }

    let (residual, ratio) = log_ratio(a, b, x);
    let deviance_parts = (-ratio.0, -ratio.1);
    let deviance = deviance_parts.0 + deviance_parts.1;
    if !(0.0..=MAX_DEVIANCE).contains(&deviance) {
        return None;
    }

    let (series_mean, delta) = mean_and_delta(a, b, x);
    let (c0, c1) = coefficients(series_mean, delta.0 + delta.1);
    let root_sum = scale.sqrt() * scaled_sum.sqrt();
    let root_deviance = deviance_parts.0.sqrt();
    let root_error = (deviance_parts.1 + (-root_deviance).mul_add(root_deviance, deviance_parts.0))
        / (2.0 * root_deviance);
    let tail = if deviance == 0.0 {
        0.5
    } else {
        normal_tail((root_deviance, root_error))
    };
    let coefficient = (-c1 / (root_sum * root_sum)).mul_add(1.0, c0);
    let correction = exp(ratio) * coefficient / (consts::SQRT_2PI * root_sum);
    let result = if residual < 0.0 {
        let value = add((tail, 0.0), (correction, 0.0));
        value.0 + value.1
    } else {
        let complement = add((tail, 0.0), (-correction, 0.0));
        let value = add((1.0, 0.0), (-complement.0, -complement.1));
        value.0 + value.1
    };
    (0.0..=1.0).contains(&result).then_some(result)
}
