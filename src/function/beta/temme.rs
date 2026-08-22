#[cfg(all(not(feature = "std"), not(test)))]
use super::Float;
use super::{dd_add, dd_div, dd_mul};

// DLMF 8.18.9--12 gives the uniform expansion and c0. The centered series and
// c1 recurrence follow Temme, Special Functions (1996), section 11.3.3.2.

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

pub(super) fn temme_coefficients(mean: f64, delta: f64) -> (f64, f64) {
    let complement = 1.0 - mean;
    let variance_root = (mean * complement).sqrt();
    let mut deviance = [0.0; SERIES_DEGREE + 1];
    let mut mean_power = mean;
    let mut complement_power = complement;
    for (index, coefficient) in deviance.iter_mut().enumerate() {
        let order = index + 2;
        let sign = if order & 1 == 0 { 1.0 } else { -1.0 };
        *coefficient =
            2.0 * mean * complement * (sign * mean_power.recip() + complement_power.recip())
                / order as f64;
        mean_power *= mean;
        complement_power *= complement;
    }

    let eta_over_delta = series_sqrt(&deviance);
    let delta_over_eta = series_reciprocal(&eta_over_delta);
    let mut c0_coefficients = [0.0; SERIES_DEGREE + 1];
    for index in 0..SERIES_DEGREE {
        c0_coefficients[index] = variance_root * delta_over_eta[index + 1];
    }

    let mut eta_derivative = [0.0; SERIES_DEGREE + 1];
    for index in 0..=SERIES_DEGREE {
        eta_derivative[index] = (index + 1) as f64 * eta_over_delta[index];
    }
    let mut eta_derivative_reciprocal = series_reciprocal(&eta_derivative);
    for coefficient in &mut eta_derivative_reciprocal {
        *coefficient *= variance_root;
    }
    let mut c0_delta_derivative = [0.0; SERIES_DEGREE + 1];
    for index in 0..SERIES_DEGREE {
        c0_delta_derivative[index] = (index + 1) as f64 * c0_coefficients[index + 1];
    }
    let c0_eta_derivative = series_multiply(&c0_delta_derivative, &eta_derivative_reciprocal);
    let mut divided_numerator = [0.0; SERIES_DEGREE + 1];
    for index in 0..SERIES_DEGREE {
        divided_numerator[index] = -c0_eta_derivative[index + 1];
    }
    let mut c1_coefficients = series_multiply(&delta_over_eta, &divided_numerator);
    let stirling = (1.0 - mean.recip() - complement.recip()) / 12.0;
    for index in 0..=SERIES_DEGREE {
        c1_coefficients[index] =
            variance_root.mul_add(c1_coefficients[index], -stirling * c0_coefficients[index]);
    }

    (
        evaluate(&c0_coefficients, delta),
        evaluate(&c1_coefficients, delta),
    )
}

pub(super) fn temme_delta(a: f64, b: f64, x: f64) -> (f64, (f64, f64)) {
    let scale = a.max(b);
    let scaled_a = (a / scale, (-a / scale).mul_add(scale, a) / scale);
    let scaled_b = (b / scale, (-b / scale).mul_add(scale, b) / scale);
    let scaled_sum = dd_add(scaled_a, scaled_b);
    let mean = dd_div(scaled_a, scaled_sum);
    let numerator = dd_add(dd_mul((x, 0.0), scaled_sum), (-scaled_a.0, -scaled_a.1));
    let delta = dd_div(numerator, scaled_sum);
    (mean.0 + mean.1, delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficients_match_symbolic_reference() {
        let (c0, c1) = temme_coefficients(1.0 / 3.0, -2.7216552652253867e-5);
        assert!((c0 - 0.23571910018211537).abs() < 2e-15, "c0={c0:?}");
        assert!((c1 - -0.0360076116745462).abs() < 2e-14, "c1={c1:?}");
    }

    #[test]
    fn compensated_delta_matches_exact_input_reference() {
        let (mean, delta) = temme_delta(40_000_000.0, 80_000_000.0, 1.0 / 3.0);
        assert_eq!(mean, 1.0 / 3.0);
        assert!((delta.0 + delta.1 - -1.850371707708594e-17).abs() < 1e-32);
    }
}
