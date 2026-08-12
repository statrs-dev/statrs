use super::*;

mod bounds;

use bounds::{central_cell_is_certified, extreme_ratio_cell_is_certified};

fn beta_mean(a: f64, b: f64, mean: f64) -> (f64, f64) {
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
    let residual =
        difference + (difference_error + scaled_a_error - product_error - mean * scaled_sum_error);
    (mean, residual / scaled_sum)
}

pub(super) fn beta_concentrated_quantile(a: f64, b: f64, probability: f64) -> Option<f64> {
    let (mean, complement, _, root_sum) = beta_shape_statistics(a, b);
    let spacing = if mean == 0.0 {
        f64::from_bits(1)
    } else if mean == 1.0 {
        1.0 - f64::from_bits(1.0_f64.to_bits() - 1)
    } else {
        let lower_spacing = mean - f64::from_bits(mean.to_bits() - 1);
        let upper_spacing = f64::from_bits(mean.to_bits() + 1) - mean;
        lower_spacing.min(upper_spacing)
    };
    let standard_deviation = (mean * complement).sqrt() / root_sum;
    if 64.0 * standard_deviation >= 0.5 * spacing {
        return None;
    }

    let mean = beta_mean(a, b, mean);
    let normal_quantile = -core::f64::consts::SQRT_2 * erf::erfc_inv(2.0 * probability);
    let reciprocal_sum = (1.0 / root_sum) / root_sum;
    let skew_correction =
        (complement - mean.0) * normal_quantile.mul_add(normal_quantile, -1.0) * reciprocal_sum
            / 3.0;
    let offset = normal_quantile.mul_add(standard_deviation, mean.1 + skew_correction);
    let candidate = (mean.0 + offset).clamp(0.0, 1.0);

    if a.min(b) >= ASYMPTOTIC_MIN_SHAPE && mean.0.min(complement) >= 0.1 {
        central_cell_is_certified(a, b, probability, mean, candidate).then_some(candidate)
    } else {
        extreme_ratio_cell_is_certified(a, b, probability, mean, candidate).then_some(candidate)
    }
}
