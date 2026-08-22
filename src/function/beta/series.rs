// (C) Copyright John Maddock 2006.
// (C) Copyright Matt Borland 2024.
// SPDX-License-Identifier: MIT AND BSL-1.0
// Use, modification and distribution are subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE-BOOST.md or copy at
// https://www.boost.org/LICENSE_1_0.txt)

use super::*;

pub(super) fn beta_power_series_log_parts_with_log_beta(
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

pub(super) fn beta_power_series_log_parts(
    a: f64,
    b: f64,
    x: f64,
) -> Result<(f64, f64), BetaFuncError> {
    beta_power_series_log_parts_with_log_beta(a, b, x, None)
}

pub(super) fn beta_power_series_log(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    beta_power_series_log_parts(a, b, x).map(|(result, error)| result + error)
}

pub(super) fn beta_small_shapes_series_log(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
) -> Result<Option<(f64, bool)>, BetaFuncError> {
    beta_small_shapes_series_log_with_log_beta(a, b, x, y, None)
}

pub(super) fn beta_small_shapes_series_log_with_log_beta(
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

pub(super) fn use_beta_gamma_limit(a: f64, b: f64, scaled_x: f64) -> bool {
    let correction_scale = a + scaled_x + 1.0;
    correction_scale.is_finite() && correction_scale / b.sqrt() <= 0.25 * f64::EPSILON.sqrt()
}

pub(super) fn use_beta_power_series(a: f64, b: f64, x: f64) -> bool {
    let scaled_x = b * x;
    x < 1.0
        && ((scaled_x <= 0.7 && x <= 0.95)
            || (a <= f64::EPSILON.sqrt() && scaled_x <= 2.0 && x < beta_symmetry_split(a, b))
            || (a <= 0.3 && b >= 32.0 && scaled_x <= 2.0)
            || (a <= 40.0 && b >= 32.0 && x < beta_symmetry_split(a, b))
            || (use_beta_gamma_limit(a, b, scaled_x) && scaled_x <= 64.0))
}

pub(super) fn use_beta_power_series_before_symmetry(a: f64, b: f64, x: f64) -> bool {
    let scaled_x = b * x;
    x < 1.0
        && !(a <= f64::EPSILON.sqrt() && b >= STIRLING_MIN && x.powf(a) > 0.5)
        && ((a <= f64::EPSILON.sqrt() && scaled_x <= 2.0 && x < beta_symmetry_split(a, b))
            || (a <= 0.3 && b >= 32.0 && scaled_x <= 2.0)
            || (a <= 40.0 && b >= 32.0 && x < beta_symmetry_split(a, b))
            || (use_beta_gamma_limit(a, b, scaled_x) && scaled_x <= 64.0))
}

pub(super) fn beta_symmetry_split(a: f64, b: f64) -> f64 {
    let a1 = a + 1.0;
    let b1 = b + 1.0;
    let scale = a1.max(b1);
    (a1 / scale) / (a1 / scale + b1 / scale)
}

pub(super) fn use_beta_symmetry(a: f64, b: f64, x: f64) -> bool {
    a < 1.0 && a <= f64::EPSILON.sqrt() && b >= STIRLING_MIN && x.powf(a) > 0.5
        || (a < 1.0 || x > f64::EPSILON) && 1.0 - x < 1.0 && x >= beta_symmetry_split(a, b)
}
