// (C) Copyright John Maddock 2006.
// (C) Copyright Matt Borland 2024.
// SPDX-License-Identifier: MIT AND BSL-1.0
// Use, modification and distribution are subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE-BOOST.md or copy at
// https://www.boost.org/LICENSE_1_0.txt)

use super::*;

/// Computes the regularized lower incomplete beta function
/// `I_x(a,b) = 1/Beta(a,b) * int(t^(a-1)*(1-t)^(b-1), t=0..x)`
/// `a > 0`, `b > 0`, `1 >= x >= 0` where `a` is the first beta parameter,
/// `b` is the second beta parameter, and `x` is the upper limit of the
/// integral.
///
/// # Errors
///
/// if `a <= 0.0`, `b <= 0.0`, `x < 0.0`, or `x > 1.0`
pub fn checked_beta_reg(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    if a <= 0.0 {
        return Err(BetaFuncError::ANotGreaterThanZero);
    }

    if b <= 0.0 {
        return Err(BetaFuncError::BNotGreaterThanZero);
    }

    if !(0.0..=1.0).contains(&x) {
        return Err(BetaFuncError::XOutOfRange);
    }

    if x == 0.0 {
        return Ok(0.0);
    }
    if x == 1.0 {
        return Ok(1.0);
    }
    if a == b && x == 0.5 {
        return Ok(0.5);
    }
    if b == 1.0 {
        return Ok(x.powf(a));
    }
    if a == 1.0 {
        return Ok(-(b * (-x).ln_1p()).exp_m1());
    }
    let y = 1.0 - x;
    if let Some((log_result, invert)) = beta_small_shapes_series_log(a, b, x, y)? {
        let result = if invert {
            -log_result.exp_m1()
        } else {
            log_result.exp()
        };
        return if (0.0..=1.0).contains(&result) {
            Ok(result)
        } else {
            Err(BetaFuncError::ConvergenceFailed)
        };
    }
    if let Some(result) = beta_reg_asymptotic(a, b, x) {
        return Ok(result);
    }
    if a.mul_add(y, -(b * x)) >= 0.0
        && let Some(result) = beta_reg_small_b_large_a(a, b, x, y)?
    {
        return Ok(result);
    }
    if use_beta_small_b_shifted_accurate(a, b, y) {
        return beta_reg_small_b_shifted_accurate(a, b, x, y).map(|result| result.0);
    }
    if (1.0..10.0).contains(&a) && b < 1.0 && y < 0.3 {
        let result = beta_reg_small_b_shifted_log(a, b, x, y, ln_beta_accurate_parts(a, b))?.exp();
        return if (0.0..=1.0).contains(&result) {
            Ok(result)
        } else {
            Err(BetaFuncError::ConvergenceFailed)
        };
    }
    let symm_transform =
        !use_beta_power_series_before_symmetry(a, b, x) && use_beta_symmetry(a, b, x);
    let (transformed_a, transformed_b, transformed_x, transformed_y) = if symm_transform {
        (b, a, y, x)
    } else {
        (a, b, x, y)
    };
    if !use_exact_complement_continued_fraction(a, b, symm_transform)
        && let Some(tail) =
            beta_reg_small_b_large_a(transformed_a, transformed_b, transformed_x, transformed_y)?
    {
        return Ok(if symm_transform { 1.0 - tail } else { tail });
    }
    if use_beta_power_series(transformed_a, transformed_b, transformed_x) {
        let log_result = beta_power_series_log_parts(transformed_a, transformed_b, transformed_x)?;
        let result = if symm_transform {
            dd_negative_expm1(log_result)
        } else {
            (log_result.0 + log_result.1).exp()
        };
        return if (0.0..=1.0).contains(&result) {
            Ok(result)
        } else {
            Err(BetaFuncError::ConvergenceFailed)
        };
    }

    let lanczos_power = use_beta_reg_lanczos_power(transformed_a, transformed_b).then(|| {
        let coordinates = if symm_transform {
            (two_sum(1.0, -x), (x, 0.0))
        } else {
            ((x, 0.0), two_sum(1.0, -x))
        };
        beta_reg_lanczos_power(transformed_a, transformed_b, coordinates.0, coordinates.1)
    });
    let log_power = lanczos_power
        .is_none()
        .then(|| beta_reg_log_power_parts(a, b, x));
    let power = lanczos_power
        .map(|value| value.0 + value.1)
        .unwrap_or_else(|| {
            let value = log_power.unwrap();
            (value.0 + value.1).exp()
        });
    if power == 0.0 {
        return Ok(if symm_transform { 1.0 } else { 0.0 });
    }
    let fraction = beta_fraction_for_transformed_tail(
        a,
        b,
        x,
        transformed_a,
        transformed_b,
        transformed_x,
        symm_transform,
    )?;
    let accurate_fraction =
        1.0 - transformed_x == 1.0 || use_exact_complement_continued_fraction(a, b, symm_transform);
    let result = if let Some(lanczos_power) = lanczos_power {
        let tail = dd_div(lanczos_power, fraction);
        if symm_transform {
            dd_add((1.0, 0.0), (-tail.0, -tail.1)).0
        } else {
            tail.0 + tail.1
        }
    } else if accurate_fraction {
        let log_power = log_power.unwrap();
        let log_fraction = accurate_ln_dd(fraction);
        let log_result = dd_add(log_power, (-log_fraction.0, -log_fraction.1));
        if symm_transform {
            dd_negative_expm1(log_result)
        } else {
            dd_exp(log_result)
        }
    } else if symm_transform {
        1.0 - power / (fraction.0 + fraction.1)
    } else {
        power / (fraction.0 + fraction.1)
    };
    if (0.0..=1.0).contains(&result) {
        Ok(result)
    } else {
        Err(BetaFuncError::ConvergenceFailed)
    }
}
