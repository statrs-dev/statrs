use super::*;

pub(super) fn log1mexp(x: f64) -> f64 {
    if x < -core::f64::consts::LN_2 {
        (-x.exp()).ln_1p()
    } else {
        (-x.exp_m1()).ln()
    }
}

pub(crate) fn checked_ln_beta_reg_complement(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    if a <= 0.0 {
        return Err(BetaFuncError::ANotGreaterThanZero);
    }
    if b <= 0.0 {
        return Err(BetaFuncError::BNotGreaterThanZero);
    }
    if !(0.0..=1.0).contains(&x) {
        return Err(BetaFuncError::XOutOfRange);
    }
    if x == 1.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if a <= f64::EPSILON.sqrt() && b >= STIRLING_MIN && x.powf(a) > 0.5 {
        let log_cdf = checked_ln_beta_reg(a, b, x)?;
        return Ok(log1mexp(log_cdf));
    }
    if use_beta_symmetry(a, b, x) {
        let y = 1.0 - x;
        if use_beta_power_series(b, a, y) {
            return beta_power_series_log(b, a, y);
        }
    }
    let log_cdf = checked_ln_beta_reg(a, b, x)?;
    if log_cdf < -core::f64::consts::LN_2 {
        Ok(log1mexp(log_cdf))
    } else {
        checked_ln_beta_reg(b, a, 1.0 - x)
    }
}

pub(super) fn checked_ln_beta_reg_with_log_beta(
    a: f64,
    b: f64,
    x: f64,
    log_beta: Option<(f64, f64)>,
) -> Result<f64, BetaFuncError> {
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
        return Ok(f64::NEG_INFINITY);
    }
    if x == 1.0 {
        return Ok(0.0);
    }
    if a == b && x == 0.5 {
        return Ok(-core::f64::consts::LN_2);
    }
    if b == 1.0 {
        return Ok(a * x.ln());
    }
    if a == 1.0 {
        return Ok((-(b * (-x).ln_1p()).exp_m1()).ln());
    }
    let y = 1.0 - x;
    if let Some((log_result, invert)) =
        beta_small_shapes_series_log_with_log_beta(a, b, x, y, log_beta)?
    {
        return Ok(if invert {
            log1mexp(log_result)
        } else {
            log_result
        });
    }
    if let Some(result) = beta_reg_asymptotic(a, b, x) {
        return Ok(result.ln());
    }
    if a.mul_add(y, -(b * x)) >= 0.0
        && let Some(result) = beta_reg_small_b_large_a_log(a, b, x, y)?
    {
        return Ok(result);
    }
    if (0.0..1.0).contains(&a) && b <= f64::EPSILON.sqrt() * a && y < 0.3 {
        let result = beta_reg_small_b_shifted_accurate(a, b, x, y)?.1;
        return Ok(result.0 + result.1);
    }
    if (1.0..10.0).contains(&a) && b < 1.0 && y < 0.3 {
        return beta_reg_small_b_shifted_log(a, b, x, y, ln_beta_accurate_parts(a, b));
    }
    let symm_transform =
        !use_beta_power_series_before_symmetry(a, b, x) && use_beta_symmetry(a, b, x);
    let (transformed_a, transformed_b, transformed_x, transformed_y) = if symm_transform {
        (b, a, y, x)
    } else {
        (a, b, x, y)
    };
    if !use_exact_complement_continued_fraction(a, b, symm_transform)
        && let Some(log_tail) = beta_reg_small_b_large_a_log(
            transformed_a,
            transformed_b,
            transformed_x,
            transformed_y,
        )?
    {
        return Ok(if symm_transform {
            log1mexp(log_tail)
        } else {
            log_tail
        });
    }
    if use_beta_power_series(transformed_a, transformed_b, transformed_x) {
        let log_result = beta_power_series_log_parts_with_log_beta(
            transformed_a,
            transformed_b,
            transformed_x,
            log_beta,
        )?;
        let log_result = log_result.0 + log_result.1;
        return Ok(if symm_transform {
            log1mexp(log_result)
        } else {
            log_result
        });
    }

    let log_power = if let Some(log_beta) = log_beta {
        beta_reg_log_power_parts_with_log_beta(a, b, x, log_beta)
    } else {
        beta_reg_log_power_parts(a, b, x)
    };
    if symm_transform && (log_power.0 + log_power.1).exp() == 0.0 {
        return Ok(0.0);
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
    let smaller = a.min(b);
    let larger = a.max(b);
    let log_fraction = if fraction.1 != 0.0
        || (larger >= STIRLING_MIN && (smaller < STIRLING_MIN || smaller <= 0.25 * larger))
    {
        accurate_ln_dd(fraction)
    } else {
        (fraction.0.ln(), 0.0)
    };
    let log_result = dd_add(log_power, (-log_fraction.0, -log_fraction.1));
    let log_result = log_result.0 + log_result.1;
    if symm_transform {
        Ok(log1mexp(log_result))
    } else {
        Ok(log_result)
    }
}
