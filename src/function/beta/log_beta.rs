use super::*;

/// Computes the natural logarithm of the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter
/// and `a > 0`, `b > 0`.
///
/// # Panics
///
/// if `a <= 0.0` or `b <= 0.0`
pub fn ln_beta(a: f64, b: f64) -> f64 {
    checked_ln_beta(a, b).unwrap()
}

/// Computes the natural logarithm
/// of the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter
/// and `a > 0`, `b > 0`.
///
/// # Errors
///
/// if `a <= 0.0` or `b <= 0.0`
pub fn checked_ln_beta(a: f64, b: f64) -> Result<f64, BetaFuncError> {
    if a <= 0.0 {
        Err(BetaFuncError::ANotGreaterThanZero)
    } else if b <= 0.0 {
        Err(BetaFuncError::BNotGreaterThanZero)
    } else {
        Ok(ln_beta_stable(a, b))
    }
}

pub(super) fn stirling_correction(x: f64) -> f64 {
    let reciprocal = 1.0 / x;
    let x2 = reciprocal * reciprocal;
    reciprocal
        * (1.0 / 12.0
            + x2 * (-1.0 / 360.0
                + x2 * (1.0 / 1260.0
                    + x2 * (-1.0 / 1680.0 + x2 * (1.0 / 1188.0 - x2 * 691.0 / 360360.0)))))
}

pub(super) fn stirling_correction_log(log_x: f64) -> f64 {
    let reciprocal = (-log_x).exp();
    let x2 = reciprocal * reciprocal;
    reciprocal
        * (1.0 / 12.0
            + x2 * (-1.0 / 360.0
                + x2 * (1.0 / 1260.0
                    + x2 * (-1.0 / 1680.0 + x2 * (1.0 / 1188.0 - x2 * 691.0 / 360360.0)))))
}

pub(super) fn ln_gamma_delta(base: f64, delta: f64) -> f64 {
    let log_ratio = (delta / base).ln_1p();
    let log_sum = base.ln() + log_ratio;
    delta * base.ln() + base.mul_add(log_ratio, (delta - 0.5) * log_ratio) - delta
        + stirling_correction_log(log_sum)
        - stirling_correction(base)
}

pub(super) fn ln_gamma_stable(x: f64) -> f64 {
    if x < 0.5 {
        gamma::ln_gamma(1.0 + x) - x.ln()
    } else {
        gamma::ln_gamma(x)
    }
}

pub(super) fn ln_gamma_stirling_parts(value: (f64, f64)) -> (f64, f64) {
    let shifted = dd_add(value, (-0.5, 0.0));
    let mut result = dd_mul(shifted, accurate_ln_dd(value));
    result = dd_add(result, (-value.0, -value.1));
    result = dd_add(result, (consts::LN_SQRT_2PI, -3.8782941580672414e-17));
    dd_add(result, (stirling_correction(value.0), 0.0))
}

pub(super) fn ln_gamma_accurate_parts(x: f64) -> (f64, f64) {
    if x == 1.0 || x == 2.0 {
        return (0.0, 0.0);
    }
    if x <= 0.125 {
        let mut result = dd_add((x, 0.0), (1.0, 0.0));
        let mut recurrence = (0.0, 0.0);
        while result.0 < STIRLING_MIN {
            recurrence = dd_add(recurrence, accurate_ln_dd(result));
            result = dd_add(result, (1.0, 0.0));
        }
        let gamma_one_plus = dd_add(
            ln_gamma_stirling_parts(result),
            (-recurrence.0, -recurrence.1),
        );
        let logarithm = accurate_ln(x);
        return dd_add(gamma_one_plus, (-logarithm.0, -logarithm.1));
    }

    let mut shifted = (x, 0.0);
    let mut recurrence = (0.0, 0.0);
    while shifted.0 < STIRLING_MIN {
        recurrence = dd_add(recurrence, accurate_ln_dd(shifted));
        shifted = dd_add(shifted, (1.0, 0.0));
    }
    let result = ln_gamma_stirling_parts(shifted);
    dd_add(result, (-recurrence.0, -recurrence.1))
}

pub(super) fn ln_gamma_fast_accurate(x: f64) -> f64 {
    if x <= 0.125 {
        ln_gamma_one_plus_series(x) - x.ln()
    } else {
        ln_gamma_stable(x)
    }
}

pub(super) fn ln_gamma_delta_parts(base: f64, delta: f64) -> (f64, f64) {
    let base_log = accurate_ln(base);
    let ratio = dd_div_f64((delta, 0.0), base);
    let log_ratio = accurate_ln_one_plus_dd(ratio);
    let mut result = dd_mul((delta, 0.0), base_log);
    result = dd_add(result, dd_mul((base, 0.0), log_ratio));
    result = dd_add(result, dd_mul((delta - 0.5, 0.0), log_ratio));
    result = dd_add(result, (-delta, 0.0));
    result = dd_add(result, (stirling_correction(base + delta), 0.0));
    dd_add(result, (-stirling_correction(base), 0.0))
}

pub(super) fn ln_gamma_delta_accurate_parts(base: f64, delta: f64) -> (f64, f64) {
    if base >= STIRLING_MIN {
        return ln_gamma_delta_parts(base, delta);
    }
    let steps = (STIRLING_MIN - base).ceil() as usize;
    let mut result = ln_gamma_delta_parts(base + steps as f64, delta);
    for step in 0..steps {
        let ratio = dd_div_f64((delta, 0.0), base + step as f64);
        let logarithm = accurate_ln_one_plus_dd(ratio);
        result = dd_add(result, (-logarithm.0, -logarithm.1));
    }
    result
}

fn ln_beta_accurate_parts_impl(a: f64, b: f64, fast_small_gamma: bool) -> (f64, f64) {
    let gamma_parts = |x| {
        if fast_small_gamma && x <= 0.125 {
            ln_gamma_small_accurate_parts(x)
        } else {
            ln_gamma_accurate_parts(x)
        }
    };
    let smaller = a.min(b);
    let larger = a.max(b);
    if larger >= STIRLING_MIN && (smaller < STIRLING_MIN || smaller <= 0.25 * larger) {
        let gamma = gamma_parts(smaller);
        let delta = ln_gamma_delta_parts(larger, smaller);
        return dd_add(gamma, (-delta.0, -delta.1));
    }
    if a + b == f64::INFINITY {
        return (ln_beta_stable(a, b), 0.0);
    }
    let gamma_a = gamma_parts(a);
    let gamma_b = gamma_parts(b);
    let gamma_sum = gamma_parts(a + b);
    dd_add(dd_add(gamma_a, gamma_b), (-gamma_sum.0, -gamma_sum.1))
}

pub(super) fn ln_beta_accurate_parts(a: f64, b: f64) -> (f64, f64) {
    ln_beta_accurate_parts_impl(a, b, false)
}

pub(super) fn ln_beta_inverse_accurate_parts(a: f64, b: f64) -> (f64, f64) {
    ln_beta_accurate_parts_impl(a, b, true)
}

pub(super) fn ln_a_beta_small_first_shape_parts(a: f64, b: f64) -> (f64, f64) {
    let gamma_one_plus = ln_gamma_one_plus_series_parts(a);
    let gamma_delta = ln_gamma_delta_parts(b, a);
    dd_add(gamma_one_plus, (-gamma_delta.0, -gamma_delta.1))
}

pub(super) fn ln_beta_stable_parts(a: f64, b: f64) -> (f64, f64) {
    let smaller = a.min(b);
    let larger = a.max(b);
    if larger >= STIRLING_MIN && (smaller < STIRLING_MIN || smaller <= 0.25 * larger) {
        ln_beta_accurate_parts(a, b)
    } else {
        (ln_beta_stable(a, b), 0.0)
    }
}

pub(super) fn ln_beta_inverse_parts(a: f64, b: f64) -> (f64, f64) {
    let smaller = a.min(b);
    let larger = a.max(b);
    if larger >= STIRLING_MIN && (smaller < STIRLING_MIN || smaller <= 0.25 * larger) {
        ln_beta_inverse_accurate_parts(a, b)
    } else {
        (ln_beta_stable(a, b), 0.0)
    }
}

pub(super) fn imbalanced_ln_beta(a: f64, b: f64) -> Option<f64> {
    let smaller = a.min(b);
    let larger = a.max(b);
    if larger >= STIRLING_MIN && smaller < STIRLING_MIN {
        Some(ln_gamma_stable(smaller) - ln_gamma_delta(larger, smaller))
    } else if smaller <= 1e-8 * larger {
        Some(ln_gamma_stable(smaller) - smaller * gamma::digamma(larger))
    } else {
        None
    }
}

pub(super) fn ln_beta_stable(a: f64, b: f64) -> f64 {
    if a.min(b) <= 0.125 {
        if a.max(b) >= STIRLING_MIN {
            let result = ln_beta_accurate_parts(a, b);
            return result.0 + result.1;
        }
        if a.max(b) <= 0.125 {
            return (a + b).ln() - a.ln() - b.ln()
                + ln_gamma_one_plus_series(a)
                + ln_gamma_one_plus_series(b)
                - ln_gamma_one_plus_series(a + b);
        }
        return ln_gamma_fast_accurate(a) + ln_gamma_fast_accurate(b) - ln_gamma_stable(a + b);
    }
    if let Some(ln_beta) = imbalanced_ln_beta(a, b) {
        return ln_beta;
    }
    if a < STIRLING_MIN || b < STIRLING_MIN {
        return ln_gamma_stable(a) + ln_gamma_stable(b) - ln_gamma_stable(a + b);
    }

    let (mean, complement, log_sum, _) = beta_shape_statistics(a, b);
    a * mean.ln()
        + b * complement.ln()
        + consts::LN_SQRT_2PI
        + 0.5 * (log_sum - a.ln() - b.ln())
        + stirling_correction(a)
        + stirling_correction(b)
        - stirling_correction_log(log_sum)
}
