use super::*;

const ACCURATE_SMALL_B_MAX_RATIO: f64 = 1e-4;

pub(super) fn use_beta_small_b_shifted_accurate(a: f64, b: f64, y: f64) -> bool {
    (0.0..1.0).contains(&a) && b <= ACCURATE_SMALL_B_MAX_RATIO * a && y < 0.3
}

pub(super) fn beta_small_b_large_a_factor(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
) -> Result<(f64, f64), BetaFuncError> {
    let bm1 = b - 1.0;
    let t = a + 0.5 * bm1;
    let lx = if y < 0.35 { (-y).ln_1p() } else { x.ln() };
    let u = -t * lx;
    let log_h = b * u.ln() - u - ln_gamma_stable(b);
    let log_prefix = log_h + ln_gamma_delta(a, b) - b * t.ln();

    let mut odd_factorials = [1.0; 30];
    let mut factorial = 1.0;
    for k in 1..=59 {
        factorial *= k as f64;
        if k >= 3 && k % 2 == 1 {
            odd_factorials[(k - 3) as usize / 2] = factorial;
        }
    }

    let mut coefficients = [0.0; 30];
    coefficients[0] = 1.0;
    let mut j = if u >= SCALED_GAMMA_MIN_X {
        upper_gamma_scaled_asymptotic(b, u)?
    } else if u > 1.0 {
        upper_gamma_scaled_continued_fraction(b, u)?
    } else if b <= 1e-4 && u <= 1.0 {
        upper_gamma_scaled_small_shape(b, u)?
    } else {
        gamma::gamma_ur(b, u) / log_h.exp()
    };
    let mut sum = j;
    let mut compensation = 0.0_f64;
    let lx2 = (0.5 * lx) * (0.5 * lx);
    let mut lx_power = 1.0;
    let t4 = 4.0 * t * t;
    let mut b_plus_2n = b;
    let mut converged = false;

    for n in 1..30 {
        let n_f64 = n as f64;
        let mut coefficient = 0.0;
        for m in 1..n {
            coefficient += (m as f64 * b - n_f64) * coefficients[n - m] / odd_factorials[m - 1];
        }
        coefficient /= n_f64;
        coefficient += bm1 / odd_factorials[n - 1];
        coefficients[n] = coefficient;

        j = (b_plus_2n * (b_plus_2n + 1.0) * j + (u + b_plus_2n + 1.0) * lx_power) / t4;
        lx_power *= lx2;
        b_plus_2n += 2.0;
        let term = coefficient * j;
        let corrected = term - compensation;
        let next = sum + corrected;
        compensation = (next - sum) - corrected;
        sum = next;
        if term.abs() <= prec::F64_PREC * sum.abs() {
            converged = true;
            break;
        }
    }

    if converged && sum > 0.0 {
        Ok((log_prefix, sum))
    } else {
        Err(BetaFuncError::ConvergenceFailed)
    }
}

pub(super) fn beta_small_b_large_a_series(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
    initial: f64,
) -> Result<f64, BetaFuncError> {
    let (log_prefix, factor) = beta_small_b_large_a_factor(a, b, x, y)?;
    let sum = initial + log_prefix.exp() * factor;
    if (0.0..=1.0).contains(&sum) {
        Ok(sum)
    } else {
        Err(BetaFuncError::ConvergenceFailed)
    }
}

pub(super) fn beta_small_b_large_a_series_log(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
    initial: f64,
) -> Result<f64, BetaFuncError> {
    let (log_prefix, factor) = beta_small_b_large_a_factor(a, b, x, y)?;
    let tail = log_prefix + factor.ln();
    if initial == 0.0 {
        Ok(tail)
    } else {
        let initial = initial.ln();
        let maximum = initial.max(tail);
        Ok(maximum + (initial.min(tail) - maximum).exp().ln_1p())
    }
}

pub(super) fn beta_reg_small_b_shifted_log(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
    log_beta: (f64, f64),
) -> Result<f64, BetaFuncError> {
    let steps = (10.0 - a).ceil() as usize;
    let shifted = a + steps as f64;
    let shifted_log = beta_small_b_large_a_series_log(shifted, b, x, y, 0.0)?;
    let recurrence_log = beta_a_step_log(a, b, x, steps, log_beta);
    let maximum = shifted_log.max(recurrence_log);
    Ok(maximum + (shifted_log.min(recurrence_log) - maximum).exp().ln_1p())
}

pub(super) fn beta_reg_small_b_shifted_accurate(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
) -> Result<(f64, (f64, f64)), BetaFuncError> {
    let steps = (10.0 - a).ceil() as usize;
    let shifted = a + steps as f64;
    let (_, factor) = beta_small_b_large_a_factor(shifted, b, x, y)?;
    let t = shifted + 0.5 * (b - 1.0);
    let lx = if y < 0.35 { (-y).ln_1p() } else { x.ln() };
    let u = -t * lx;
    let log_b = accurate_ln(b);
    let log_gamma_b = dd_add((ln_gamma_one_plus_series(b), 0.0), (-log_b.0, -log_b.1));
    let log_h = dd_add(
        dd_add(dd_mul((b, 0.0), accurate_ln(u)), (-u, 0.0)),
        (-log_gamma_b.0, -log_gamma_b.1),
    );
    let gamma_delta = ln_gamma_delta_accurate_parts(shifted, b);
    let b_log_t = dd_mul((b, 0.0), accurate_ln(t));
    let log_prefix = dd_add(dd_add(log_h, gamma_delta), (-b_log_t.0, -b_log_t.1));
    let shifted_value = dd_exp(log_prefix) * factor;
    let shifted_log = dd_add(log_prefix, accurate_ln(factor));
    let gamma = ln_gamma_accurate_parts(b);
    let gamma_delta = ln_gamma_delta_accurate_parts(a, b);
    let log_beta = dd_add(gamma, (-gamma_delta.0, -gamma_delta.1));
    let recurrence_log = beta_a_step_log_parts(a, b, x, steps, log_beta);
    let recurrence_value = dd_exp(recurrence_log);
    let value = dd_add((shifted_value, 0.0), (recurrence_value, 0.0));
    let (maximum, minimum) = if shifted_log.0 > recurrence_log.0 {
        (shifted_log, recurrence_log)
    } else {
        (recurrence_log, shifted_log)
    };
    let difference = dd_add(minimum, (-maximum.0, -maximum.1));
    let correction = accurate_ln_one_plus_dd((dd_exp(difference), 0.0));
    let logarithm = dd_add(maximum, correction);
    Ok((value.0 + value.1, logarithm))
}

pub(super) fn beta_reg_small_b_large_a(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
) -> Result<Option<f64>, BetaFuncError> {
    if a < 10.0 || b >= 40.0 || y >= 0.3 {
        return Ok(None);
    }
    let mut steps = b.floor() as usize;
    if b == steps as f64 {
        steps -= 1;
    }
    let reduced_b = b - steps as f64;
    let initial = if steps == 0 {
        0.0
    } else {
        beta_a_step(reduced_b, a, y, steps)
    };
    beta_small_b_large_a_series(a, reduced_b, x, y, initial).map(Some)
}

pub(super) fn beta_reg_small_b_large_a_log(
    a: f64,
    b: f64,
    x: f64,
    y: f64,
) -> Result<Option<f64>, BetaFuncError> {
    if a < 10.0 || b >= 40.0 || y >= 0.3 {
        return Ok(None);
    }
    let mut steps = b.floor() as usize;
    if b == steps as f64 {
        steps -= 1;
    }
    let reduced_b = b - steps as f64;
    let initial = if steps == 0 {
        0.0
    } else {
        beta_a_step(reduced_b, a, y, steps)
    };
    beta_small_b_large_a_series_log(a, reduced_b, x, y, initial).map(Some)
}
