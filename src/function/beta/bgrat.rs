use super::*;

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
