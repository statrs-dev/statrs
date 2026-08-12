mod initial;
mod solve;

use super::*;
use initial::*;
use solve::*;

/// Computes the inverse of the regularized incomplete beta function
pub fn inv_beta_reg(a: f64, b: f64, probability: f64) -> f64 {
    debug_assert!((0.0..=1.0).contains(&probability) && a > 0.0 && b > 0.0);

    if probability == 0.0 {
        return 0.0;
    }
    if probability == 1.0 {
        return 1.0;
    }
    if a == b && probability == 0.5 {
        return 0.5;
    }
    if let Some(quantile) = beta_concentrated_quantile(a, b, probability) {
        return quantile;
    }
    if b == 1.0 {
        return probability.powf(1.0 / a);
    }
    if a == 1.0 {
        return -((-probability).ln_1p() / b).exp_m1();
    }

    let log_beta = ln_beta_inverse_parts(a, b);
    let flip = inverse_beta_reflect(a, b, probability, log_beta);
    let (a, b, target) = if flip {
        (b, a, 1.0 - probability)
    } else {
        (a, b, probability)
    };
    let ln_beta = log_beta.0 + log_beta.1;
    let (mut current, mut log_initial) = inverse_beta_initial(a, b, target, ln_beta);
    let smaller = a.min(b);
    let larger = a.max(b);
    if log_initial.is_finite()
        && larger >= STIRLING_MIN
        && (smaller < STIRLING_MIN || smaller <= 0.25 * larger)
    {
        let accurate_initial = lower_tail_initial_accurate(a, target, log_beta);
        if accurate_initial.0 > 0.0 && accurate_initial.0 < 1.0 {
            current = accurate_initial.0;
            log_initial = accurate_initial.1.0 + accurate_initial.1.1;
            let first_correction = ((b - 1.0).abs() / (a + 1.0)) * current;
            let remainder_ratio = (b - 2.0).abs() * current;
            if first_correction <= f64::EPSILON / 32.0 && remainder_ratio <= 0.5 {
                return if flip { 1.0 - current } else { current };
            }
        }
    }
    let min_subnormal = f64::from_bits(1);
    if current == 0.0 && log_initial < min_subnormal.ln() - core::f64::consts::LN_2 {
        return if flip { 1.0 } else { 0.0 };
    }
    let first_correction = ((b - 1.0).abs() / (a + 1.0)) * current;
    let remainder_ratio = (b - 2.0).abs() * current;
    if first_correction <= f64::EPSILON / 32.0 && remainder_ratio <= 0.5 {
        return if flip { 1.0 - current } else { current };
    }
    if current < f64::MIN_POSITIVE {
        let relative_correction = b * current / (a + 1.0);
        let relative_half_ulp = 0.5 * (min_subnormal / current);
        if relative_correction < 0.25 * relative_half_ulp {
            return if flip { 1.0 - current } else { current };
        }
    }
    let result = inverse_beta_log_tail(a, b, target, current, log_beta, ln_beta);
    if flip { 1.0 - result } else { result }
}
