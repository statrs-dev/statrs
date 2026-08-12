use super::super::*;

pub(super) fn lower_tail_initial(a: f64, b: f64, probability: f64, ln_beta: f64) -> (f64, f64) {
    let log_initial = (probability.ln() + a.ln() + ln_beta) / a;
    let initial = log_initial.exp();
    let initial = if initial == 0.0 {
        0.0
    } else if initial < 1.0 {
        initial
    } else {
        let (mean, _, _, _) = beta_shape_statistics(a, b);
        if mean < 1.0 {
            mean
        } else {
            f64::from_bits(1.0_f64.to_bits() - 1)
        }
    };
    (initial, log_initial)
}

pub(super) fn lower_tail_initial_accurate(
    a: f64,
    probability: f64,
    log_beta: (f64, f64),
) -> (f64, (f64, f64)) {
    let mut logarithm = accurate_ln(probability);
    logarithm = dd_add(logarithm, accurate_ln(a));
    logarithm = dd_add(logarithm, log_beta);
    logarithm = dd_div_f64(logarithm, a);
    (dd_exp(logarithm), logarithm)
}

pub(super) fn inverse_beta_initial(a: f64, b: f64, probability: f64, ln_beta: f64) -> (f64, f64) {
    if a > 1.0 && b > 1.0 && (probability >= 1e-4 || a.min(b) >= STIRLING_MIN) {
        let normal_tail = (-2.0 * probability.ln()).sqrt();
        let normal_quantile = normal_tail
            - (2.30753 + 0.27061 * normal_tail)
                / (1.0 + (0.99229 + 0.04481 * normal_tail) * normal_tail);
        let correction = (normal_quantile * normal_quantile - 3.0) / 6.0;
        let reciprocal_a = 1.0 / (2.0 * a - 1.0);
        let reciprocal_b = 1.0 / (2.0 * b - 1.0);
        let scale = 2.0 / (reciprocal_a + reciprocal_b);
        let w = normal_quantile * (scale + correction).sqrt() / scale
            - (reciprocal_b - reciprocal_a) * (correction + 5.0 / 6.0 - 2.0 / (3.0 * scale));
        let log_ratio = b.ln() - a.ln() + 2.0 * w;
        let initial = if log_ratio > 0.0 {
            let reciprocal = (-log_ratio).exp();
            reciprocal / (1.0 + reciprocal)
        } else {
            1.0 / (1.0 + log_ratio.exp())
        };
        if initial > 0.0 && initial < 1.0 {
            return (initial, f64::NAN);
        }
    }

    lower_tail_initial(a, b, probability, ln_beta)
}
