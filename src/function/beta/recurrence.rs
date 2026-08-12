use super::*;

pub(super) fn beta_a_step(a: f64, b: f64, x: f64, steps: usize) -> f64 {
    let power = beta_reg_log_power_parts(a, b, x);
    (power.0 + power.1 + beta_a_step_log_sum(a, b, x, steps) - a.ln()).exp()
}

pub(super) fn beta_a_step_log_sum(a: f64, b: f64, x: f64, steps: usize) -> f64 {
    let mut log_sum = 0.0_f64;
    let mut log_term = 0.0_f64;
    let log_x = x.ln();
    for i in 0..steps.saturating_sub(1) {
        let i = i as f64;
        log_term += (a + b + i).ln() + log_x - (a + i + 1.0).ln();
        let maximum = log_sum.max(log_term);
        log_sum = maximum + (log_sum.min(log_term) - maximum).exp().ln_1p();
    }
    log_sum
}

pub(super) fn beta_a_step_log_parts(
    a: f64,
    b: f64,
    x: f64,
    steps: usize,
    log_beta: (f64, f64),
) -> (f64, f64) {
    let power = beta_reg_log_power_parts_accurate_with_log_beta(a, b, x, log_beta);
    let log_a = accurate_ln(a);
    dd_add(
        dd_add(power, (beta_a_step_log_sum(a, b, x, steps), 0.0)),
        (-log_a.0, -log_a.1),
    )
}

pub(super) fn beta_a_step_log(a: f64, b: f64, x: f64, steps: usize, log_beta: (f64, f64)) -> f64 {
    let result = beta_a_step_log_parts(a, b, x, steps, log_beta);
    result.0 + result.1
}
