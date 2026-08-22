use super::*;

pub(super) fn beta_reg_central_log_power_parts(a: f64, b: f64, x: f64) -> Option<(f64, f64)> {
    if a >= STIRLING_MIN && b >= STIRLING_MIN && 1.0 - x < 1.0 {
        let (residual, log_ratio) = beta_log_ratio(a, b, x);
        if residual.abs() <= 0.1 * a.min(b) {
            let (_, _, log_sum, _) = beta_shape_statistics(a, b);
            let log_scale = consts::LN_SQRT_2PI
                + 0.5 * (log_sum - a.ln() - b.ln())
                + stirling_correction(a)
                + stirling_correction(b)
                - stirling_correction_log(log_sum);
            return Some(two_sum(log_ratio, -log_scale));
        }
    }
    None
}

pub(super) fn beta_reg_log_power_parts_with_log_x(
    a: f64,
    b: f64,
    (log_x, log_x_error): (f64, f64),
    (log_y, log_y_error): (f64, f64),
    (log_beta, log_beta_error): (f64, f64),
) -> (f64, f64) {
    let a_log_x = a * log_x;
    let a_log_x_error = a.mul_add(log_x, -a_log_x) + a * log_x_error;
    let b_log_y = b * log_y;
    let b_log_y_error = b.mul_add(log_y, -b_log_y) + b * log_y_error;
    let (variable, variable_error) = two_sum(a_log_x, b_log_y);
    let variable_error = variable_error + a_log_x_error + b_log_y_error;
    let (result, result_error) = two_sum(variable, -log_beta);
    (result, result_error + variable_error - log_beta_error)
}

pub(super) fn beta_reg_log_power_parts(a: f64, b: f64, x: f64) -> (f64, f64) {
    beta_reg_central_log_power_parts(a, b, x).unwrap_or_else(|| {
        let smaller = a.min(b);
        let larger = a.max(b);
        if larger >= STIRLING_MIN && (smaller < STIRLING_MIN || smaller <= 0.25 * larger) {
            return beta_reg_log_power_parts_with_log_x(
                a,
                b,
                accurate_ln(x),
                accurate_ln_one_minus(x),
                ln_beta_accurate_parts(a, b),
            );
        }
        beta_reg_log_power_parts_with_log_x(
            a,
            b,
            compensated_ln(x),
            compensated_ln_one_minus(x),
            ln_beta_stable_parts(a, b),
        )
    })
}

pub(super) fn beta_reg_log_power_parts_with_log_beta(
    a: f64,
    b: f64,
    x: f64,
    log_beta: (f64, f64),
) -> (f64, f64) {
    beta_reg_central_log_power_parts(a, b, x).unwrap_or_else(|| {
        let smaller = a.min(b);
        let larger = a.max(b);
        if larger >= STIRLING_MIN && (smaller < STIRLING_MIN || smaller <= 0.25 * larger) {
            beta_reg_log_power_parts_with_log_x(
                a,
                b,
                accurate_ln(x),
                accurate_ln_one_minus(x),
                log_beta,
            )
        } else {
            beta_reg_log_power_parts_with_log_x(
                a,
                b,
                compensated_ln(x),
                compensated_ln_one_minus(x),
                log_beta,
            )
        }
    })
}

pub(super) fn beta_reg_log_power_parts_accurate(a: f64, b: f64, x: f64) -> (f64, f64) {
    beta_reg_log_power_parts_accurate_with_log_beta(a, b, x, ln_beta_accurate_parts(a, b))
}

pub(super) fn beta_reg_log_power_parts_accurate_with_log_beta(
    a: f64,
    b: f64,
    x: f64,
    log_beta: (f64, f64),
) -> (f64, f64) {
    beta_reg_central_log_power_parts(a, b, x).unwrap_or_else(|| {
        beta_reg_log_power_parts_with_log_x(
            a,
            b,
            accurate_ln(x),
            accurate_ln_one_minus(x),
            log_beta,
        )
    })
}
