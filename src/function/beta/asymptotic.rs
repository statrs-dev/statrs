use super::*;

pub(super) fn beta_shape_statistics(a: f64, b: f64) -> (f64, f64, f64, f64) {
    let scale = a.max(b);
    let scaled_a = a / scale;
    let scaled_b = b / scale;
    let scaled_sum = scaled_a + scaled_b;
    let mean = scaled_a / scaled_sum;
    let complement = scaled_b / scaled_sum;
    let log_sum = scale.ln() + scaled_sum.ln();
    let root_sum = scale.sqrt() * scaled_sum.sqrt();
    (mean, complement, log_sum, root_sum)
}

pub(super) fn beta_log_ratio(a: f64, b: f64, x: f64) -> (f64, f64) {
    let residual = x.mul_add(b, -((1.0 - x) * a));
    let log_ratio = a * log1pmx(residual / a) + b * log1pmx(-residual / b);
    (residual, log_ratio)
}

pub(super) fn beta_reg_asymptotic(a: f64, b: f64, x: f64) -> Option<f64> {
    let (mean, complement, _, root_sum) = beta_shape_statistics(a, b);
    if root_sum < ASYMPTOTIC_MIN_SUM.sqrt() {
        return None;
    }

    if mean.min(complement) < 0.1 && a.min(b) < ASYMPTOTIC_MIN_SHAPE {
        return None;
    }

    let (residual, log_ratio) = beta_log_ratio(a, b, x);
    let scaled_deviance = -log_ratio;
    if scaled_deviance > ASYMPTOTIC_MAX_DEVIANCE {
        if scaled_deviance > -f64::from_bits(1).ln() {
            return Some(if residual < 0.0 { 0.0 } else { 1.0 });
        }
        return None;
    }

    let scale = a.max(b);
    let delta = (residual / scale) / (a / scale + b / scale);
    let root_variance = (mean * complement).sqrt();
    let eta = if residual == 0.0 {
        0.0
    } else {
        ((2.0 * scaled_deviance).sqrt() / root_sum).copysign(residual)
    };
    let c0 = if residual.abs() < 1e-4 * a.min(b) {
        let variance = mean * complement;
        (1.0 - 2.0 * mean) / (3.0 * root_variance)
            + (variance - 1.0) * (delta / variance) / (12.0 * root_variance)
    } else {
        1.0 / eta - a.sqrt() * b.sqrt() / residual
    };
    let normal_argument = -scaled_deviance.sqrt().copysign(residual);
    let leading = if normal_argument == 0.0 {
        0.5
    } else {
        let tail = 0.5 * gamma::gamma_ur(0.5, normal_argument * normal_argument);
        if normal_argument > 0.0 {
            tail
        } else {
            1.0 - tail
        }
    };
    let correction = (-scaled_deviance).exp() * c0 / (consts::SQRT_2PI * root_sum);
    let result = leading + correction;
    if (0.0..=1.0).contains(&result) {
        Some(result)
    } else {
        None
    }
}
