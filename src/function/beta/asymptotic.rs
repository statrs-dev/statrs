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

fn beta_asymptotic_log_ratio(a: f64, b: f64, x: f64) -> (f64, (f64, f64)) {
    let complement = two_sum(1.0, -x);
    let left = dd_mul((x, 0.0), (b, 0.0));
    let right = dd_mul(complement, (a, 0.0));
    let residual_parts = dd_add(left, (-right.0, -right.1));
    let residual = residual_parts.0 + residual_parts.1;
    let left_ratio = dd_div((residual_parts.0, residual_parts.1), (a, 0.0));
    let right_ratio = dd_div((-residual_parts.0, -residual_parts.1), (b, 0.0));
    let left_log = dd_add(
        accurate_ln_one_plus_dd(left_ratio),
        (-left_ratio.0, -left_ratio.1),
    );
    let right_log = dd_add(
        accurate_ln_one_plus_dd(right_ratio),
        (-right_ratio.0, -right_ratio.1),
    );
    let log_ratio = dd_add(dd_mul((a, 0.0), left_log), dd_mul((b, 0.0), right_log));
    (residual, log_ratio)
}

fn beta_reg_symmetric_central(a: f64, b: f64, x: f64) -> Option<f64> {
    if a != b || a < 100.0 {
        return None;
    }

    let delta = x - 0.5;
    let delta_squared = delta * delta;
    if 4.0 * a * delta_squared > 0.5 {
        return None;
    }

    // DLMF 8.17.1 and 5.5.5 give the symmetric-center integral; DLMF
    // 5.11.13 supplies the asymptotic gamma ratio in its normalization.
    let inverse_a = 1.0 / a;
    let mut gamma_ratio: f64 = 869.0 / 4_194_304.0;
    for coefficient in [
        -399.0 / 262_144.0,
        -21.0 / 32_768.0,
        5.0 / 1_024.0,
        1.0 / 128.0,
        -1.0 / 8.0,
        1.0,
    ] {
        gamma_ratio = gamma_ratio.mul_add(inverse_a, coefficient);
    }
    let central_density = 2.0 * (a / core::f64::consts::PI).sqrt() * gamma_ratio;

    let mut term = delta;
    let mut integral = term;
    for index in 1..=32 {
        let n = f64::from(index);
        term *= -4.0 * (a - n) * delta_squared * (2.0 * n - 1.0) / (n * (2.0 * n + 1.0));
        let previous = integral;
        integral += term;
        if integral == previous {
            break;
        }
    }
    Some(central_density.mul_add(integral, 0.5))
}

pub(super) fn beta_reg_asymptotic(a: f64, b: f64, x: f64) -> Option<f64> {
    if let Some(result) = beta_reg_symmetric_central(a, b, x) {
        return Some(result);
    }

    let (mean, complement, _, root_sum) = beta_shape_statistics(a, b);
    if a < ASYMPTOTIC_MIN_SUM && b < ASYMPTOTIC_MIN_SUM - a {
        return None;
    }

    if mean.min(complement) < 0.01
        || (mean.min(complement) < 0.1 && a.min(b) < ASYMPTOTIC_MIN_SHAPE)
    {
        return None;
    }

    let (residual, log_ratio) = beta_asymptotic_log_ratio(a, b, x);
    let scaled_deviance_parts = (-log_ratio.0, -log_ratio.1);
    let scaled_deviance = scaled_deviance_parts.0 + scaled_deviance_parts.1;
    if scaled_deviance > ASYMPTOTIC_MAX_DEVIANCE {
        if scaled_deviance > -f64::from_bits(1).ln() {
            return Some(if residual < 0.0 { 0.0 } else { 1.0 });
        }
        return None;
    }

    let use_centered = mean.min(complement) >= 0.01 && residual.abs() < 0.05 * a.min(b);
    let (c0, c1) = if use_centered {
        let (series_mean, delta_parts) = temme_delta(a, b, x);
        temme_coefficients(series_mean, delta_parts.0 + delta_parts.1)
    } else {
        let eta = ((2.0 * scaled_deviance).sqrt() / root_sum).copysign(residual);
        let c0 = 1.0 / eta - a.sqrt() * b.sqrt() / residual;
        (c0, 0.0)
    };
    let normal_argument = if scaled_deviance == 0.0 {
        (0.0, 0.0)
    } else {
        let normal_root = scaled_deviance_parts.0.sqrt();
        let normal_root_error = (scaled_deviance_parts.1
            + (-normal_root).mul_add(normal_root, scaled_deviance_parts.0))
            / (2.0 * normal_root);
        if residual < 0.0 {
            (normal_root, normal_root_error)
        } else {
            (-normal_root, -normal_root_error)
        }
    };
    let absolute_normal_argument = if normal_argument.0 >= 0.0 {
        normal_argument
    } else {
        (-normal_argument.0, -normal_argument.1)
    };
    let tail = normal_tail(absolute_normal_argument);
    let coefficient = (-c1 / (root_sum * root_sum)).mul_add(1.0, c0);
    let correction = dd_exp(log_ratio) * coefficient / (consts::SQRT_2PI * root_sum);
    let result_parts = if normal_argument.0 >= 0.0 {
        dd_add((tail, 0.0), (correction, 0.0))
    } else {
        let complement = dd_add((tail, 0.0), (-correction, 0.0));
        dd_add((1.0, 0.0), (-complement.0, -complement.1))
    };
    let result = result_parts.0 + result_parts.1;
    if (0.0..=1.0).contains(&result) {
        Some(result)
    } else {
        None
    }
}
