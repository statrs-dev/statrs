use super::super::super::*;

const MIN_SUBNORMAL: f64 = f64::from_bits(1);
const LOG_TWO: (f64, f64) = (core::f64::consts::LN_2, 2.3190468138462996e-17);

fn log_cdf_at_min_subnormal_multiple(a: f64, multiple: f64) -> (f64, f64) {
    let log_x = dd_add(accurate_ln(multiple), dd_mul((-1074.0, 0.0), LOG_TWO));
    if a > f64::MAX / -log_x.0 {
        return (f64::NEG_INFINITY, 0.0);
    }
    let ax = dd_mul(dd_mul((a, 0.0), (MIN_SUBNORMAL, 0.0)), (multiple, 0.0));
    let factor = dd_add(dd_add((1.0, 0.0), (a, 0.0)), (-ax.0, -ax.1));
    dd_add(dd_mul((a, 0.0), log_x), accurate_ln_dd(factor))
}

fn compare_logs(left: (f64, f64), right: (f64, f64)) -> core::cmp::Ordering {
    if right.0 == f64::NEG_INFINITY {
        return core::cmp::Ordering::Greater;
    }
    let difference = dd_add(left, (-right.0, -right.1));
    (difference.0 + difference.1).total_cmp(&0.0)
}

pub(super) fn lower_endpoint_result(a: f64, b: f64, probability: f64) -> Option<f64> {
    if b != 2.0 {
        return None;
    }
    let target = accurate_ln(probability);
    if compare_logs(target, log_cdf_at_min_subnormal_multiple(a, 0.5)).is_le() {
        return Some(0.0);
    }
    if compare_logs(target, log_cdf_at_min_subnormal_multiple(a, 1.5)).is_lt() {
        return Some(MIN_SUBNORMAL);
    }
    None
}
