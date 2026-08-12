use super::super::super::*;

const MIN_SUBNORMAL: f64 = f64::from_bits(1);
const MIN_NORMAL_BITS: u64 = f64::MIN_POSITIVE.to_bits();
const LOG_TWO: (f64, f64) = (core::f64::consts::LN_2, 2.3190468138462996e-17);

fn log_cdf_at_min_subnormal_multiple(a: f64, multiple: (f64, f64)) -> (f64, f64) {
    let log_x = dd_add(accurate_ln_dd(multiple), dd_mul((-1074.0, 0.0), LOG_TWO));
    if a > f64::MAX / -log_x.0 {
        return (f64::NEG_INFINITY, 0.0);
    }
    let ax = dd_mul(dd_mul((a, 0.0), (MIN_SUBNORMAL, 0.0)), multiple);
    let factor = dd_add(dd_add((1.0, 0.0), (a, 0.0)), (-ax.0, -ax.1));
    dd_add(dd_mul((a, 0.0), log_x), accurate_ln_dd(factor))
}

fn compare_logs(left: (f64, f64), right: (f64, f64)) -> core::cmp::Ordering {
    if right.0 == f64::NEG_INFINITY {
        return core::cmp::Ordering::Greater;
    }
    let difference = dd_add(left, (-right.0, -right.1));
    let difference = difference.0 + difference.1;
    if difference < 0.0 {
        core::cmp::Ordering::Less
    } else if difference > 0.0 {
        core::cmp::Ordering::Greater
    } else {
        core::cmp::Ordering::Equal
    }
}

fn midpoint_log_cdf(a: f64, lower_bits: u64) -> (f64, f64) {
    let multiple = dd_add((lower_bits as f64, 0.0), (0.5, 0.0));
    log_cdf_at_min_subnormal_multiple(a, multiple)
}

pub(super) fn lower_endpoint_result(a: f64, b: f64, probability: f64, initial: f64) -> Option<f64> {
    if b != 2.0 || initial > f64::MIN_POSITIVE {
        return None;
    }
    let target = accurate_ln(probability);
    let last_subnormal = MIN_NORMAL_BITS - 1;
    match compare_logs(target, midpoint_log_cdf(a, last_subnormal)) {
        core::cmp::Ordering::Greater => return None,
        core::cmp::Ordering::Equal => return Some(f64::MIN_POSITIVE),
        core::cmp::Ordering::Less => {}
    }
    let mut lower = 0_u64;
    let mut upper = last_subnormal;
    while lower < upper {
        let midpoint = lower + (upper - lower) / 2;
        match compare_logs(target, midpoint_log_cdf(a, midpoint)) {
            core::cmp::Ordering::Less => upper = midpoint,
            core::cmp::Ordering::Greater => lower = midpoint + 1,
            core::cmp::Ordering::Equal => {
                let even = if midpoint & 1 == 0 {
                    midpoint
                } else {
                    midpoint + 1
                };
                return Some(f64::from_bits(even));
            }
        }
    }
    Some(f64::from_bits(lower))
}
