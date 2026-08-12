use super::super::super::*;
use super::endpoint_beta_two;
use super::endpoint_certified;
use super::endpoint_certified::Certificate;

const MIN_SUBNORMAL: f64 = f64::from_bits(1);
const MIN_NORMAL_BITS: u64 = f64::MIN_POSITIVE.to_bits();
const FIRST_NORMAL_BIN_END_BITS: u64 = f64::MIN_POSITIVE.to_bits() + (1_u64 << 52) - 1;
const BETA_TWO_ENDPOINT_SHAPE: f64 = f64::from_bits(1507_u64 << 52);
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

pub(super) fn overlap_result(lower_bits: u64) -> f64 {
    let result_bits = if lower_bits & 1 == 0 {
        lower_bits
    } else {
        lower_bits + 1
    };
    f64::from_bits(result_bits)
}

fn midpoint_order(
    a: f64,
    b: f64,
    probability: f64,
    target: (f64, f64),
    lower_bits: u64,
) -> Option<Result<core::cmp::Ordering, f64>> {
    if a == 2.0 {
        if let Some(order) = endpoint_beta_two::certified_midpoint_order(b, probability, lower_bits)
        {
            return Some(Ok(order));
        }
        return Some(
            match endpoint_certified::midpoint_certificate(b, probability, lower_bits).ok()? {
                Certificate::Ordered(order) => Ok(order),
                Certificate::Overlap => Err(overlap_result(lower_bits)),
            },
        );
    }
    let order = compare_logs(target, midpoint_log_cdf(a, lower_bits));
    if order.is_eq() {
        let even = if lower_bits & 1 == 0 {
            lower_bits
        } else {
            lower_bits + 1
        };
        Some(Err(f64::from_bits(even)))
    } else {
        Some(Ok(order))
    }
}

pub(super) fn lower_endpoint_result(a: f64, b: f64, probability: f64, initial: f64) -> Option<f64> {
    let last_candidate = if a == 2.0 {
        FIRST_NORMAL_BIN_END_BITS
    } else {
        MIN_NORMAL_BITS - 1
    };
    if !b.is_finite()
        || a == 2.0 && b < BETA_TWO_ENDPOINT_SHAPE
        || a != 2.0 && (b != 2.0 || initial >= f64::MIN_POSITIVE)
    {
        return None;
    }
    let target = accurate_ln(probability);
    let candidate = if a == 2.0 {
        endpoint_beta_two::initial_candidate(b, probability)
    } else {
        initial
    }
    .to_bits()
    .min(last_candidate);
    let (mut lower, mut upper) = match midpoint_order(a, b, probability, target, candidate)? {
        Err(result) => return Some(result),
        Ok(core::cmp::Ordering::Less) => {
            let mut upper = candidate;
            let mut step = 1_u64;
            loop {
                if upper == 0 {
                    if a == 2.0 {
                        break (0, 0);
                    }
                    return Some(0.0);
                }
                let probe = upper.saturating_sub(step);
                match midpoint_order(a, b, probability, target, probe)? {
                    Err(result) => return Some(result),
                    Ok(core::cmp::Ordering::Less) => {
                        upper = probe;
                        step = step.saturating_mul(2);
                    }
                    Ok(core::cmp::Ordering::Greater) => break (probe + 1, upper),
                    Ok(core::cmp::Ordering::Equal) => unreachable!(),
                }
            }
        }
        Ok(core::cmp::Ordering::Greater) => {
            let mut lower = candidate + 1;
            let mut probe = candidate;
            let mut step = 1_u64;
            loop {
                if probe == last_candidate {
                    if a == 2.0 {
                        break (last_candidate + 1, last_candidate + 1);
                    }
                    return None;
                }
                probe = probe.saturating_add(step).min(last_candidate);
                match midpoint_order(a, b, probability, target, probe)? {
                    Err(result) => return Some(result),
                    Ok(core::cmp::Ordering::Less) => break (lower, probe),
                    Ok(core::cmp::Ordering::Greater) => {
                        lower = probe + 1;
                        step = step.saturating_mul(2);
                    }
                    Ok(core::cmp::Ordering::Equal) => unreachable!(),
                }
            }
        }
        Ok(core::cmp::Ordering::Equal) => unreachable!(),
    };
    while lower < upper {
        let midpoint = lower + (upper - lower) / 2;
        match midpoint_order(a, b, probability, target, midpoint)? {
            Err(result) => return Some(result),
            Ok(core::cmp::Ordering::Less) => upper = midpoint,
            Ok(core::cmp::Ordering::Greater) => lower = midpoint + 1,
            Ok(core::cmp::Ordering::Equal) => unreachable!(),
        }
    }
    (lower <= last_candidate).then(|| f64::from_bits(lower))
}

#[cfg(test)]
mod tests;
