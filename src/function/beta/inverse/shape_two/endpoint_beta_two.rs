use super::super::super::*;

const CERTIFIED_DD_RADIUS: f64 = f64::from_bits((1023_u64 - 79) << 52);
const CERTIFIED_MEDIAN_RADIUS: f64 = f64::from_bits((1023_u64 - 95) << 52);
const ENDPOINT_LIMIT_BITS: u64 = 0x0020_0000_0000_0000;
const MINIMUM_SHAPE_BITS: u64 = (1023_u64 + 484) << 52;
const MEDIAN_SCALE: f64 = f64::from_bits(0x3ffada825f9762b2);
const MEDIAN_SCALE_LOW: f64 = f64::from_bits(0x3c9f4a493534d79d);

fn normal_power_of_two(exponent: i32) -> f64 {
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

fn binary_exponent(value: f64) -> i32 {
    ((value.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

fn normalized_probability(value: f64) -> (f64, i32) {
    let (scaled, adjustment) = if value < f64::MIN_POSITIVE {
        (value * 18_014_398_509_481_984.0, -54)
    } else {
        (value, 0)
    };
    let bits = scaled.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023 + adjustment;
    let mantissa = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | (1023_u64 << 52));
    (mantissa, exponent)
}

fn scale_probability(value: f64, adjustment: i32) -> Option<f64> {
    let (mantissa, exponent) = normalized_probability(value);
    let exponent = exponent + adjustment;
    if exponent > 1023 {
        None
    } else if exponent < -1022 {
        Some(0.0)
    } else {
        Some(mantissa * normal_power_of_two(exponent))
    }
}

pub(super) fn initial_candidate(b: f64, probability: f64) -> f64 {
    if probability == 0.5 {
        return MEDIAN_SCALE / b;
    }
    let log_tail = (-probability).ln_1p();
    let tail_scale = -log_tail;
    let mut scaled = (2.0 * probability).sqrt().max(tail_scale + tail_scale.ln());
    for _ in 0..3 {
        if scaled < 0.0001 {
            break;
        }
        let log_survival = -scaled + scaled.ln_1p();
        let survival = log_survival.exp();
        let cdf = 1.0 - survival;
        let derivative = scaled * survival / (1.0 + scaled);
        let step = (cdf - probability) / derivative;
        let derivative_ratio = 1.0 / scaled - 1.0;
        let next = scaled - step / (1.0 - 0.5 * step * derivative_ratio);
        if next == scaled || !next.is_finite() || next <= 0.0 {
            break;
        }
        scaled = next;
    }
    scaled / b
}

fn limiting_correction(scaled_x: (f64, f64)) -> (f64, f64) {
    let mut term = (0.5, 0.0);
    let mut tail = (0.0, 0.0);
    for index in 1..=64 {
        let index = f64::from(index);
        let coefficient = dd_div_f64((-(index + 1.0), 0.0), index * (index + 2.0));
        term = dd_mul(dd_mul(term, scaled_x), coefficient);
        tail = dd_add(tail, term);
        if index >= 8.0 && term.0.abs() <= f64::EPSILON * f64::EPSILON * tail.0.abs() {
            break;
        }
    }
    tail
}

fn scaled_midpoint(b: f64, lower_bits: u64) -> (f64, f64) {
    let lower = f64::from_bits(lower_bits);
    let upper = f64::from_bits(lower_bits + 1);
    let scaled_lower = dd_mul((b, 0.0), (lower, 0.0));
    let scaled_step = dd_mul((b, 0.0), (upper - lower, 0.0));
    dd_add(scaled_lower, (0.5 * scaled_step.0, 0.5 * scaled_step.1))
}

fn midpoint_difference(b: f64, probability: f64, lower_bits: u64) -> f64 {
    let scaled_x = scaled_midpoint(b, lower_bits);
    let exponent = binary_exponent(scaled_x.0);
    let scale = normal_power_of_two(-exponent);
    let normalized_x = (scaled_x.0 * scale, scaled_x.1 * scale);
    let normalized_square = dd_mul(normalized_x, normalized_x);
    let Some(normalized_probability) = scale_probability(probability, -2 * exponent) else {
        return f64::INFINITY;
    };
    if normalized_probability == 0.0 {
        return f64::NEG_INFINITY;
    }
    let high_square = dd_mul((normalized_x.0, 0.0), (normalized_x.0, 0.0));
    let mut numerator = dd_add(
        (normalized_probability, 0.0),
        (-0.5 * high_square.0, -0.5 * high_square.1),
    );
    let cross = dd_mul((normalized_x.0, 0.0), (normalized_x.1, 0.0));
    numerator = dd_add(numerator, (-cross.0, -cross.1));
    let low_square = dd_mul((normalized_x.1, 0.0), (normalized_x.1, 0.0));
    numerator = dd_add(numerator, (-0.5 * low_square.0, -0.5 * low_square.1));
    let target = dd_div(numerator, normalized_square);
    let limiting = limiting_correction(scaled_x);
    let difference = dd_add(target, (-limiting.0, -limiting.1));
    difference.0 + difference.1
}

fn certified_median_midpoint_order(b: f64, lower_bits: u64) -> Option<core::cmp::Ordering> {
    let scaled_x = scaled_midpoint(b, lower_bits);
    let difference = dd_add(scaled_x, (-MEDIAN_SCALE, -MEDIAN_SCALE_LOW));
    let difference = difference.0 + difference.1;
    if difference > CERTIFIED_MEDIAN_RADIUS {
        Some(core::cmp::Ordering::Less)
    } else if difference < -CERTIFIED_MEDIAN_RADIUS {
        Some(core::cmp::Ordering::Greater)
    } else {
        None
    }
}

pub(super) fn certified_midpoint_order(
    b: f64,
    probability: f64,
    lower_bits: u64,
) -> Option<core::cmp::Ordering> {
    if !b.is_finite()
        || b.is_sign_negative()
        || b.to_bits() < MINIMUM_SHAPE_BITS
        || !(0.0..1.0).contains(&probability)
        || lower_bits >= ENDPOINT_LIMIT_BITS
    {
        return None;
    }
    if probability == 0.5 {
        return certified_median_midpoint_order(b, lower_bits);
    }
    let difference = midpoint_difference(b, probability, lower_bits);
    (difference.abs() > CERTIFIED_DD_RADIUS).then(|| difference.total_cmp(&0.0))
}
