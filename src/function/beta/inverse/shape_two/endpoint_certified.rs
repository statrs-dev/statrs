use super::endpoint_fixed::Fixed;
use core::cmp::Ordering;

const ENDPOINT_LIMIT_BITS: u64 = 0x0020_0000_0000_0000;
const MINIMUM_SHAPE_BITS: u64 = (1023_u64 + 484) << 52;
const FRACTION_LIMBS: usize = 32;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Certificate {
    Ordered(Ordering),
    Overlap,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CertifierError {
    InvalidInput,
    ArithmeticInvariant,
}

enum Stage {
    Ordered(Ordering),
    Overlap,
    Refine,
}

#[derive(Clone, Copy)]
struct Interval {
    lower: Fixed,
    upper: Fixed,
    cutoff: usize,
}

impl Interval {
    fn exact(value: Fixed, cutoff: usize) -> Option<Self> {
        Some(Self {
            lower: value.quantize_floor(cutoff)?,
            upper: value.quantize_ceil(cutoff)?,
            cutoff,
        })
    }

    fn zero(cutoff: usize) -> Self {
        Self {
            lower: Fixed::zero(),
            upper: Fixed::zero(),
            cutoff,
        }
    }

    fn same_precision(self, other: Self) -> Option<usize> {
        (self.cutoff == other.cutoff).then_some(self.cutoff)
    }

    fn checked_add(self, other: Self) -> Option<Self> {
        let cutoff = self.same_precision(other)?;
        Some(Self {
            lower: self.lower.checked_add(other.lower, cutoff)?,
            upper: self.upper.checked_add(other.upper, cutoff)?,
            cutoff,
        })
    }

    fn checked_sub(self, other: Self) -> Option<Self> {
        let cutoff = self.same_precision(other)?;
        Some(Self {
            lower: self.lower.checked_sub(other.upper, cutoff)?,
            upper: self.upper.checked_sub(other.lower, cutoff)?,
            cutoff,
        })
    }

    fn checked_mul(self, other: Self) -> Option<Self> {
        let cutoff = self.same_precision(other)?;
        Some(Self {
            lower: self.lower.mul_floor(other.lower, cutoff)?,
            upper: self.upper.mul_ceil(other.upper, cutoff)?,
            cutoff,
        })
    }

    fn checked_mul_small(self, factor: u64) -> Option<Self> {
        Some(Self {
            lower: self.lower.checked_mul_small(factor, self.cutoff)?,
            upper: self.upper.checked_mul_small(factor, self.cutoff)?,
            cutoff: self.cutoff,
        })
    }

    fn checked_div_small(self, divisor: u64) -> Option<Self> {
        Some(Self {
            lower: self.lower.div_small_floor(divisor, self.cutoff),
            upper: self.upper.div_small_ceil(divisor, self.cutoff)?,
            cutoff: self.cutoff,
        })
    }
}

fn next_term(term: Interval, factor: Interval, index: u64) -> Option<Interval> {
    term.checked_mul(factor)?
        .checked_mul_small(index + 1)?
        .checked_div_small(index.checked_mul(index + 2)?)
}

fn remainder_interval(
    positive: Interval,
    negative: Interval,
    term: Interval,
    index: u64,
) -> Option<Interval> {
    let cutoff = positive.same_precision(negative)?;
    (cutoff == term.cutoff).then_some(())?;
    let lower = positive.lower.checked_sub(negative.upper, cutoff)?;
    let upper = positive.upper.checked_sub(negative.lower, cutoff)?;
    if index & 1 == 0 {
        Some(Interval {
            lower,
            upper: upper.checked_add(term.upper, cutoff)?,
            cutoff,
        })
    } else {
        Some(Interval {
            lower: lower.checked_sub(term.upper, cutoff)?,
            upper,
            cutoff,
        })
    }
}

fn series_interval(y: Interval, x: Interval) -> Option<Interval> {
    let cutoff = y.same_precision(x)?;
    let half = Interval::exact(Fixed::half()?, cutoff)?;
    let quantum = Fixed::quantum(cutoff)?;
    let mut positive = half;
    let mut negative = Interval::zero(cutoff);
    let mut term = half;
    for index in 1_u64..=513 {
        let factor = y.checked_sub(x.checked_mul_small(index)?)?;
        term = next_term(term, factor, index)?;
        if index >= 8 && term.upper <= quantum {
            return remainder_interval(positive, negative, term, index);
        }
        if index & 1 == 0 {
            positive = positive.checked_add(term)?;
        } else {
            negative = negative.checked_add(term)?;
        }
    }
    None
}

fn order_at_precision(b: f64, probability: f64, lower_bits: u64, active_limbs: usize) -> Stage {
    let Some(cutoff) = FRACTION_LIMBS.checked_sub(active_limbs) else {
        return Stage::Refine;
    };
    let Some(x) = Fixed::midpoint(lower_bits).and_then(|value| Interval::exact(value, cutoff))
    else {
        return Stage::Refine;
    };
    let Some(y) =
        Fixed::scaled_midpoint(b, lower_bits).and_then(|value| Interval::exact(value, cutoff))
    else {
        return Stage::Refine;
    };
    let Some(eight) = Fixed::integer(8) else {
        return Stage::Refine;
    };
    if y.upper >= eight {
        return Stage::Refine;
    }
    let Some(series) = series_interval(y, x) else {
        return Stage::Refine;
    };
    let Some(cdf) = y
        .checked_add(x)
        .and_then(|sum| y.checked_mul(sum))
        .and_then(|prefactor| prefactor.checked_mul(series))
    else {
        return Stage::Refine;
    };
    let Some(probability) =
        Fixed::from_f64(probability).and_then(|value| Interval::exact(value, cutoff))
    else {
        return Stage::Refine;
    };
    if probability.upper < cdf.lower {
        Stage::Ordered(Ordering::Less)
    } else if probability.lower > cdf.upper {
        Stage::Ordered(Ordering::Greater)
    } else {
        Stage::Overlap
    }
}

fn exponent(value: f64) -> Option<i32> {
    let bits = value.to_bits();
    let encoded = ((bits >> 52) & 0x7ff) as i32;
    if encoded != 0 {
        return (encoded != 0x7ff).then_some(encoded - 1023);
    }
    let mantissa = bits & 0x000f_ffff_ffff_ffff;
    if mantissa == 0 {
        return None;
    }
    let leading = i32::try_from(63 - mantissa.leading_zeros()).ok()?;
    Some(leading - 1074)
}

fn initial_active_limbs(probability: f64) -> Option<usize> {
    let bits = exponent(probability)?.checked_neg()?.checked_add(128)?;
    let rounded = bits.checked_add(63)?.checked_div(64)?;
    usize::try_from(rounded)
        .ok()
        .map(|limbs| limbs.clamp(4, FRACTION_LIMBS))
}

pub(super) fn midpoint_certificate(
    b: f64,
    probability: f64,
    lower_bits: u64,
) -> Result<Certificate, CertifierError> {
    if !b.is_finite()
        || b.is_sign_negative()
        || b.to_bits() < MINIMUM_SHAPE_BITS
        || !(probability > 0.0 && probability < 1.0)
        || lower_bits >= ENDPOINT_LIMIT_BITS
    {
        return Err(CertifierError::InvalidInput);
    }
    let initial = initial_active_limbs(probability).ok_or(CertifierError::InvalidInput)?;
    let stages = [initial, 8, 16, 24];
    let mut previous = 0;
    for active in stages {
        let active = active.max(initial).min(FRACTION_LIMBS);
        if active == previous {
            continue;
        }
        if let Stage::Ordered(order) = order_at_precision(b, probability, lower_bits, active) {
            return Ok(Certificate::Ordered(order));
        }
        previous = active;
    }
    match order_at_precision(b, probability, lower_bits, FRACTION_LIMBS) {
        Stage::Ordered(order) => Ok(Certificate::Ordered(order)),
        Stage::Overlap => Ok(Certificate::Overlap),
        Stage::Refine => Err(CertifierError::ArithmeticInvariant),
    }
}

#[cfg(test)]
mod tests;
