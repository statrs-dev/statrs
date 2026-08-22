use core::cmp::Ordering;

const LIMBS: usize = 34;
const WIDE_LIMBS: usize = 68;
const FRACTION_BITS: i32 = 2048;
const FRACTION_LIMBS: usize = 32;

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) struct Fixed([u64; LIMBS]);

impl Ord for Fixed {
    fn cmp(&self, other: &Self) -> Ordering {
        for index in (0..LIMBS).rev() {
            match self.0[index].cmp(&other.0[index]) {
                Ordering::Equal => {}
                order => return order,
            }
        }
        Ordering::Equal
    }
}

impl PartialOrd for Fixed {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Fixed {
    pub(super) const fn zero() -> Self {
        Self([0; LIMBS])
    }

    fn from_shifted_u128(value: u128, shift: usize) -> Option<Self> {
        let mut result = Self::zero();
        let word = shift / 64;
        let offset = shift % 64;
        for (part_index, part) in [value as u64, (value >> 64) as u64].into_iter().enumerate() {
            if part == 0 {
                continue;
            }
            let index = word + part_index;
            if index >= LIMBS {
                return None;
            }
            result.0[index] |= part << offset;
            if offset != 0 {
                if index + 1 >= LIMBS {
                    return None;
                }
                result.0[index + 1] |= part >> (64 - offset);
            }
        }
        Some(result)
    }

    pub(super) fn integer(value: u64) -> Option<Self> {
        Self::from_shifted_u128(value as u128, FRACTION_BITS as usize)
    }

    pub(super) fn half() -> Option<Self> {
        Self::from_shifted_u128(1, FRACTION_BITS as usize - 1)
    }

    pub(super) fn quantum(cutoff: usize) -> Option<Self> {
        (cutoff <= FRACTION_LIMBS).then(|| {
            let mut result = Self::zero();
            result.0[cutoff] = 1;
            result
        })
    }

    pub(super) fn quantize_floor(mut self, cutoff: usize) -> Option<Self> {
        (cutoff <= FRACTION_LIMBS).then(|| {
            self.0[..cutoff].fill(0);
            self
        })
    }

    pub(super) fn quantize_ceil(mut self, cutoff: usize) -> Option<Self> {
        if cutoff > FRACTION_LIMBS {
            return None;
        }
        let discarded = self.0[..cutoff].iter().any(|&limb| limb != 0);
        self.0[..cutoff].fill(0);
        if discarded {
            self = self.checked_add(Self::quantum(cutoff)?, cutoff)?;
        }
        Some(self)
    }

    pub(super) fn from_f64(value: f64) -> Option<Self> {
        let bits = value.to_bits();
        let encoded_exponent = ((bits >> 52) & 0x7ff) as i32;
        let (mantissa, power) = if encoded_exponent == 0 {
            (bits & 0x000f_ffff_ffff_ffff, -1074)
        } else {
            (
                (bits & 0x000f_ffff_ffff_ffff) | (1_u64 << 52),
                encoded_exponent - 1023 - 52,
            )
        };
        let shift = FRACTION_BITS.checked_add(power)?;
        Self::from_shifted_u128(mantissa as u128, usize::try_from(shift).ok()?)
    }

    pub(super) fn midpoint(lower_bits: u64) -> Option<Self> {
        let numerator = (2_u128).checked_mul(lower_bits as u128)?.checked_add(1)?;
        Self::from_shifted_u128(numerator, (FRACTION_BITS - 1075) as usize)
    }

    pub(super) fn scaled_midpoint(b: f64, lower_bits: u64) -> Option<Self> {
        let bits = b.to_bits();
        let encoded_exponent = ((bits >> 52) & 0x7ff) as i32;
        if encoded_exponent == 0 || encoded_exponent == 0x7ff {
            return None;
        }
        let mantissa = (bits & 0x000f_ffff_ffff_ffff) | (1_u64 << 52);
        let numerator = (2_u128)
            .checked_mul(lower_bits as u128)?
            .checked_add(1)?
            .checked_mul(mantissa as u128)?;
        let exponent = encoded_exponent - 1023;
        let shift = FRACTION_BITS.checked_add(exponent)?.checked_sub(1127)?;
        Self::from_shifted_u128(numerator, usize::try_from(shift).ok()?)
    }

    pub(super) fn checked_add(self, other: Self, cutoff: usize) -> Option<Self> {
        let mut result = Self::zero();
        let mut carry = 0_u128;
        for index in cutoff..LIMBS {
            let sum = self.0[index] as u128 + other.0[index] as u128 + carry;
            result.0[index] = sum as u64;
            carry = sum >> 64;
        }
        (carry == 0).then_some(result)
    }

    pub(super) fn checked_sub(self, other: Self, cutoff: usize) -> Option<Self> {
        if self < other {
            return None;
        }
        let mut result = Self::zero();
        let mut borrow = 0_u128;
        for index in cutoff..LIMBS {
            let subtrahend = other.0[index] as u128 + borrow;
            let value = self.0[index] as u128;
            result.0[index] = value.wrapping_sub(subtrahend) as u64;
            borrow = u128::from(value < subtrahend);
        }
        (borrow == 0).then_some(result)
    }

    pub(super) fn checked_mul_small(self, factor: u64, cutoff: usize) -> Option<Self> {
        let mut result = Self::zero();
        let mut carry = 0_u128;
        for index in cutoff..LIMBS {
            let product = self.0[index] as u128 * factor as u128 + carry;
            result.0[index] = product as u64;
            carry = product >> 64;
        }
        (carry == 0).then_some(result)
    }

    fn div_small(self, divisor: u64, cutoff: usize) -> (Self, u64) {
        let mut result = Self::zero();
        let mut remainder = 0_u128;
        for index in (cutoff..LIMBS).rev() {
            let numerator = (remainder << 64) | self.0[index] as u128;
            result.0[index] = (numerator / divisor as u128) as u64;
            remainder = numerator % divisor as u128;
        }
        (result, remainder as u64)
    }

    pub(super) fn div_small_floor(self, divisor: u64, cutoff: usize) -> Self {
        self.div_small(divisor, cutoff).0
    }

    pub(super) fn div_small_ceil(self, divisor: u64, cutoff: usize) -> Option<Self> {
        let (mut result, remainder) = self.div_small(divisor, cutoff);
        if remainder != 0 {
            result = result.checked_add(Self::quantum(cutoff)?, cutoff)?;
        }
        Some(result)
    }

    fn product(self, other: Self, cutoff: usize) -> Option<[u64; WIDE_LIMBS]> {
        let mut wide = [0_u64; WIDE_LIMBS];
        for left in cutoff..LIMBS {
            if self.0[left] == 0 {
                continue;
            }
            let mut carry = 0_u128;
            for right in cutoff..LIMBS {
                let index = left + right;
                let product =
                    self.0[left] as u128 * other.0[right] as u128 + wide[index] as u128 + carry;
                wide[index] = product as u64;
                carry = product >> 64;
            }
            let mut index = left + LIMBS;
            while carry != 0 {
                if index == WIDE_LIMBS {
                    return None;
                }
                let sum = wide[index] as u128 + carry;
                wide[index] = sum as u64;
                carry = sum >> 64;
                index += 1;
            }
        }
        Some(wide)
    }

    fn scaled_product(self, other: Self, cutoff: usize) -> Option<(Self, bool)> {
        let wide = self.product(other, cutoff)?;
        if wide[FRACTION_LIMBS + LIMBS..].iter().any(|&limb| limb != 0) {
            return None;
        }
        let mut result = Self::zero();
        result
            .0
            .copy_from_slice(&wide[FRACTION_LIMBS..FRACTION_LIMBS + LIMBS]);
        result.0[..cutoff].fill(0);
        let discarded = wide[..FRACTION_LIMBS + cutoff]
            .iter()
            .any(|&limb| limb != 0);
        Some((result, discarded))
    }

    pub(super) fn mul_floor(self, other: Self, cutoff: usize) -> Option<Self> {
        Some(self.scaled_product(other, cutoff)?.0)
    }

    pub(super) fn mul_ceil(self, other: Self, cutoff: usize) -> Option<Self> {
        let (mut result, discarded) = self.scaled_product(other, cutoff)?;
        if discarded {
            result = result.checked_add(Self::quantum(cutoff)?, cutoff)?;
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests;
