//! Provides the [gamma](https://en.wikipedia.org/wiki/Gamma_function) and
//! related functions

use crate::consts;
use crate::function::evaluate;
use crate::prec::{dekker_product_err, two_diff};
use core::f64;
use core::f64::consts as f64_consts;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Represents the errors that can occur when computing any of the incomplete
/// gamma functions.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum GammaFuncError {
    /// `a` is infinite, zero or less than zero.
    AInvalid,

    /// `x` is infinite, zero or less than zero.
    XInvalid,
}

impl core::fmt::Display for GammaFuncError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            GammaFuncError::AInvalid => write!(f, "a is infinite, zero or less than zero"),
            GammaFuncError::XInvalid => write!(f, "x is infinite, zero or less than zero"),
        }
    }
}

impl core::error::Error for GammaFuncError {}

/// Computes `sin(PI * x)` for use in reflection formulas.
///
/// Evaluating `(PI * x).sin()` directly loses the fractional part of `x` to
/// rounding once `PI * x` is large, and near the zeros of the sine (integer
/// `x`) the representation error of `PI` alone costs `~1.2e-16 / dist_to_pole`
/// of relative accuracy. Reducing with the period first (`x - round(x)` is
/// exact) keeps the error relative to the fractional part instead.
#[inline]
fn sin_pi(x: f64) -> f64 {
    let m = x.round();
    let r = x - m; // exact, |r| <= 0.5
    // sin(PI * (m + r)) = (-1)^m * sin(PI * r); `m * 0.5` is exact, so its
    // fractional part is 0.5 exactly when `m` is odd. (For |m| >= 2^53 every
    // representable float is even.)
    let sin_r = (f64::consts::PI * r).sin();
    if (m * 0.5).fract() == 0.0 {
        sin_r
    } else {
        -sin_r
    }
}

/// Computes `tan(PI * x)` with the same period reduction as [`sin_pi`]
/// (`tan` has period `PI`, so no sign correction is needed).
#[inline]
fn tan_pi(x: f64) -> f64 {
    let r = x - x.round(); // exact, |r| <= 0.5
    (f64::consts::PI * r).tan()
}

/// Auxiliary variable when evaluating the `gamma_ln` function
const GAMMA_R: f64 = 10.900511;

// The Lanczos sum used by `ln_gamma` and `gamma` is Pugh's 10-term
// partial-fraction approximation with g = GAMMA_R ("An Analysis of the Lanczos
// Gamma Approximation", Glendon Ralph Pugh, 2004 p. 116):
//
//   S(z) = d_0 + sum_{k=1..10} d_k / (z + k - 1)
//
// with residues d_k = [2.48574089138753565546e-5, 1.05142378581721974210,
// -3.45687097222016235469, 4.51227709466894823700, -2.98285225323576655721,
// 1.05639711577126713077, -1.95428773191645869583e-1, 1.70970543404441224307e-2,
// -5.71926117404305781283e-4, 4.63399473359905636708e-6,
// -2.71994908488607703910e-9].
//
// The residues alternate in sign, so evaluating the sum directly cancels
// catastrophically (condition number ~3600 around z = 50, i.e. ~440 eps of
// relative error in the sum). The constants below are the mathematically
// identical single-fraction form S(z) = N(z) / D(z), derived from the d_k in
// exact rational arithmetic: D(z) = z (z+1) ... (z+9) (whose expanded
// coefficients are unsigned Stirling numbers of the first kind, exact in f64)
// and N = d_0 D + sum_k d_k D / (z + k - 1). Every coefficient of both
// polynomials is positive, so for z > 0 both Horner evaluations have condition
// number 1 and the sum is accurate to a few eps.

/// Numerator `N(z)` of the Lanczos sum in single-fraction form (ascending).
const LANCZOS_NUM: &[f64] = &[
    381540.6633973527,
    365505.352696257,
    157567.99949360118,
    40253.83538142639,
    6748.767525934567,
    775.8779405455635,
    61.94528891422096,
    3.391366244015308,
    0.12184807036444657,
    0.002594340508809067,
    2.4857408913875355e-5,
];

/// Denominator `D(z) = z (z+1) ... (z+9)` expanded (ascending; unsigned
/// Stirling numbers of the first kind, exact integers).
const LANCZOS_DENOM: &[f64] = &[
    0.0, 362880.0, 1026576.0, 1172700.0, 723680.0, 269325.0, 63273.0, 9450.0, 870.0, 45.0, 1.0,
];

/// `LANCZOS_NUM` reversed: `N(z) / z^10` as a polynomial in `w = 1/z`.
const LANCZOS_NUM_REV: &[f64] = &[
    2.4857408913875355e-5,
    0.002594340508809067,
    0.12184807036444657,
    3.391366244015308,
    61.94528891422096,
    775.8779405455635,
    6748.767525934567,
    40253.83538142639,
    157567.99949360118,
    365505.352696257,
    381540.6633973527,
];

/// `LANCZOS_DENOM` reversed: `D(z) / z^10` as a polynomial in `w = 1/z`.
const LANCZOS_DENOM_REV: &[f64] = &[
    1.0, 45.0, 870.0, 9450.0, 63273.0, 269325.0, 723680.0, 1172700.0, 1026576.0, 362880.0, 0.0,
];

/// Low half of Euler's number: `e == f64::consts::E + E_LO` to double-double
/// precision.
const E_LO: f64 = 1.4456468917292502e-16;

/// Computes `((p + GAMMA_R) / e)^exponent`, compensating the rounding of the base.
///
/// `powf` amplifies a relative error `eps` in its base by the exponent, so the
/// two roundings in `(p + GAMMA_R) / e` cost `~2 |p|` ulps - the dominant
/// error of the direct evaluation (~190 ulps in `gamma` by x = 122). The
/// rounding residuals of the addition (two-sum) and the division (Dekker
/// product, plus the low word of `e`) are recovered exactly and applied as the
/// first-order correction `(b (1 + delta))^p ~= b^p (1 + p delta)`.
/// `p_err` is the rounding residual of the exponent itself (the true exponent
/// is `p + p_err`), folded in as `b^(p_err) ~= 1 + p_err * ln(b)`; it must be
/// zero unless `exponent == p`.
///
/// `exponent` is normally `p`. The overflow-avoiding path in [`gamma`] passes
/// `p / 2` and squares the result, which needs the *base* to keep coming from
/// the full `p` - halving `p` in the base as well would compute an entirely
/// different quantity.
#[inline]
fn lanczos_power(p: f64, p_err: f64, exponent: f64) -> f64 {
    let (zgh, add_err) = two_diff(p, -GAMMA_R);
    let b = zgh / f64::consts::E;
    let pw = b * f64::consts::E;
    // exact residual of the division against the double-double e
    let div_err = (zgh - pw) - dekker_product_err(b, f64::consts::E, pw);
    let delta = (div_err + add_err + p_err - b * E_LO) / zgh;
    let mut corr = exponent * delta;
    if p_err != 0.0 {
        corr += p_err * b.ln();
    }
    let base = b.powf(exponent);
    // `(b (1 + delta))^p ~= b^p (1 + p delta)` only holds while `|p delta| << 1`.
    // Since `delta` is of order `eps`, that fails once `p` reaches ~1e13 - far
    // past where `b^p` overflows, so the uncorrected value is the right answer
    // there. Applying it anyway would let a negative `corr` flip the sign and
    // turn an overflowing `gamma` into `-inf`.
    if corr.abs() < 0.25 {
        base * (1.0 + corr)
    } else {
        base
    }
}

/// Evaluates the Lanczos sum `S(z)` for `z >= 0.5` via the well-conditioned
/// single-fraction form (see the derivation note above).
#[inline]
fn lanczos_sum(z: f64) -> f64 {
    if z < 1e29 {
        evaluate::polynomial(z, LANCZOS_NUM) / evaluate::polynomial(z, LANCZOS_DENOM)
    } else {
        // z^10 would overflow; divide both polynomials by z^10 and evaluate
        // in w = 1/z instead.
        let w = 1.0 / z;
        evaluate::polynomial(w, LANCZOS_NUM_REV) / evaluate::polynomial(w, LANCZOS_DENOM_REV)
    }
}

/// Computes the logarithm of the gamma function
/// with an accuracy of 16 floating point digits.
/// The implementation is derived from
/// "An Analysis of the Lanczos Gamma Approximation",
/// Glendon Ralph Pugh, 2004 p. 116
pub fn ln_gamma(x: f64) -> f64 {
    // ln Gamma(n) = ln((n - 1)!) via the exact factorial table (~0.4 ulp, and
    // exactly 0 at n = 1, 2, where the Lanczos formula's terms only cancel
    // approximately).
    if x.fract() == 0.0 && (1.0..=171.0).contains(&x) {
        return crate::function::factorial::ln_factorial(x as u64 - 1);
    }
    if x < 0.5 {
        let s = lanczos_sum(1.0 - x);

        consts::LN_PI
            - sin_pi(x).ln()
            - s.ln()
            - consts::LN_2_SQRT_E_OVER_PI
            - (0.5 - x) * ((0.5 - x + GAMMA_R) / f64_consts::E).ln()
    } else {
        let s = lanczos_sum(x);

        s.ln()
            + consts::LN_2_SQRT_E_OVER_PI
            + (x - 0.5) * ((x - 0.5 + GAMMA_R) / f64_consts::E).ln()
    }
}

/// Computes the gamma function. The implementation is derived from
/// "An Analysis of the Lanczos Gamma Approximation",
/// Glendon Ralph Pugh, 2004 p. 116.
///
/// Exact at the positive integers up to 171; elsewhere accurate to about
/// `4e-15` relative, which is the approximation floor of the (f64-rounded)
/// Pugh coefficient set itself - the evaluation is compensated to well below
/// that.
pub fn gamma(x: f64) -> f64 {
    // Gamma(n) = (n - 1)! exactly at the positive integers where the factorial
    // is representable; the Lanczos path is only good to ~1 ulp there.
    if x.fract() == 0.0 && (1.0..=171.0).contains(&x) {
        return crate::function::factorial::factorial(x as u64 - 1);
    }
    if x < 0.5 {
        let s = lanczos_sum(1.0 - x);
        // 0.5 - x rounds for x below the binade of 0.5; keep its residual
        let (pw, pw_err) = two_diff(0.5, x);

        f64_consts::PI
            / (sin_pi(x) * s * consts::TWO_SQRT_E_OVER_PI * lanczos_power(pw, pw_err, pw))
    } else {
        let s = lanczos_sum(x);

        // x - 0.5 is exact for 0.5 <= x < 2^52 (same or finer grid)
        let p = x - 0.5;
        if p > 168.0 {
            // `lanczos_power` alone overflows from about x = 169.7, while the
            // full product stays representable up to x ~ 171.61 (the true
            // overflow point of Gamma). Halve the exponent and square at the
            // end so the intermediate stays in range; this costs a couple of
            // ulp, well inside the approximation floor of the coefficient set.
            let half =
                s.sqrt() * consts::TWO_SQRT_E_OVER_PI.sqrt() * lanczos_power(p, 0.0, 0.5 * p);
            return half * half;
        }
        s * consts::TWO_SQRT_E_OVER_PI * lanczos_power(p, 0.0, p)
    }
}

/// Computes the upper incomplete gamma function
/// `Gamma(a,x) = int(exp(-t)t^(a-1), t=0..x) for a > 0, x > 0`
/// where `a` is the argument for the gamma function and
/// `x` is the lower intergral limit.
///
/// # Panics
///
/// if `a` or `x` are not in `(0, +inf)`
pub fn gamma_ui(a: f64, x: f64) -> f64 {
    checked_gamma_ui(a, x).unwrap()
}

/// Computes the upper incomplete gamma function
/// `Gamma(a,x) = int(exp(-t)t^(a-1), t=0..x) for a > 0, x > 0`
/// where `a` is the argument for the gamma function and
/// `x` is the lower intergral limit.
///
/// # Errors
///
/// if `a` or `x` are not in `(0, +inf)`
pub fn checked_gamma_ui(a: f64, x: f64) -> Result<f64, GammaFuncError> {
    checked_gamma_ur(a, x).map(|x| x * gamma(a))
}

/// Computes the lower incomplete gamma function
/// `gamma(a,x) = int(exp(-t)t^(a-1), t=0..x) for a > 0, x > 0`
/// where `a` is the argument for the gamma function and `x`
/// is the upper integral limit.
///
///
/// # Panics
///
/// if `a` or `x` are not in `(0, +inf)`
pub fn gamma_li(a: f64, x: f64) -> f64 {
    checked_gamma_li(a, x).unwrap()
}

/// Computes the lower incomplete gamma function
/// `gamma(a,x) = int(exp(-t)t^(a-1), t=0..x) for a > 0, x > 0`
/// where `a` is the argument for the gamma function and `x`
/// is the upper integral limit.
///
///
/// # Errors
///
/// if `a` or `x` are not in `(0, +inf)`
pub fn checked_gamma_li(a: f64, x: f64) -> Result<f64, GammaFuncError> {
    checked_gamma_lr(a, x).map(|x| x * gamma(a))
}

/// Computes the upper incomplete regularized gamma function
/// `Q(a,x) = 1 / Gamma(a) * int(exp(-t)t^(a-1), t=0..x) for a > 0, x > 0`
/// where `a` is the argument for the gamma function and
/// `x` is the lower integral limit.
///
/// # Remarks
///
/// Returns `f64::NAN` if either argument is `f64::NAN`
///
/// # Panics
///
/// if `a` or `x` are not in `(0, +inf)`
pub fn gamma_ur(a: f64, x: f64) -> f64 {
    checked_gamma_ur(a, x).unwrap()
}

/// Computes the upper incomplete regularized gamma function
/// `Q(a,x) = 1 / Gamma(a) * int(exp(-t)t^(a-1), t=0..x) for a > 0, x > 0`
/// where `a` is the argument for the gamma function and
/// `x` is the lower integral limit.
///
/// # Remarks
///
/// Returns `f64::NAN` if either argument is `f64::NAN`
///
/// # Errors
///
/// if `a` or `x` are not in `(0, +inf)`
pub fn checked_gamma_ur(a: f64, x: f64) -> Result<f64, GammaFuncError> {
    if a.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if a <= 0.0 || a == f64::INFINITY {
        return Err(GammaFuncError::AInvalid);
    }
    if x <= 0.0 || x == f64::INFINITY {
        return Err(GammaFuncError::XInvalid);
    }

    let eps = 0.000000000000001;
    let big = 4503599627370496.0;
    let big_inv = 2.22044604925031308085e-16;

    if x < 1.0 || x <= a {
        return Ok(1.0 - gamma_lr(a, x));
    }

    let mut ax = a * x.ln() - x - ln_gamma(a);
    if ax < -709.78271289338399 {
        return if a < x { Ok(0.0) } else { Ok(1.0) };
    }

    ax = ax.exp();
    let mut y = 1.0 - a;
    let mut z = x + y + 1.0;
    let mut c = 0.0;
    let mut pkm2 = 1.0;
    let mut qkm2 = x;
    let mut pkm1 = x + 1.0;
    let mut qkm1 = z * x;
    let mut ans = pkm1 / qkm1;
    loop {
        y += 1.0;
        z += 2.0;
        c += 1.0;
        let yc = y * c;
        let pk = pkm1 * z - pkm2 * yc;
        let qk = qkm1 * z - qkm2 * yc;

        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if pk.abs() > big {
            pkm2 *= big_inv;
            pkm1 *= big_inv;
            qkm2 *= big_inv;
            qkm1 *= big_inv;
        }

        if qk != 0.0 {
            let r = pk / qk;
            let t = ((ans - r) / r).abs();
            ans = r;

            if t <= eps {
                break;
            }
        }
    }
    Ok(ans * ax)
}

/// Computes the lower incomplete regularized gamma function
/// `P(a,x) = 1 / Gamma(a) * int(exp(-t)t^(a-1), t=0..x) for real a > 0, x > 0`
/// where `a` is the argument for the gamma function and `x` is the upper
/// integral limit.
///
/// # Remarks
///
/// Returns `f64::NAN` if either argument is `f64::NAN`
///
/// # Panics
///
/// if `a` or `x` are not in `(0, +inf)`
pub fn gamma_lr(a: f64, x: f64) -> f64 {
    checked_gamma_lr(a, x).unwrap()
}

/// Computes the lower incomplete regularized gamma function
/// `P(a,x) = 1 / Gamma(a) * int(exp(-t)t^(a-1), t=0..x) for real a > 0, x > 0`
/// where `a` is the argument for the gamma function and `x` is the upper
/// integral limit.
///
/// # Remarks
///
/// Returns `f64::NAN` if either argument is `f64::NAN`
///
/// # Errors
///
/// if `a` or `x` are not in `(0, +inf)`
pub fn checked_gamma_lr(a: f64, x: f64) -> Result<f64, GammaFuncError> {
    if a.is_nan() || x.is_nan() {
        return Ok(f64::NAN);
    }
    if a <= 0.0 || a == f64::INFINITY {
        return Err(GammaFuncError::AInvalid);
    }
    if x <= 0.0 || x == f64::INFINITY {
        return Err(GammaFuncError::XInvalid);
    }

    let eps = 0.000000000000001;
    let big = 4503599627370496.0;
    let big_inv = 2.22044604925031308085e-16;

    let ax = a * x.ln() - x - ln_gamma(a);
    if ax < -709.78271289338399 {
        if a < x {
            return Ok(1.0);
        }
        return Ok(0.0);
    }
    if x <= 1.0 || x <= a {
        let mut r2 = a;
        let mut c2 = 1.0;
        let mut ans2 = 1.0;
        loop {
            r2 += 1.0;
            c2 *= x / r2;
            ans2 += c2;

            if c2 / ans2 <= eps {
                break;
            }
        }
        return Ok(ax.exp() * ans2 / a);
    }

    let mut y = 1.0 - a;
    let mut z = x + y + 1.0;
    let mut c = 0;

    let mut p3 = 1.0;
    let mut q3 = x;
    let mut p2 = x + 1.0;
    let mut q2 = z * x;
    let mut ans = p2 / q2;

    loop {
        y += 1.0;
        z += 2.0;
        c += 1;
        let yc = y * f64::from(c);

        let p = p2 * z - p3 * yc;
        let q = q2 * z - q3 * yc;

        p3 = p2;
        p2 = p;
        q3 = q2;
        q2 = q;

        if p.abs() > big {
            p3 *= big_inv;
            p2 *= big_inv;
            q3 *= big_inv;
            q2 *= big_inv;
        }

        if q != 0.0 {
            let nextans = p / q;
            let error = ((ans - nextans) / nextans).abs();
            ans = nextans;

            if error <= eps {
                break;
            }
        }
    }
    Ok(1.0 - ax.exp() * ans)
}

/// Computes the Digamma function which is defined as the derivative of
/// the log of the gamma function. The implementation is based on
/// "Algorithm AS 103", Jose Bernardo, Applied Statistics, Volume 25, Number 3
/// 1976, pages 315 - 317
pub fn digamma(x: f64) -> f64 {
    let c = 12.0;
    let d1 = -0.57721566490153286;
    let d2 = 1.6449340668482264365;
    let s = 1e-6;
    let s3 = 1.0 / 12.0;
    let s4 = 1.0 / 120.0;
    let s5 = 1.0 / 252.0;
    let s6 = 1.0 / 240.0;
    let s7 = 1.0 / 132.0;

    if x == f64::NEG_INFINITY || x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 && x.floor() == x {
        return f64::NEG_INFINITY;
    }
    if x < 0.0 {
        // Reflection formula `psi(x) = psi(1 - x) - PI / tan(PI * x)`, with the
        // period reduction done by `tan_pi`. `(PI * x).tan()` evaluated directly
        // lost up to ~6 decimal digits near the poles (5586 ulp at x = -12.72,
        // 3e-7 relative at 1e-10 from a pole).
        return digamma(1.0 - x) - f64_consts::PI / tan_pi(x);
    }
    if x <= s {
        return d1 - 1.0 / x + d2 * x;
    }

    let mut result = 0.0;
    let mut z = x;
    while z < c {
        result -= 1.0 / z;
        z += 1.0;
    }

    if z >= c {
        let mut r = 1.0 / z;
        result += z.ln() - 0.5 * r;
        r *= r;

        result -= r * (s3 - r * (s4 - r * (s5 - r * (s6 - r * s7))));
    }
    result
}

pub fn inv_digamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::NEG_INFINITY {
        return 0.0;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    let mut y = x.exp();
    let mut i = 1.0;
    while i > 1e-15 {
        y += i * signum(x - digamma(y));
        i /= 2.0;
    }
    y
}

// modified signum that returns 0.0 if x == 0.0. Used
// by inv_digamma, may consider extracting into a public
// method
fn signum(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x.signum() }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::prec;

    use core::f64::consts;

    /// `x` walked `n` steps toward `+inf` via `f64::next_up`, i.e. `n` ULPs
    /// above `x`.
    fn ulps_above(x: f64, n: u32) -> f64 {
        (0..n).fold(x, |v, _| v.next_up())
    }

    #[test]
    fn test_gamma() {
        assert!(gamma(f64::NAN).is_nan());
        prec::assert_abs_diff_eq!(
            gamma(1.000001e-35),
            9.9999900000099999900000099999899999522784235098567139293e+34,
            epsilon = 1e20
        );
        prec::assert_abs_diff_eq!(
            gamma(1.000001e-10),
            9.99998999943278432519738283781280989934496494539074049002e+9,
            epsilon = 1e-5
        );
        prec::assert_abs_diff_eq!(
            gamma(1.000001e-5),
            99999.32279432557746387178953902739303931424932435387031653234,
            epsilon = 1e-10
        );
        prec::assert_abs_diff_eq!(
            gamma(1.000001e-2),
            99.43248512896257405886134437203369035261893114349805309870831,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma(-4.8),
            -0.06242336135475955314181664931547009890495158793105543559676,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma(-1.5),
            2.363271801207354703064223311121526910396732608163182837618410,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma(-0.5),
            -3.54490770181103205459633496668229036559509891224477425642761,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma(1.0e-5 + 1.0e-16),
            99999.42279322556767360213300482199406241771308740302819426480,
            epsilon = 1e-9
        );
        prec::assert_abs_diff_eq!(
            gamma(0.1),
            9.513507698668731836292487177265402192550578626088377343050000,
            epsilon = 1e-14
        );
        assert_eq!(
            gamma(1.0 - 1.0e-14),
            1.000000000000005772156649015427511664653698987042926067639529
        );
        prec::assert_abs_diff_eq!(gamma(1.0), 1.0, epsilon = 1e-15);
        prec::assert_abs_diff_eq!(
            gamma(1.0 + 1.0e-14),
            0.99999999999999422784335098477029953441189552403615306268023,
            epsilon = 1e-15
        );
        prec::assert_abs_diff_eq!(
            gamma(1.5),
            0.886226925452758013649083741670572591398774728061193564106903,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            gamma(consts::PI / 2.0),
            0.890560890381539328010659635359121005933541962884758999762766,
            epsilon = 1e-15
        );
        assert_eq!(gamma(2.0), 1.0);
        prec::assert_abs_diff_eq!(
            gamma(2.5),
            1.329340388179137020473625612505858887098162092091790346160355,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(gamma(3.0), 2.0, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(
            gamma(consts::PI),
            2.288037795340032417959588909060233922889688153356222441199380,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma(3.5),
            3.323350970447842551184064031264647217745405230229475865400889,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(gamma(4.0), 6.0, epsilon = 1e-13);
        prec::assert_abs_diff_eq!(
            gamma(4.5),
            11.63172839656744892914422410942626526210891830580316552890311,
            epsilon = 1e-12
        );
        prec::assert_abs_diff_eq!(
            gamma(5.0 - 1.0e-14),
            23.99999999999963853175957637087420162718107213574617032780374,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(gamma(5.0), 24.0, epsilon = 1e-12);
        prec::assert_abs_diff_eq!(
            gamma(5.0 + 1.0e-14),
            24.00000000000036146824042363510111050137786752408660789873592,
            epsilon = 1e-12
        );
        prec::assert_abs_diff_eq!(
            gamma(5.5),
            52.34277778455352018114900849241819367949013237611424488006401,
            epsilon = 1e-12
        );
        prec::assert_abs_diff_eq!(
            gamma(10.1),
            454760.7514415859508673358368319076190405047458218916492282448,
            epsilon = 1e-7
        );
        prec::assert_abs_diff_eq!(
            gamma(150.0 + 1.0e-12),
            3.8089226376496421386707466577615064443807882167327097140e+260,
            epsilon = 1e248
        );
    }

    #[test]
    fn test_ln_gamma() {
        assert!(super::ln_gamma(f64::NAN).is_nan());
        assert_eq!(
            super::ln_gamma(1.000001e-35),
            80.59047725479209894029636783061921392709972287131139201585211
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(1.000001e-10),
            23.02584992988323521564308637407936081168344192865285883337793,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(1.000001e-5),
            11.51291869289055371493077240324332039045238086972508869965363,
            epsilon = 1e-14
        );
        prec::assert_relative_eq!(
            super::ln_gamma(1.000001e-2),
            4.599478872433667224554543378460164306444416156144779542513592,
            epsilon = 0.0,
            max_relative = 1e-15
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(0.1),
            2.252712651734205959869701646368495118615627222294953765041739,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(1.0 - 1.0e-14),
            5.772156649015410852768463312546533565566459794933360600e-15,
            epsilon = 1e-15
        );
        prec::assert_abs_diff_eq!(super::ln_gamma(1.0), 0.0, epsilon = 1e-15);
        prec::assert_abs_diff_eq!(
            super::ln_gamma(1.0 + 1.0e-14),
            -5.77215664901524635936177848990288632404978978079827014e-15,
            epsilon = 1e-15
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(1.5),
            -0.12078223763524522234551844578164721225185272790259946836386,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(consts::PI / 2.0),
            -0.11590380084550241329912089415904874214542604767006895,
            epsilon = 1e-14
        );
        assert_eq!(super::ln_gamma(2.0), 0.0);
        prec::assert_abs_diff_eq!(
            super::ln_gamma(2.5),
            0.284682870472919159632494669682701924320137695559894729250145,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(3.0),
            f64_consts::LN_2,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(consts::PI),
            0.82769459232343710152957855845235995115350173412073715,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(3.5),
            1.200973602347074224816021881450712995770238915468157197042113,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(4.0),
            1.791759469228055000812477358380702272722990692183004705855374,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(4.5),
            2.453736570842442220504142503435716157331823510689763131380823,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(5.0 - 1.0e-14),
            3.178053830347930558470257283303394288448414225994179545985931,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(5.0),
            3.178053830347945619646941601297055408873990960903515214096734,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(5.0 + 1.0e-14),
            3.178053830347960680823625919312848824873279228348981287761046,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(5.5),
            3.957813967618716293877400855822590998551304491975006780729532,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(10.1),
            13.02752673863323795851370097886835481188051062306253294740504,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(150.0 + 1.0e-12),
            600.0094705553324354062157737572509902987070089159051628001813,
            epsilon = 1e-12
        );
        prec::assert_abs_diff_eq!(
            super::ln_gamma(1.001e+7),
            1.51342135323817913130119829455205139905331697084416059779e+8,
            epsilon = 1e-13
        );
    }

    #[test]
    fn test_gamma_lr() {
        assert!(gamma_lr(f64::NAN, f64::NAN).is_nan());
        prec::assert_abs_diff_eq!(
            gamma_lr(0.1, 1.0),
            0.97587265627367222115949155252812057714751052498477013,
            epsilon = 1e-14
        );
        assert_eq!(
            gamma_lr(0.1, 2.0),
            0.99432617602018847196075251078067514034772764693462125
        );
        assert_eq!(
            gamma_lr(0.1, 8.0),
            0.99999507519205198048686442150578226823401842046310854
        );
        prec::assert_abs_diff_eq!(
            gamma_lr(1.5, 1.0),
            0.42759329552912016600095238564127189392715996802703368,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_lr(1.5, 2.0),
            0.73853587005088937779717792402407879809718939080920993,
            epsilon = 1e-15
        );
        assert_eq!(
            gamma_lr(1.5, 8.0),
            0.99886601571021467734329986257903021041757398191304284
        );
        prec::assert_abs_diff_eq!(
            gamma_lr(2.5, 1.0),
            0.15085496391539036377410688601371365034788861473418704,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_lr(2.5, 2.0),
            0.45058404864721976739416885516693969548484517509263197,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            gamma_lr(2.5, 8.0),
            0.99315592607757956900093935107222761316136944145439676,
            epsilon = 1e-15
        );
        prec::assert_abs_diff_eq!(
            gamma_lr(5.5, 1.0),
            0.0015041182825838038421585211353488839717739161316985392,
            epsilon = 1e-15
        );
        prec::assert_abs_diff_eq!(
            gamma_lr(5.5, 2.0),
            0.030082976121226050615171484772387355162056796585883967,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            gamma_lr(5.5, 8.0),
            0.85886911973294184646060071855669224657735916933487681,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(gamma_lr(100.0, 0.5), 0.0, epsilon = 1e-188);
        prec::assert_abs_diff_eq!(gamma_lr(100.0, 1.5), 0.0, epsilon = 1e-141);
        prec::assert_abs_diff_eq!(
            gamma_lr(100.0, 90.0),
            0.1582209891864301681049696996709105316998233457433473,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_lr(100.0, 100.0),
            0.5132987982791486648573142565640291634709251499279450,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_lr(100.0, 110.0),
            0.8417213299399129061982996209829688531933500308658222,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(gamma_lr(100.0, 200.0), 1.0, epsilon = 1e-14);
        assert_eq!(gamma_lr(500.0, 0.5), 0.0);
        assert_eq!(gamma_lr(500.0, 1.5), 0.0);
        prec::assert_abs_diff_eq!(gamma_lr(500.0, 200.0), 0.0, epsilon = 1e-70);
        prec::assert_abs_diff_eq!(
            gamma_lr(500.0, 450.0),
            0.0107172380912897415573958770655204965434869949241480,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            gamma_lr(500.0, 500.0),
            0.5059471461707603580470479574412058032802735425634263,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_lr(500.0, 550.0),
            0.9853855918737048059548470006900844665580616318702748,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(gamma_lr(500.0, 700.0), 1.0, epsilon = 1e-15);
        assert_eq!(gamma_lr(1000.0, 10000.0), 1.0);
        assert_eq!(gamma_lr(1e+50, 1e+48), 0.0);
        assert_eq!(gamma_lr(1e+50, 1e+52), 1.0);
    }

    #[test]
    fn test_gamma_lr_a_near_zero_boundary() {
        // P(a, x) -> 1 as a -> 0+, for any fixed x > 0. `a` can approach the
        // boundary from one side only (a > 0 is required), so check that the
        // sequence climbs monotonically to the limit as a shrinks toward it.
        let x = 1.0;
        // `a` values small enough that P(a, x) is already within a few ULPs
        // of 1.0, but far from the boundary the `a == 0.0` short circuit
        // guards; then the literal near-zero region the short circuit
        // covers, where the series expansion below underflows to exactly
        // 1.0 on its own. Values much smaller than 1e-10 pick up several
        // ULPs of noise from cancellation in the series/exp evaluation, so
        // they're intentionally left out of the monotonicity check below.
        let as_: [f64; 4] = [
            1e-2,
            1e-10,
            ulps_above(0.0, 5), // a few ULPs above 0.0
            0.0f64.next_up(),   // smallest positive subnormal, 1 ULP above 0.0
        ];
        let mut prev = 0.0;
        for &a in &as_ {
            let p = gamma_lr(a, x);
            assert!(p <= 1.0, "gamma_lr({a}, {x}) = {p} exceeds 1.0");
            assert!(p >= prev, "gamma_lr({a}, {x}) = {p} regressed below {prev}");
            prev = p;
        }
        assert_eq!(prev, 1.0);
    }

    #[test]
    fn test_gamma_ur_a_near_zero_boundary() {
        // Q(a, x) = 1 - P(a, x) is the complement gamma_lr's `a == 0.0` short
        // circuit collapses to exactly 0.0 for. Unlike `1.0 - gamma_lr(a, x)`,
        // `gamma_ur` doesn't route through a subtraction from 1.0 here (its
        // own `x < 1.0 || x <= a` short circuit doesn't fire for x = 1 and
        // a < 1), so it keeps ~15 significant digits of `a`-dependent
        // resolution across roughly 300 decades - via `ln_gamma`, which never
        // saturates - instead of losing everything the moment P(a, x) rounds
        // to 1.0 (which happens once a is only a few multiples of
        // `f64::EPSILON`). That resolution is what lets this assert *strict*
        // descent: if the short circuit were hiding a real discontinuity in
        // the general formula, collapsing this sequence to ties, this test
        // would catch it in a way the saturated `gamma_lr` value cannot.
        let x = 1.0;
        let as_: [f64; 8] = [1e-2, 1e-10, 1e-50, 1e-100, 1e-150, 1e-200, 1e-250, 1e-300];
        let mut prev = f64::INFINITY;
        for &a in &as_ {
            let q = gamma_ur(a, x);
            assert!(q > 0.0 && q < 1.0, "gamma_ur({a}, {x}) = {q} out of (0, 1)");
            assert!(q < prev, "gamma_ur({a}, {x}) = {q} did not strictly descend below {prev}");
            prev = q;
        }
        // Below `a`'s double-precision underflow threshold (~1e-308 here),
        // Q(a, x) genuinely can't be distinguished from 0.0 - that's a real
        // hardware floor, not a collapse the short circuit is responsible
        // for, so the tail is checked for exact equality instead of descent.
        for a in [ulps_above(0.0, 5), 0.0f64.next_up()] {
            assert_eq!(gamma_ur(a, x), 0.0);
        }
    }

    #[test]
    fn test_gamma_lr_x_near_zero_boundary() {
        // P(a, x) -> 0 as x -> 0+, for any fixed a > 0. `x` can approach the
        // boundary from one side only (x > 0 is required). Unlike the `a`
        // direction, P(a, x) is itself the small quantity here (it isn't
        // reached by subtracting from 1.0), so - for a = 1, where
        // P(1, x) = 1 - exp(-x) ~ x - it keeps ~15 significant digits of
        // `x`-dependent resolution across roughly 300 decades, letting this
        // assert *strict* descent the way `gamma_lr`'s own saturated `a == 0`
        // direction can't: if the short circuit were hiding a real
        // discontinuity in the general formula, this would catch it.
        let a = 1.0;
        let xs: [f64; 8] = [1e-2, 1e-10, 1e-50, 1e-100, 1e-150, 1e-200, 1e-250, 1e-300];
        let mut prev = f64::INFINITY;
        for &x in &xs {
            let p = gamma_lr(a, x);
            assert!(p > 0.0 && p < 1.0, "gamma_lr({a}, {x}) = {p} out of (0, 1)");
            assert!(p < prev, "gamma_lr({a}, {x}) = {p} did not strictly descend below {prev}");
            prev = p;
        }
        // Below `x`'s double-precision underflow threshold (~1e-308 here),
        // P(a, x) genuinely can't be distinguished from 0.0 - that's a real
        // hardware floor, not a collapse the short circuit is responsible
        // for, so the tail is checked for exact equality instead of descent.
        for x in [ulps_above(0.0, 5), 0.0f64.next_up()] {
            assert_eq!(gamma_lr(a, x), 0.0);
        }
    }

    #[test]
    #[should_panic]
    fn test_gamma_lr_a_lower_bound() {
        gamma_lr(-1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_lr_a_upper_bound() {
        gamma_lr(f64::INFINITY, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_lr_x_lower_bound() {
        gamma_lr(1.0, -1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_lr_x_upper_bound() {
        gamma_lr(1.0, f64::INFINITY);
    }

    #[test]
    fn test_checked_gamma_lr_a_lower_bound() {
        assert!(super::checked_gamma_lr(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_gamma_lr_a_upper_bound() {
        assert!(super::checked_gamma_lr(f64::INFINITY, 1.0).is_err());
    }

    #[test]
    fn test_checked_gamma_lr_x_lower_bound() {
        assert!(super::checked_gamma_lr(1.0, -1.0).is_err());
    }

    #[test]
    fn test_checked_gamma_lr_x_upper_bound() {
        assert!(super::checked_gamma_lr(1.0, f64::INFINITY).is_err());
    }

    #[test]
    fn test_gamma_li() {
        assert!(gamma_li(f64::NAN, f64::NAN).is_nan());
        prec::assert_abs_diff_eq!(
            gamma_li(0.1, 1.0),
            9.2839720283798852469443229940217320532607158711056334,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            gamma_li(0.1, 2.0),
            9.4595297305559030536119885480983751098528458886962883,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            gamma_li(0.1, 8.0),
            9.5134608464704033372127589212547718314010339263844976,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_li(1.5, 1.0),
            0.37894469164098470380394366597039213790868855578083847,
            epsilon = 1e-15
        );
        prec::assert_abs_diff_eq!(
            gamma_li(1.5, 2.0),
            0.65451037345177732033319477475056262302270310457635612,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            gamma_li(1.5, 8.0),
            0.88522195804210983776635107858848816480298923071075222,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_li(2.5, 1.0),
            0.20053759629003473411039172879412733941722170263949,
            epsilon = 1e-16
        );
        prec::assert_abs_diff_eq!(
            gamma_li(2.5, 2.0),
            0.59897957413602228465664030130712917348327070206302442,
            epsilon = 1e-15
        );
        prec::assert_abs_diff_eq!(
            gamma_li(2.5, 8.0),
            1.3202422842943799358198434659248530581833764879301293,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            gamma_li(5.5, 1.0),
            0.078729729026968321691794205337720556329618007004848672,
            epsilon = 1e-16
        );
        prec::assert_abs_diff_eq!(
            gamma_li(5.5, 2.0),
            1.5746265342113649473739798668921124454837064926448459,
            epsilon = 2e-15
        );
        prec::assert_abs_diff_eq!(
            gamma_li(5.5, 8.0),
            44.955595480196465884619737757794960132425035578313584,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic]
    fn test_gamma_li_a_lower_bound() {
        gamma_li(-1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_li_a_upper_bound() {
        gamma_li(f64::INFINITY, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_li_x_lower_bound() {
        gamma_li(1.0, -1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_li_x_upper_bound() {
        gamma_li(1.0, f64::INFINITY);
    }

    #[test]
    fn test_checked_gamma_li_a_lower_bound() {
        assert!(super::checked_gamma_li(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_gamma_li_a_upper_bound() {
        assert!(super::checked_gamma_li(f64::INFINITY, 1.0).is_err());
    }

    #[test]
    fn test_checked_gamma_li_x_lower_bound() {
        assert!(super::checked_gamma_li(1.0, -1.0).is_err());
    }

    #[test]
    fn test_checked_gamma_li_x_upper_bound() {
        assert!(super::checked_gamma_li(1.0, f64::INFINITY).is_err());
    }

    // TODO: precision testing could be more accurate, borrowed wholesale from Math.NET
    #[test]
    fn test_gamma_ur() {
        assert!(gamma_ur(f64::NAN, f64::NAN).is_nan());
        prec::assert_abs_diff_eq!(
            gamma_ur(0.1, 1.0),
            0.0241273437263277773829694356333550393309597428392044,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_ur(0.1, 2.0),
            0.0056738239798115280392474892193248596522723530653781,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_ur(0.1, 8.0),
            0.0000049248079480195131355784942177317659815795368919702,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_ur(1.5, 1.0),
            0.57240670447087983399904761435872810607284003197297,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_ur(1.5, 2.0),
            0.26146412994911062220282207597592120190281060919079,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_ur(1.5, 8.0),
            0.0011339842897853226567001374209697895824260180869567,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_ur(2.5, 1.0),
            0.84914503608460963622589311398628634965211138526581,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_ur(2.5, 2.0),
            0.54941595135278023260583114483306030451515482490737,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_ur(2.5, 8.0),
            0.0068440739224204309990606489277723868386305585456026,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_ur(5.5, 1.0),
            0.9984958817174161961578414788646511160282260838683,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_ur(5.5, 2.0),
            0.96991702387877394938482851522761264483794320341412,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(
            gamma_ur(5.5, 8.0),
            0.14113088026705815353939928144330775342264083066512,
            epsilon = 1e-13
        );
        prec::assert_abs_diff_eq!(gamma_ur(100.0, 0.5), 1.0, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(gamma_ur(100.0, 1.5), 1.0, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(
            gamma_ur(100.0, 90.0),
            0.8417790108135698318950303003290894683001766542566526,
            epsilon = 1e-12
        );
        prec::assert_abs_diff_eq!(
            gamma_ur(100.0, 100.0),
            0.4867012017208513351426857434359708365290748500720549,
            epsilon = 1e-12
        );
        prec::assert_abs_diff_eq!(
            gamma_ur(100.0, 110.0),
            0.1582786700600870938017003790170311468066499691341777,
            epsilon = 1e-12
        );
        prec::assert_abs_diff_eq!(gamma_ur(100.0, 200.0), 0.0, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(gamma_ur(500.0, 0.5), 1.0, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(gamma_ur(500.0, 1.5), 1.0, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(gamma_ur(500.0, 200.0), 1.0, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(
            gamma_ur(500.0, 450.0),
            0.9892827619087102584426041229344795034565130050758519,
            epsilon = 1e-12
        );
        prec::assert_abs_diff_eq!(
            gamma_ur(500.0, 500.0),
            0.4940528538292396419529520425587941967197264574365736,
            epsilon = 1e-12
        );
        prec::assert_abs_diff_eq!(
            gamma_ur(500.0, 550.0),
            0.0146144081262951940451529993099155334419383681297251,
            epsilon = 1e-12
        );
        prec::assert_abs_diff_eq!(gamma_ur(500.0, 700.0), 0.0, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(gamma_ur(1000.0, 10000.0), 0.0, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(gamma_ur(1e+50, 1e+48), 1.0, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(gamma_ur(1e+50, 1e+52), 0.0, epsilon = 1e-14);
    }

    #[test]
    #[should_panic]
    fn test_gamma_ur_a_lower_bound() {
        gamma_ur(-1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_ur_a_upper_bound() {
        gamma_ur(f64::INFINITY, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_ur_x_lower_bound() {
        gamma_ur(1.0, -1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_ur_x_upper_bound() {
        gamma_ur(1.0, f64::INFINITY);
    }

    #[test]
    fn test_checked_gamma_ur_a_lower_bound() {
        assert!(super::checked_gamma_ur(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_gamma_ur_a_upper_bound() {
        assert!(super::checked_gamma_ur(f64::INFINITY, 1.0).is_err());
    }

    #[test]
    fn test_checked_gamma_ur_x_lower_bound() {
        assert!(super::checked_gamma_ur(1.0, -1.0).is_err());
    }

    #[test]
    fn test_checked_gamma_ur_x_upper_bound() {
        assert!(super::checked_gamma_ur(1.0, f64::INFINITY).is_err());
    }

    #[test]
    fn test_gamma_ui() {
        assert!(gamma_ui(f64::NAN, f64::NAN).is_nan());
        prec::assert_abs_diff_eq!(
            gamma_ui(0.1, 1.0),
            0.2295356702888460382790772147651768201739736396141314,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            gamma_ui(0.1, 2.0),
            0.053977968112828232195991347726857391060870217694027,
            epsilon = 1e-15
        );
        prec::assert_abs_diff_eq!(
            gamma_ui(0.1, 8.0),
            0.000046852198327948595220974570460669512682180005810156,
            epsilon = 1e-19
        );
        prec::assert_abs_diff_eq!(
            gamma_ui(1.5, 1.0),
            0.50728223381177330984514007570018045349008617228036,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            gamma_ui(1.5, 2.0),
            0.23171655200098069331588896692000996837607162348484,
            epsilon = 1e-15
        );
        prec::assert_abs_diff_eq!(
            gamma_ui(1.5, 8.0),
            0.0010049674106481758827326630820844265957854973504417,
            epsilon = 1e-17
        );
        prec::assert_abs_diff_eq!(
            gamma_ui(2.5, 1.0),
            1.1288027918891022863632338837117315476809403894523,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            gamma_ui(2.5, 2.0),
            0.73036081404311473581698531119872971361489139002877,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            gamma_ui(2.5, 8.0),
            0.0090981038847570846537821465810058289147856041616617,
            epsilon = 1e-17
        );
        prec::assert_abs_diff_eq!(
            gamma_ui(5.5, 1.0),
            52.264048055526551859457214287080473123160514369109,
            epsilon = 1e-12
        );
        prec::assert_abs_diff_eq!(
            gamma_ui(5.5, 2.0),
            50.768151250342155233775028625526081234006425883469,
            epsilon = 1e-12
        );
        prec::assert_abs_diff_eq!(
            gamma_ui(5.5, 8.0),
            7.3871823043570542965292707346232335470650967978006,
            epsilon = 1e-13
        );
    }

    #[test]
    #[should_panic]
    fn test_gamma_ui_a_lower_bound() {
        gamma_ui(-1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_ui_a_upper_bound() {
        gamma_ui(f64::INFINITY, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_ui_x_lower_bound() {
        gamma_ui(1.0, -1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_ui_x_upper_bound() {
        gamma_ui(1.0, f64::INFINITY);
    }

    #[test]
    fn test_checked_gamma_ui_a_lower_bound() {
        assert!(super::checked_gamma_ui(-1.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_gamma_ui_a_upper_bound() {
        assert!(super::checked_gamma_ui(f64::INFINITY, 1.0).is_err());
    }

    #[test]
    fn test_checked_gamma_ui_x_lower_bound() {
        assert!(super::checked_gamma_ui(1.0, -1.0).is_err());
    }

    #[test]
    fn test_checked_gamma_ui_x_upper_bound() {
        assert!(super::checked_gamma_ui(1.0, f64::INFINITY).is_err());
    }

    /// `digamma` has poles at the non-positive integers and returns -inf there.
    /// The pole test is now `x.floor() == x`, exact; it used to be
    /// `prec::ulps_eq!(x.floor(), x)` with a `1e-9` absolute default, so a
    /// whole neighbourhood around each pole returned -inf instead of a large
    /// finite value.
    #[test]
    fn test_digamma_near_negative_integer_poles_is_finite() {
        prec::assert_relative_eq!(digamma(-1.0 + f64::powi(2.0, -33)), -8589934591.5772156646, epsilon = 0.0, max_relative = 1e-13);
        prec::assert_relative_eq!(digamma(-1.0 - f64::powi(2.0, -33)),  8589934592.4227843348, epsilon = 0.0, max_relative = 1e-13);
        // the poles themselves are unchanged
        assert_eq!(digamma(-1.0), f64::NEG_INFINITY);
        assert_eq!(digamma(0.0), f64::NEG_INFINITY);
        assert_eq!(digamma(-5.0), f64::NEG_INFINITY);
    }

    /// Before `tan_pi` reduced the reflection argument by the period, these
    /// lost 3-4 decimal digits because `(PI * x).tan()` was evaluated at a
    /// large rounded argument. Reference values are mpmath at 40 significant
    /// digits, computed at the exact `f64` of each literal. The tolerances
    /// reflect the cancellation between the two reflection terms
    /// (`psi(1 - x)` and `pi * cot(pi x)` are each ~2.6 at x = -12.72 and
    /// cancel to ~0.017, amplifying their ~1 ulp errors ~150x).
    #[test]
    fn test_digamma_negative_arguments_far_from_zero() {
        prec::assert_relative_eq!(digamma(-12.72), -0.0169824608177603739173, epsilon = 0.0, max_relative = 1e-13);
        prec::assert_relative_eq!(digamma(-14.72), 0.1238386191670285663687, epsilon = 0.0, max_relative = 1e-14);
        prec::assert_relative_eq!(digamma(-20.02), 52.95568424702714411621, epsilon = 0.0, max_relative = 1e-15);
    }

    // TODO: precision testing could be more accurate
    #[test]
    fn test_digamma() {
        assert!(super::digamma(f64::NAN).is_nan());
        prec::assert_abs_diff_eq!(
            super::digamma(-1.5),
            0.70315664064524318722569033366791109947350706200623256,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::digamma(-0.5),
            0.036489973978576520559023667001244432806840395339565891,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::digamma(0.1),
            -10.423754940411076232100295314502760886768558023951363,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::digamma(1.0),
            -0.57721566490153286060651209008240243104215933593992359,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::digamma(1.5),
            0.036489973978576520559023667001244432806840395339565888,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::digamma(consts::PI / 2.0),
            0.10067337642740238636795561404029690452798358068944001,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::digamma(2.0),
            0.42278433509846713939348790991759756895784066406007641,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::digamma(2.5),
            0.70315664064524318722569033366791109947350706200623255,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::digamma(3.0),
            0.92278433509846713939348790991759756895784066406007641,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::digamma(consts::PI),
            0.97721330794200673329206948640618234364083460999432603,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::digamma(3.5),
            1.1031566406452431872256903336679110994735070620062326,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::digamma(4.0),
            1.2561176684318004727268212432509309022911739973934097,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::digamma(4.5),
            1.3888709263595289015114046193821968137592213477205183,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::digamma(5.0),
            1.5061176684318004727268212432509309022911739973934097,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::digamma(5.5),
            1.6110931485817511237336268416044190359814435699427405,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::digamma(10.1),
            2.2622143570941481235561593642219403924532310597356171,
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_digamma_near_negative_integer_is_finite() {
        assert!(super::digamma(-1.0 + 5e-10).is_finite());
    }

    #[test]
    fn test_inv_digamma() {
        assert!(super::inv_digamma(f64::NAN).is_nan());
        assert_eq!(super::inv_digamma(f64::NEG_INFINITY), 0.0);
        prec::assert_abs_diff_eq!(
            super::inv_digamma(-10.423754940411076232100295314502760886768558023951363),
            0.1,
            epsilon = 1e-15
        );
        prec::assert_abs_diff_eq!(
            super::inv_digamma(-0.57721566490153286060651209008240243104215933593992359),
            1.0,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::inv_digamma(0.036489973978576520559023667001244432806840395339565888),
            1.5,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::inv_digamma(0.10067337642740238636795561404029690452798358068944001),
            consts::PI / 2.0,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::inv_digamma(0.42278433509846713939348790991759756895784066406007641),
            2.0,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::inv_digamma(0.70315664064524318722569033366791109947350706200623255),
            2.5,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::inv_digamma(0.92278433509846713939348790991759756895784066406007641),
            3.0,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::inv_digamma(0.97721330794200673329206948640618234364083460999432603),
            consts::PI,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::inv_digamma(1.1031566406452431872256903336679110994735070620062326),
            3.5,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::inv_digamma(1.2561176684318004727268212432509309022911739973934097),
            4.0,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::inv_digamma(1.3888709263595289015114046193821968137592213477205183),
            4.5,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::inv_digamma(1.5061176684318004727268212432509309022911739973934097),
            5.0,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::inv_digamma(1.6110931485817511237336268416044190359814435699427405),
            5.5,
            epsilon = 1e-14
        );
        prec::assert_abs_diff_eq!(
            super::inv_digamma(2.2622143570941481235561593642219403924532310597356171),
            10.1,
            epsilon = 1e-13
        );
    }

    /// `gamma` must overflow to `+inf`, never `-inf`, and must not overflow
    /// early. The first-order correction in `lanczos_power` is only a valid
    /// perturbation while `|p * delta| << 1`; applying it unguarded let a
    /// negative correction flip the sign for `x` past ~1e15. Separately, the
    /// power alone overflows from about x = 169.7 while the full product stays
    /// representable to x ~ 171.61.
    #[test]
    fn test_gamma_overflow_boundary_and_sign() {
        // finite and positive right up to the true overflow point
        for x in [168.0f64, 169.0, 169.7, 170.5, 171.0, 171.5, 171.6] {
            let g = gamma(x);
            assert!(g.is_finite() && g > 0.0, "gamma({x}) should be finite positive, got {g}");
        }
        // mpmath at 30 digits
        prec::assert_relative_eq!(gamma(171.6), 1.5858969096673029e308, epsilon = 0.0, max_relative = 1e-13);
        prec::assert_relative_eq!(gamma(170.5), 5.5620924145599996e305, epsilon = 0.0, max_relative = 1e-13);
        prec::assert_relative_eq!(gamma(169.7), 9.155822000376269e303, epsilon = 0.0, max_relative = 1e-13);
        // and always +inf beyond it, at every scale
        for x in [172.0f64, 200.0, 1e3, 1e6, 1e14, 1e15, 1e100, 1e300, f64::MAX] {
            let g = gamma(x);
            assert!(g.is_infinite() && g > 0.0, "gamma({x:e}) should be +inf, got {g}");
        }
    }

    #[test]
    fn test_error_is_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<GammaFuncError>();
    }

    #[test]
    fn test_gamma_lr_x_near_zero_boundary_small_a() {
        // P(a, x) ~ x^a / (a * Gamma(a)) for small x, and x^a -> 1 as a -> 0,
        // so for small a this stays well clear of the x == 0.0 boundary even
        // a handful of ULPs above it - regression guard for a since-fixed
        // bug where a tolerance-based boundary check collapsed this to 0.0.
        let a = 0.001;
        for bits in 1..=5u64 {
            let x = f64::from_bits(bits);
            let p = gamma_lr(a, x);
            assert!(
                p > 0.0,
                "gamma_lr({a}, {bits} ULPs above 0.0) = {p}, expected > 0.0"
            );
        }
    }
}
