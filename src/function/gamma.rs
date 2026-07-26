//! Provides the [gamma](https://en.wikipedia.org/wiki/Gamma_function) and
//! related functions

use crate::consts;
use crate::function::evaluate;
use crate::prec;
use crate::prec::{dekker_product_err, two_diff};
use core::f64;
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

// ---------------------------------------------------------------------------
// Saddle-point building blocks (Loader, "Fast and Accurate Computation of
// Binomial Probabilities", 2000).
//
// Densities and incomplete-function prefixes in the gamma/beta family are
// naturally written as `exp(a * ln x - x - ln_gamma(a))` and friends. Those
// forms are numerically poor: for `a ~ 1e4` each term is `~1e5` while the sum
// is `O(1)`, so the `~1e-11` absolute error of the sum becomes the accuracy
// ceiling of everything downstream.
//
// The fix is to split each such exponent into two pieces that are individually
// `O(1)`:
//
//   * `stirling_delta(z)`, the Stirling-series remainder of `ln_gamma`, and
//   * `bd0(x, np)`, the "deviance" `x * ln(x / np) - (x - np)`, which is
//     evaluated by a series in `(x - np) / (x + np)` so that the cancellation
//     for `x` near `np` is performed analytically rather than in floating
//     point.
//
// Nothing large is ever formed, so the cancellation disappears entirely.
// ---------------------------------------------------------------------------

/// Coefficients `B_2n / (2n (2n - 1))` of the Stirling series for
/// [`stirling_delta`], ascending in `1/z^2`.
const STIRLING_SERIES: &[f64] = &[
    0.08333333333333333,    // 1/12
    -0.002777777777777778,  // -1/360
    0.0007936507936507937,  // 1/1260
    -0.0005952380952380953, // -1/1680
    0.0008417508417508417,  // 1/1188
    -0.0019175269175269176, // -691/360360
];

/// Argument above which the truncated Stirling series is used directly. Six
/// terms there are good to `1.4e-18` absolute, i.e. invisible.
pub(crate) const STIRLING_SERIES_MIN: f64 = 16.0;

/// The Stirling-series remainder of `ln_gamma`, for `z > 0`:
///
/// ```text
/// stirling_delta(z) = ln_gamma(z) - [(z - 1/2) ln z - z + ln(2 pi) / 2]
/// ```
///
/// Equivalently `ln(n!) - ln(sqrt(2 pi n) (n / e)^n)` at `z = n`, which is
/// Loader's `stirlerr`. The value is `O(1 / (12 z))`, and the implementation
/// keeps it accurate to `~1e-15` *absolute* - which is what matters, since
/// every caller adds it to an exponent.
///
/// Below [`STIRLING_SERIES_MIN`] the recurrence
/// `delta(z) = delta(z + 1) + (z + 1/2) ln(1 + 1/z) - 1` lifts the argument
/// into the series range. Each step costs one rounding, so no table (and no
/// recursive dependency on [`ln_gamma`]) is needed.
pub(crate) fn stirling_delta(z: f64) -> f64 {
    let mut acc = 0.0;
    let mut w = z;
    while w < STIRLING_SERIES_MIN {
        acc += (w + 0.5) * (1.0 / w).ln_1p() - 1.0;
        w += 1.0;
    }
    // Horner in 1/z^2. For `w * w == inf` this collapses to the leading
    // `1 / (12 w)` term, which is the correct limit.
    let ww = w * w;
    let series = STIRLING_SERIES.iter().rev().fold(0.0, |s, &c| s / ww + c);
    acc + series / w
}

/// The deviance `bd0(x, np) = x * ln(x / np) - (x - np)`, for `x >= 0`,
/// `np > 0`.
///
/// This is the Kullback-Leibler-like term of the Poisson/binomial saddle-point
/// expansions, and is non-negative with a double root at `x == np`. Writing it
/// directly loses all precision near that root; with
/// `v = (x - np) / (x + np)` it has the all-positive series
///
/// ```text
/// bd0 = (x - np) v + 2 x sum_{j >= 1} v^(2j + 1) / (2j + 1)
/// ```
///
/// which is used whenever `|x - np| < (x + np) / 10`, giving `~1e-16` relative
/// accuracy uniformly.
pub(crate) fn bd0(x: f64, np: f64) -> f64 {
    if x == 0.0 {
        // 0 * ln 0 == 0 by convention; the direct form would give NaN
        return np;
    }
    if (x - np).abs() < 0.1 * (x + np) {
        let v = (x - np) / (x + np);
        let mut s = (x - np) * v;
        if s.abs() < f64::MIN_POSITIVE {
            return s;
        }
        let mut ej = 2.0 * x * v;
        let v2 = v * v;
        // |v| < 0.1, so v^(2j) is below f64::MIN_POSITIVE well before j = 200
        for j in 1..200 {
            ej *= v2;
            let s1 = s + ej / (2 * j + 1) as f64;
            if s1 == s {
                return s1;
            }
            s = s1;
        }
        return s;
    }
    // `(x / np).ln()` silently loses the whole term when the ratio leaves the
    // normal range - `bd0(1e-300, 5e99)` has `x / np` underflow to zero, giving
    // `-inf` where the answer is `+5e99`, which then poisoned the beta prefix
    // into NaN. Reaching this branch requires `|x - np| >= (x + np) / 10`, so
    // the ratio is bounded away from 1 and the difference of logs cannot cancel
    // badly; it is only used where the ratio itself is unusable, since it is
    // otherwise the less accurate of the two forms.
    let ratio = x / np;
    let log_ratio = if (f64::MIN_POSITIVE..f64::INFINITY).contains(&ratio) {
        ratio.ln()
    } else {
        x.ln() - np.ln()
    };
    x * log_ratio - (x - np)
}

/// [`bd0`] with the mean supplied as an exact double-double `np_hi + np_lo`.
///
/// The mean usually arrives as a rounded product such as `n * p`, and `bd0` is
/// sensitive enough for that last half-ulp to dominate everything else:
/// `d bd0 / d np = 1 - x / np`, so at `n = 2e6`, `p = 0.3` the rounding of
/// `n * p` alone shifts `bd0` by `~2.5e-13` - a thousand ulps in the resulting
/// pmf. The correction is first order in `np_lo`; the next term is smaller by
/// another factor of `np_lo / np`, i.e. utterly negligible.
pub(crate) fn bd0_dd(x: f64, np_hi: f64, np_lo: f64) -> f64 {
    bd0(x, np_hi) + (1.0 - x / np_hi) * np_lo
}

/// `ln(x^a e^-x / Gamma(a))`, the prefix of the incomplete gamma functions.
///
/// Written out, this is `a ln x - x - ln_gamma(a)`, where all three terms grow
/// like `a ln a` while the sum stays `O(ln a)`. The saddle-point form
///
/// ```text
/// = -bd0(a, x) - stirling_delta(a) + ln(a / (2 pi)) / 2
/// ```
///
/// is the same quantity with every piece `O(1)`, so it is accurate to a few
/// `1e-16` absolute instead of `a ln a * eps` (`~1e-11` at `a = 1e4`).
pub(crate) fn ln_gamma_prefix(a: f64, x: f64) -> f64 {
    if a < STIRLING_SERIES_MIN {
        // No cancellation to remove yet (every term is already `O(1)`), and the
        // recurrence in `stirling_delta` would cost more roundings than it
        // saves.
        return a * x.ln() - x - ln_gamma(a);
    }
    -bd0(a, x) - stirling_delta(a) + 0.5 * (a / f64::consts::TAU).ln()
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
            - (0.5 - x) * ((0.5 - x + GAMMA_R) / f64::consts::E).ln()
    } else {
        let s = lanczos_sum(x);

        s.ln()
            + consts::LN_2_SQRT_E_OVER_PI
            + (x - 0.5) * ((x - 0.5 + GAMMA_R) / f64::consts::E).ln()
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

        f64::consts::PI
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

    // saddle-point form; see `ln_gamma_prefix`
    let mut ax = ln_gamma_prefix(a, x);
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

    if prec::ulps_eq!(a, 0.0, epsilon = 0.0) {
        return Ok(1.0);
    }
    if prec::ulps_eq!(x, 0.0, epsilon = 0.0) {
        return Ok(0.0);
    }

    // saddle-point form; see `ln_gamma_prefix`
    let ax = ln_gamma_prefix(a, x);
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
    if x <= 0.0 && prec::ulps_eq!(x.floor(), x) {
        return f64::NEG_INFINITY;
    }
    if x < 0.0 {
        // Reflection formula `psi(x) = psi(1 - x) - PI / tan(PI * x)`, with the
        // period reduction done by `tan_pi`. `(PI * x).tan()` evaluated directly
        // lost up to ~6 decimal digits near the poles (5586 ulp at x = -12.72,
        // 3e-7 relative at 1e-10 from a pole).
        return digamma(1.0 - x) - f64::consts::PI / tan_pi(x);
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
            f64::consts::LN_2,
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
    /// The pole test is `prec::ulps_eq!(x.floor(), x)`, whose default epsilon
    /// used to be `1e-9` absolute, so a whole neighbourhood around each pole
    /// returned -inf instead of a large finite value.
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

    /// `bd0`'s direct branch formed `(x / np).ln()`, which underflows to
    /// `ln(0) = -inf` when `x` is tiny and `np` huge. That propagated through
    /// the beta prefix and made `beta_reg` return NaN (or silently 1.0 instead
    /// of 0.0) for extreme parameter ratios.
    #[test]
    fn test_bd0_extreme_ratios() {
        // x / np underflows; true value is ~np
        let v = super::bd0(1e-300, 5e99);
        prec::assert_relative_eq!(v, 5e99, epsilon = 0.0, max_relative = 1e-14);
        // x / np overflows; true value is x*ln(x/np) - x + np
        let v = super::bd0(5e99, 1e-300);
        assert!(v.is_finite() && v > 0.0, "bd0(5e99, 1e-300) = {v}");
        // and it stays non-negative and finite across a wide ratio sweep
        for ea in [-300i32, -100, -10, 0, 10, 100, 300] {
            for eb in [-300i32, -100, -10, 0, 10, 100, 300] {
                let (x, np) = (10f64.powi(ea), 10f64.powi(eb));
                let v = super::bd0(x, np);
                assert!(v.is_finite() && v >= 0.0, "bd0(1e{ea}, 1e{eb}) = {v}");
            }
        }
    }

    /// `stirling_delta` lands in an exponent, so its *absolute* accuracy is what
    /// matters. References are mpmath at 40 significant digits; the tolerances
    /// sit just above the measured error (`~1e-15` below the series threshold,
    /// essentially exact above it).
    #[test]
    fn test_stirling_delta() {
        // below the series threshold the recurrence costs one rounding per step,
        // so the bound is absolute; at or above it the series is limited only by
        // its own evaluation, so a relative bound is the meaningful one
        for (z, want) in [
            (0.5, 0.15342640972002734529),
            (1.0, 0.08106146679532725822),
            (2.5, 0.033162873519936287485),
            (8.0, 0.010411265261972096497),
            (15.5, 0.0053755990329268344936),
        ] {
            prec::assert_abs_diff_eq!(super::stirling_delta(z), want, epsilon = 2e-15);
        }
        for (z, want) in [
            (16.0, 0.0052076559196096404407),
            (100.0, 0.00083333055563491468338),
            (10000.0, 8.3333333305555555635e-6),
        ] {
            prec::assert_relative_eq!(
                super::stirling_delta(z),
                want,
                epsilon = 0.0,
                max_relative = 1e-15
            );
        }
        // consistency with the defining identity, away from the recurrence
        for z in [20.0f64, 63.5, 500.0] {
            let want = ln_gamma(z) - ((z - 0.5) * z.ln() - z + 0.5 * f64::consts::TAU.ln());
            prec::assert_abs_diff_eq!(super::stirling_delta(z), want, epsilon = 1e-13);
        }
    }

    /// `bd0` must stay *relatively* accurate right through its double root at
    /// `x == np`, which is the whole point of the series form.
    #[test]
    fn test_bd0() {
        for (x, np, want) in [
            (10000.0, 10000.0, 0.0),
            (10000.0, 10001.0, 0.000049996666916646668333),
            (10000.0, 15000.0, 945.34891891835618022),
            (1.0, 2.0, 0.30685281944005469058),
            (600123.0, 600000.0, 0.012606638575794171215),
        ] {
            let got = super::bd0(x, np);
            if want == 0.0 {
                assert_eq!(got, 0.0);
            } else {
                prec::assert_relative_eq!(got, want, epsilon = 0.0, max_relative = 1e-14);
            }
        }
        // non-negative with a root only at x == np, and 0 * ln 0 == 0 at x == 0
        assert_eq!(super::bd0(0.0, 7.5), 7.5);
        for i in 1..200 {
            let x = 1000.0 + i as f64;
            assert!(super::bd0(x, 1000.0) > 0.0);
            assert!(super::bd0(1000.0, x) > 0.0);
        }
    }

    #[test]
    fn test_error_is_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<GammaFuncError>();
    }
}
