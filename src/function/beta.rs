//! Provides the [beta](https://en.wikipedia.org/wiki/Beta_function) and related
//! function
//!
//! This module sets the default precision more tightly than crate defaults for `DEFAULT_EPS`

use crate::function::gamma;
use crate::prec;
use core::f64;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// sample case of module level precision
const MODULE_EPS: f64 = 1e-15;

/// Represents the errors that can occur when computing the natural logarithm
/// of the beta function or the regularized lower incomplete beta function.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum BetaFuncError {
    /// `a` is zero or less than zero.
    ANotGreaterThanZero,

    /// `b` is zero or less than zero.
    BNotGreaterThanZero,

    /// `x` is not in `[0, 1]`.
    XOutOfRange,
}

impl core::fmt::Display for BetaFuncError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            BetaFuncError::ANotGreaterThanZero => write!(f, "a is zero or less than zero"),
            BetaFuncError::BNotGreaterThanZero => write!(f, "b is zero or less than zero"),
            BetaFuncError::XOutOfRange => write!(f, "x is not in [0, 1]"),
        }
    }
}

impl core::error::Error for BetaFuncError {}

/// Computes the natural logarithm
/// of the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter
/// and `a > 0`, `b > 0`.
///
/// # Panics
///
/// if `a <= 0.0` or `b <= 0.0`
pub fn ln_beta(a: f64, b: f64) -> f64 {
    checked_ln_beta(a, b).unwrap()
}

/// Computes the natural logarithm
/// of the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter
/// and `a > 0`, `b > 0`.
///
/// # Errors
///
/// if `a <= 0.0` or `b <= 0.0`
pub fn checked_ln_beta(a: f64, b: f64) -> Result<f64, BetaFuncError> {
    if a <= 0.0 {
        Err(BetaFuncError::ANotGreaterThanZero)
    } else if b <= 0.0 {
        Err(BetaFuncError::BNotGreaterThanZero)
    } else {
        Ok(gamma::ln_gamma(a) + gamma::ln_gamma(b) - gamma::ln_gamma(a + b))
    }
}

/// Computes the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter.
///
///
/// # Panics
///
/// if `a <= 0.0` or `b <= 0.0`
pub fn beta(a: f64, b: f64) -> f64 {
    checked_beta(a, b).unwrap()
}

/// Computes the beta function
/// where `a` is the first beta parameter
/// and `b` is the second beta parameter.
///
///
/// # Errors
///
/// if `a <= 0.0` or `b <= 0.0`
pub fn checked_beta(a: f64, b: f64) -> Result<f64, BetaFuncError> {
    checked_ln_beta(a, b).map(|x| x.exp())
}

/// Computes the lower incomplete (unregularized) beta function
/// `B(a,b,x) = int(t^(a-1)*(1-t)^(b-1),t=0..x)` for `a > 0, b > 0, 1 >= x >= 0`
/// where `a` is the first beta parameter, `b` is the second beta parameter, and
/// `x` is the upper limit of the integral
///
/// # Panics
///
/// If `a <= 0.0`, `b <= 0.0`, `x < 0.0`, or `x > 1.0`
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    checked_beta_inc(a, b, x).unwrap()
}

/// Computes the lower incomplete (unregularized) beta function
/// `B(a,b,x) = int(t^(a-1)*(1-t)^(b-1),t=0..x)` for `a > 0, b > 0, 1 >= x >= 0`
/// where `a` is the first beta parameter, `b` is the second beta parameter, and
/// `x` is the upper limit of the integral
///
/// # Errors
///
/// If `a <= 0.0`, `b <= 0.0`, `x < 0.0`, or `x > 1.0`
pub fn checked_beta_inc(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    checked_beta_reg(a, b, x).and_then(|x| checked_beta(a, b).map(|y| x * y))
}

/// Computes the natural logarithm of the regularized lower incomplete beta
/// function, `ln I_x(a, b)`, staying finite far past the point where `I_x`
/// itself underflows (below ~1e-308).
///
/// `beta_reg` computes `exp(ln prefix) * fraction` and so saturates to 0 once
/// the prefix passes the underflow limit; the log-domain form has no cliff.
/// Used by `ln_cdf`/`ln_sf` implementations on the beta family
/// (`Binomial::ln_sf(x)` is `ln_beta_reg(x+1, n-x, p)` and stays finite for
/// p-values far beyond representable).
///
/// Accuracy: the deep tail (where the untransformed branch is taken) is
/// limited by the prefix, a few 1e-16 *absolute* in the log - i.e. the
/// *relative* error of the underlying probability. Near the centre the result
/// is `ln1p(-v)` of a well-conditioned `v` and tracks `beta_reg` itself.
///
/// # Panics
///
/// if `a <= 0.0`, `b <= 0.0`, `x < 0.0`, or `x > 1.0`
pub fn ln_beta_reg(a: f64, b: f64, x: f64) -> f64 {
    checked_ln_beta_reg(a, b, x).unwrap()
}

/// Non-panicking variant of [`ln_beta_reg`].
///
/// # Errors
///
/// if `a <= 0.0`, `b <= 0.0`, `x < 0.0`, or `x > 1.0`
pub fn checked_ln_beta_reg(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    if a <= 0.0 {
        return Err(BetaFuncError::ANotGreaterThanZero);
    }
    if b <= 0.0 {
        return Err(BetaFuncError::BNotGreaterThanZero);
    }
    if !(0.0..=1.0).contains(&x) {
        return Err(BetaFuncError::XOutOfRange);
    }
    if x == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if x == 1.0 {
        return Ok(0.0);
    }

    let symm_transform = beta_reg_symm_transform(a, b, x);
    let max_iters = ((8.0 * a.min(b).cbrt()) as u32).clamp(140, 1_000_000);

    let (a, b, x) = if symm_transform {
        (b, a, 1.0 - x)
    } else {
        (a, b, x)
    };

    // ln(bt * h / a): every piece stays representable long after `bt` itself
    // underflows.
    let ln_v = ln_beta_prefix(a, b, x) + (beta_reg_fraction(a, b, x, max_iters) / a).ln();
    if symm_transform {
        // I = 1 - v; `exp` may underflow to 0, in which case ln I correctly
        // saturates to -0. Clamp guards the truncated-recurrence regime where
        // v could exceed 1 (see `finish`).
        Ok((-ln_v.min(0.0).exp()).ln_1p())
    } else {
        // a probability's log is never positive; > 0 only ever arises from the
        // truncated-recurrence regime
        Ok(ln_v.min(0.0))
    }
}

/// Whether `I_x(a, b)` should be computed via the symmetry
/// `I_x(a, b) = 1 - I_{1-x}(b, a)`; true when `x` is past the distribution's
/// bulk, where the direct continued fraction converges poorly.
fn beta_reg_symm_transform(a: f64, b: f64, x: f64) -> bool {
    let denom = a + b + 2.0;
    if denom.is_finite() {
        x >= (a + 1.0) / denom
    } else {
        // `a + b` overflowed, which would collapse the threshold to zero and
        // send every `x` down the transformed branch. Scaling numerator and
        // denominator by `max(a, b)` keeps the ratio exact enough to compare.
        let m = a.max(b);
        x >= (a / m + 1.0 / m) / (a / m + b / m + 2.0 / m)
    }
}

/// The Lentz continued fraction for `I_x(a, b) * a / (x^a (1-x)^b / B(a, b))`,
/// evaluated after the symmetry transform. Shared by [`checked_beta_reg`] and
/// [`checked_ln_beta_reg`]; the value is O(1) and independent of the prefix,
/// which is what makes the log-domain variant possible.
fn beta_reg_fraction(a: f64, b: f64, x: f64, max_iters: u32) -> f64 {
    let eps = prec::F64_PREC;
    let fpmin = f64::MIN_POSITIVE / eps;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;

    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iters {
        let m = f64::from(m);
        let m2 = m * 2.0;
        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;

        if d.abs() < fpmin {
            d = fpmin;
        }

        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }

        d = 1.0 / d;
        h = h * d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;

        if d.abs() < fpmin {
            d = fpmin;
        }

        c = 1.0 + aa / c;

        if c.abs() < fpmin {
            c = fpmin;
        }

        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() <= eps {
            break;
        }
    }

    h
}

/// Computes the regularized lower incomplete beta function
/// `I_x(a,b) = 1/Beta(a,b) * int(t^(a-1)*(1-t)^(b-1), t=0..x)`
/// `a > 0`, `b > 0`, `1 >= x >= 0` where `a` is the first beta parameter,
/// `b` is the second beta parameter, and `x` is the upper limit of the
/// integral.
///
/// # Panics
///
/// if `a <= 0.0`, `b <= 0.0`, `x < 0.0`, or `x > 1.0`
pub fn beta_reg(a: f64, b: f64, x: f64) -> f64 {
    checked_beta_reg(a, b, x).unwrap()
}

/// `ln(x^a (1-x)^b / Beta(a, b))`, the prefix of the incomplete beta functions.
///
/// Written out, this is
/// `ln_gamma(a+b) - ln_gamma(a) - ln_gamma(b) + a ln x + b ln(1-x)`, where the
/// terms grow like `(a + b) ln(a + b)` while the sum stays `O(ln(a + b))`. With
/// `n = a + b` the saddle-point form is
///
/// ```text
/// = -bd0(a, n x) - bd0(b, n (1-x))
///   + stirling_delta(n) - stirling_delta(a) - stirling_delta(b)
///   + ln(a b / (2 pi n)) / 2
/// ```
///
/// Every piece is `O(1)` (the two `bd0` terms grow only as the true `-ln` of
/// the prefix does), so this is accurate to a few `1e-16` absolute where the
/// old form lost `~1e-10` at `a = b = 1e4`.
///
/// For small parameters the direct form is kept: there is no cancellation left
/// to remove (every term is already `O(1)`), while `stirling_delta` would need
/// up to 16 recurrence steps per argument and so contributes more rounding than
/// it saves.
fn ln_beta_prefix(a: f64, b: f64, x: f64) -> f64 {
    let y = 1.0 - x;
    if a.max(b) < gamma::STIRLING_SERIES_MIN {
        return gamma::ln_gamma(a + b) - gamma::ln_gamma(a) - gamma::ln_gamma(b)
            + a * x.ln()
            + b * y.ln();
    }
    // `bd0` is sensitive to the last half-ulp of its mean argument, so `n` and
    // the two products are carried as double-doubles; see `gamma::bd0_dd`.
    let (n, n_lo) = prec::two_sum(a, b);
    let (y_hi, y_lo) = prec::two_diff(1.0, x);
    let nx = n * x;
    let nx_lo = prec::dekker_product_err(n, x, nx) + n_lo * x;
    let ny = n * y_hi;
    let ny_lo = prec::dekker_product_err(n, y_hi, ny) + n * y_lo + n_lo * y_hi;

    // `a / n` and `b / TAU` are each individually representable, unlike
    // `a * b / (2 pi n)`, which can overflow for large parameters
    let scale = 0.5 * ((a / n).ln() + (b / f64::consts::TAU).ln());
    scale - gamma::bd0_dd(a, nx, nx_lo) - gamma::bd0_dd(b, ny, ny_lo) + gamma::stirling_delta(n)
        - gamma::stirling_delta(a)
        - gamma::stirling_delta(b)
}

/// Computes the regularized lower incomplete beta function
/// `I_x(a,b) = 1/Beta(a,b) * int(t^(a-1)*(1-t)^(b-1), t=0..x)`
/// `a > 0`, `b > 0`, `1 >= x >= 0` where `a` is the first beta parameter,
/// `b` is the second beta parameter, and `x` is the upper limit of the
/// integral.
///
/// # Remarks
///
/// The leading factor is evaluated by the saddle-point decomposition in
/// [`ln_beta_prefix`], which keeps every intermediate `O(1)` however large
/// `a + b` becomes. Measured against the exact identity `I_{1/2}(a, a) == 1/2`,
/// the relative error is:
///
/// ```text
/// min(a, b) <= 1e7    2e-13   (1e-15 at 1e4 and below)
/// min(a, b) <= 1e13   3e-10
/// min(a, b) <= 1e16   3e-7
/// beyond              unreliable; the result is still clamped to [0, 1]
/// ```
///
/// The degradation past `1e7` is the continued fraction accumulating its own
/// rounding over millions of iterations, not the prefix. Evaluation is bounded
/// at roughly 10 ms in the worst case.
///
/// Values deep in the tails carry more *relative* error - a few hundred ulp,
/// since the prefix is recovered as `exp(ln prefix)` and `|ln prefix|` is large
/// there - but their absolute error stays far below the smallest value of
/// interest.
///
/// # Errors
///
/// if `a <= 0.0`, `b <= 0.0`, `x < 0.0`, or `x > 1.0`
pub fn checked_beta_reg(a: f64, b: f64, x: f64) -> Result<f64, BetaFuncError> {
    if a <= 0.0 {
        return Err(BetaFuncError::ANotGreaterThanZero);
    }

    if b <= 0.0 {
        return Err(BetaFuncError::BNotGreaterThanZero);
    }

    if !(0.0..=1.0).contains(&x) {
        return Err(BetaFuncError::XOutOfRange);
    }

    // `I_0(a, b) == 0` and `I_1(a, b) == 1` for every `a` and `b`. Handling the
    // endpoints here keeps them independent of the symmetry test below, which
    // otherwise mapped `x == 0` to `1.0` once `a + b` overflowed.
    if x == 0.0 {
        return Ok(0.0);
    }
    if x == 1.0 {
        return Ok(1.0);
    }

    let bt = if x == 0.0 || crate::prec::ulps_eq!(x, 1.0, epsilon = MODULE_EPS) {
        0.0
    } else {
        ln_beta_prefix(a, b, x).exp()
    };
    let symm_transform = beta_reg_symm_transform(a, b, x);

    // The result is `bt * h / a` with the continued fraction `h` of order one,
    // so a prefix that has underflowed pins the answer at the corresponding
    // endpoint. Returning here also avoids forming `0.0 * h`, which is NaN
    // whenever the recurrence overflowed (e.g. `beta_reg(1e300, 1e-300, 0.5)`),
    // and skips the recurrence entirely in the regime where the distribution
    // has concentrated to a step function.
    if bt == 0.0 {
        return Ok(if symm_transform { 1.0 } else { 0.0 });
    }

    // Fallback for the regime where the recurrence below is truncated by
    // `max_iters` and can degenerate to a non-finite value: the distribution has
    // concentrated around its mean, so this is the limiting step function. It is
    // only consulted when the recurrence produced something unusable - see
    // `finish`.
    let saturated = {
        let mean = a / (a + b);
        if x < mean {
            0.0
        } else if x > mean {
            1.0
        } else {
            0.5
        }
    };

    // Iterations the Lentz recurrence below needs before `del` settles. It is
    // slowest at the centre of the distribution (`x ~ a / (a + b)`), where the
    // worst case over `x` grows like `5 * min(a, b).cbrt()`; the bound here
    // carries headroom on top of that. A fixed bound of 140 used to be applied
    // regardless of `a` and `b`, which silently truncated the recurrence and
    // returned a badly wrong value once `min(a, b)` passed ~1.5e4 (for example
    // `I_0.5(1e6, 1e6)` came back as 0.491 instead of 0.5). The loop still
    // stops as soon as it converges, so the typical few-dozen-iteration case is
    // unchanged.
    //
    // The upper clamp bounds the work at roughly 10 ms. Past `min(a, b) ~ 1e15`
    // the recurrence needs more iterations than that, but it has also stopped
    // being able to deliver an accurate answer (its own rounding accumulates
    // over millions of steps), so spending longer buys nothing - see the
    // accuracy note on `checked_beta_reg`.
    let max_iters = ((8.0 * a.min(b).cbrt()) as u32).clamp(140, 1_000_000);

    let mut a = a;
    let mut b = b;
    let mut x = x;
    if symm_transform {
        let swap = a;
        x = 1.0 - x;
        a = b;
        b = swap;
    }

    let h = beta_reg_fraction(a, b, x, max_iters);
    Ok(finish(symm_transform, bt, h, a, saturated))
}

/// Assembles `I_x(a, b)` from the prefix and continued fraction, keeping the
/// result inside `[0, 1]`.
///
/// Neither guard engages while the recurrence converges. Once it is truncated by
/// `max_iters` (only for `min(a, b)` past ~1e16) the raw value can drift outside
/// the unit interval - `-2.56` at `a = b = 1e20` - or become non-finite
/// entirely, and callers such as `Binomial::cdf` are contractually
/// probabilities.
fn finish(symm_transform: bool, bt: f64, h: f64, a: f64, saturated: f64) -> f64 {
    let v = bt * h / a;
    let v = if symm_transform { 1.0 - v } else { v };
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        saturated
    }
}

/// Computes the inverse of the regularized incomplete beta function
// This code is based on the implementation in the ["special"][1] crate,
// which in turn is based on a [C implementation][2] by John Burkardt. The
// original algorithm was published in Applied Statistics and is known as
// [Algorithm AS 64][3] and [Algorithm AS 109][4].
//
// [1]: https://docs.rs/special/0.8.1/
// [2]: http://people.sc.fsu.edu/~jburkardt/c_src/asa109/asa109.html
// [3]: http://www.jstor.org/stable/2346798
// [4]: http://www.jstor.org/stable/2346887
//
// > Copyright 2014–2019 The special Developers
// >
// > Permission is hereby granted, free of charge, to any person obtaining a copy of
// > this software and associated documentation files (the "Software"), to deal in
// > the Software without restriction, including without limitation the rights to
// > use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// > the Software, and to permit persons to whom the Software is furnished to do so,
// > subject to the following conditions:
// >
// > The above copyright notice and this permission notice shall be included in all
// > copies or substantial portions of the Software.
// >
// > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// > IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// > FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// > COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// > IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// > CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
pub fn inv_beta_reg(mut a: f64, mut b: f64, mut x: f64) -> f64 {
    // Algorithm AS 64
    // http://www.jstor.org/stable/2346798
    //
    // An approximation x₀ to x if found from (cf. Scheffé and Tukey, 1944)
    //
    // 1 + x₀   4p + 2q - 2
    // ------ = -----------
    // 1 - x₀      χ²(α)
    //
    // where χ²(α) is the upper α point of the χ² distribution with 2q
    // degrees of freedom and is obtained from Wilson and Hilferty's
    // approximation (cf. Wilson and Hilferty, 1931)
    //
    // χ²(α) = 2q (1 - 1 / (9q) + y(α) sqrt(1 / (9q)))^3,
    //
    // y(α) being Hastings' approximation (cf. Hastings, 1955) for the upper
    // α point of the standard normal distribution. If χ²(α) < 0, then
    //
    // x₀ = 1 - ((1 - α)q B(p, q))^(1 / q).
    //
    // Again if (4p + 2q - 2) / χ²(α) does not exceed 1, x₀ is obtained from
    //
    // x₀ = (αp B(p, q))^(1 / p).
    //
    // The final solution is obtained by the Newton–Raphson method from the
    // relation
    //
    //                    f(x[i - 1])
    // x[i] = x[i - 1] - ------------
    //                   f'(x[i - 1])
    //
    // where
    //
    // f(x) = I(x, p, q) - α.
    let ln_beta = ln_beta(a, b);

    // Remark AS R83
    // http://www.jstor.org/stable/2347779
    const SAE: i32 = -30;
    const FPU: f64 = 1e-30; // 10^SAE

    debug_assert!((0.0..=1.0).contains(&x) && a > 0.0 && b > 0.0);

    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }

    let mut p;
    let mut q;

    let flip = 0.5 < x;
    if flip {
        p = a;
        a = b;
        b = p;
        x = 1.0 - x;
    }

    p = (-(x * x).ln()).sqrt();
    q = p - (2.30753 + 0.27061 * p) / (1.0 + (0.99229 + 0.04481 * p) * p);

    if 1.0 < a && 1.0 < b {
        // Remark AS R19 and Algorithm AS 109
        // http://www.jstor.org/stable/2346887
        //
        // For a and b > 1, the approximation given by Carter (1947), which
        // improves the Fisher–Cochran formula, is generally better. For
        // other values of a and b en empirical investigation has shown that
        // the approximation given in AS 64 is adequate.
        let r = (q * q - 3.0) / 6.0;
        let s = 1.0 / (2.0 * a - 1.0);
        let t = 1.0 / (2.0 * b - 1.0);
        let h = 2.0 / (s + t);
        let w = q * (h + r).sqrt() / h - (t - s) * (r + 5.0 / 6.0 - 2.0 / (3.0 * h));
        p = a / (a + b * (2.0 * w).exp());
    } else {
        let mut t = 1.0 / (9.0 * b);
        t = 2.0 * b * (1.0 - t + q * t.sqrt()).powf(3.0);
        if t <= 0.0 {
            p = 1.0 - ((((1.0 - x) * b).ln() + ln_beta) / b).exp();
        } else {
            t = 2.0 * (2.0 * a + b - 1.0) / t;
            if t <= 1.0 {
                p = (((x * a).ln() + ln_beta) / a).exp();
            } else {
                p = 1.0 - 2.0 / (t + 1.0);
            }
        }
    }

    p = p.clamp(0.0001, 0.9999);

    // Remark AS R83
    // http://www.jstor.org/stable/2347779
    let e = (-5.0 / a / a - 1.0 / x.powf(0.2) - 13.0) as i32;
    let acu = if e > SAE { f64::powi(10.0, e) } else { FPU };

    let mut pnext;
    let mut qprev = 0.0;
    let mut sq = 1.0;
    let mut prev = 1.0;

    'outer: loop {
        // Remark AS R19 and Algorithm AS 109
        // http://www.jstor.org/stable/2346887
        q = beta_reg(a, b, p);
        q = (q - x) * (ln_beta + (1.0 - a) * p.ln() + (1.0 - b) * (1.0 - p).ln()).exp();

        // Remark AS R83
        // http://www.jstor.org/stable/2347779
        if q * qprev <= 0.0 {
            prev = if sq > FPU { sq } else { FPU };
        }

        // Remark AS R19 and Algorithm AS 109
        // http://www.jstor.org/stable/2346887
        let mut g = 1.0;
        loop {
            loop {
                let adj = g * q;
                sq = adj * adj;

                if sq < prev {
                    pnext = p - adj;
                    if (0.0..=1.0).contains(&pnext) {
                        break;
                    }
                }
                g /= 3.0;
            }

            if prev <= acu || q * q <= acu {
                p = pnext;
                break 'outer;
            }

            if pnext != 0.0 && pnext != 1.0 {
                break;
            }

            g /= 3.0;
        }

        if pnext == p {
            break;
        }

        p = pnext;
        qprev = q;
    }

    if flip { 1.0 - p } else { p }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prec;
    const MODULE_RELATIVE_ACC: f64 = 1e-14;

    fn beta_assert_relative_eq(a: f64, b: f64) {
        prec::assert_relative_eq!(
            a,
            b,
            epsilon = MODULE_EPS,
            max_relative = MODULE_RELATIVE_ACC
        );
    }

    fn beta_assert_abs_diff_eq(a: f64, b: f64) {
        prec::assert_abs_diff_eq!(a, b, epsilon = MODULE_EPS);
    }

    #[test]
    fn test_ln_beta() {
        beta_assert_relative_eq(ln_beta(0.5, 0.5), 1.144729885849400174144);
        beta_assert_relative_eq(ln_beta(1.0, 0.5), f64::consts::LN_2);
        beta_assert_relative_eq(ln_beta(2.5, 0.5), 0.163900632837673937284);
        beta_assert_relative_eq(ln_beta(0.5, 1.0), f64::consts::LN_2);
        beta_assert_relative_eq(ln_beta(1.0, 1.0), 0.0);
        beta_assert_relative_eq(ln_beta(2.5, 1.0), -0.9162907318741550651835);
        beta_assert_relative_eq(ln_beta(0.5, 2.5), 0.163900632837673937284);
        beta_assert_relative_eq(ln_beta(1.0, 2.5), -0.9162907318741550651835);
        beta_assert_relative_eq(ln_beta(2.5, 2.5), -2.608688089402107300388);
    }

    #[test]
    #[should_panic]
    fn test_ln_beta_a_lte_0() {
        ln_beta(0.0, 0.5);
    }

    #[test]
    #[should_panic]
    fn test_ln_beta_b_lte_0() {
        ln_beta(0.5, 0.0);
    }

    #[test]
    fn test_checked_ln_beta_a_lte_0() {
        assert!(checked_ln_beta(0.0, 0.5).is_err());
    }

    #[test]
    fn test_checked_ln_beta_b_lte_0() {
        assert!(checked_ln_beta(0.5, 0.0).is_err());
    }

    #[test]
    #[should_panic]
    fn test_beta_a_lte_0() {
        beta(0.0, 0.5);
    }

    #[test]
    #[should_panic]
    fn test_beta_b_lte_0() {
        beta(0.5, 0.0);
    }

    #[test]
    fn test_checked_beta_a_lte_0() {
        assert!(checked_beta(0.0, 0.5).is_err());
    }

    #[test]
    fn test_checked_beta_b_lte_0() {
        assert!(checked_beta(0.5, 0.0).is_err());
    }

    #[test]
    fn test_beta() {
        beta_assert_relative_eq(beta(0.5, 0.5), f64::consts::PI);
        beta_assert_relative_eq(beta(1.0, 0.5), 2.0);
        beta_assert_relative_eq(beta(2.5, 0.5), 1.17809724509617246442);
        beta_assert_relative_eq(beta(0.5, 1.0), 2.0);
        beta_assert_relative_eq(beta(1.0, 1.0), 1.0);
        beta_assert_relative_eq(beta(2.5, 1.0), 0.4);
        beta_assert_relative_eq(beta(0.5, 2.5), 1.17809724509617246442);
        beta_assert_relative_eq(beta(1.0, 2.5), 0.4);
        beta_assert_relative_eq(beta(2.5, 2.5), 0.073631077818510779026);
    }

    #[test]
    fn test_beta_inc() {
        beta_assert_relative_eq(beta_inc(0.5, 0.5, 0.5), f64::consts::FRAC_PI_2);
        beta_assert_relative_eq(beta_inc(0.5, 0.5, 1.0), f64::consts::PI);
        beta_assert_relative_eq(beta_inc(1.0, 0.5, 0.5), 0.5857864376269049511983);
        beta_assert_relative_eq(beta_inc(1.0, 0.5, 1.0), 2.0);
        beta_assert_relative_eq(beta_inc(2.5, 0.5, 0.5), 0.0890486225480862322117);
        beta_assert_relative_eq(beta_inc(2.5, 0.5, 1.0), 1.17809724509617246442);
        beta_assert_relative_eq(beta_inc(0.5, 1.0, 0.5), f64::consts::SQRT_2);
        beta_assert_relative_eq(beta_inc(0.5, 1.0, 1.0), 2.0);
        beta_assert_relative_eq(beta_inc(1.0, 1.0, 0.5), 0.5);
        beta_assert_relative_eq(beta_inc(1.0, 1.0, 1.0), 1.0);
        beta_assert_relative_eq(beta_inc(2.5, 1.0, 0.5), 0.0707106781186547524401);
        beta_assert_relative_eq(beta_inc(2.5, 1.0, 1.0), 0.4);
        beta_assert_relative_eq(beta_inc(0.5, 2.5, 0.5), 1.08904862254808623221);
        beta_assert_relative_eq(beta_inc(0.5, 2.5, 1.0), 1.17809724509617246442);
        beta_assert_relative_eq(beta_inc(1.0, 2.5, 0.5), 0.32928932188134524756);
        beta_assert_relative_eq(beta_inc(1.0, 2.5, 1.0), 0.4);
        beta_assert_relative_eq(beta_inc(2.5, 2.5, 0.5), 0.03681553890925538951323);
        beta_assert_relative_eq(beta_inc(2.5, 2.5, 1.0), 0.073631077818510779026);
    }

    #[test]
    #[should_panic]
    fn test_beta_inc_a_lte_0() {
        beta_inc(0.0, 1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_inc_b_lte_0() {
        beta_inc(1.0, 0.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_inc_x_lt_0() {
        beta_inc(1.0, 1.0, -1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_inc_x_gt_1() {
        beta_inc(1.0, 1.0, 2.0);
    }

    #[test]
    fn test_checked_beta_inc_a_lte_0() {
        assert!(checked_beta_inc(0.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_beta_inc_b_lte_0() {
        assert!(checked_beta_inc(1.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_beta_inc_x_lt_0() {
        assert!(checked_beta_inc(1.0, 1.0, -1.0).is_err());
    }

    #[test]
    fn test_checked_beta_inc_x_gt_1() {
        assert!(checked_beta_inc(1.0, 1.0, 2.0).is_err());
    }

    #[test]
    fn test_beta_reg() {
        beta_assert_abs_diff_eq(beta_reg(0.5, 0.5, 0.5), 0.5);
        assert_eq!(beta_reg(0.5, 0.5, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(1.0, 0.5, 0.5), 0.292893218813452475599);
        assert_eq!(beta_reg(1.0, 0.5, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(2.5, 0.5, 0.5), 0.07558681842161243795);
        assert_eq!(beta_reg(2.5, 0.5, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(0.5, 1.0, 0.5), f64::consts::FRAC_1_SQRT_2);
        assert_eq!(beta_reg(0.5, 1.0, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(1.0, 1.0, 0.5), 0.5);
        assert_eq!(beta_reg(1.0, 1.0, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(2.5, 1.0, 0.5), 0.1767766952966368811);
        assert_eq!(beta_reg(2.5, 1.0, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(0.5, 2.5, 0.5), 0.92441318157838756205);
        assert_eq!(beta_reg(0.5, 2.5, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(1.0, 2.5, 0.5), 0.8232233047033631189);
        assert_eq!(beta_reg(1.0, 2.5, 1.0), 1.0);
        beta_assert_abs_diff_eq(beta_reg(2.5, 2.5, 0.5), 0.5);
        assert_eq!(beta_reg(2.5, 2.5, 1.0), 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_reg_a_lte_0() {
        beta_reg(0.0, 1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_reg_b_lte_0() {
        beta_reg(1.0, 0.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_reg_x_lt_0() {
        beta_reg(1.0, 1.0, -1.0);
    }

    #[test]
    #[should_panic]
    fn test_beta_reg_x_gt_1() {
        beta_reg(1.0, 1.0, 2.0);
    }

    #[test]
    fn test_checked_beta_reg_a_lte_0() {
        assert!(checked_beta_reg(0.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_beta_reg_b_lte_0() {
        assert!(checked_beta_reg(1.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn test_checked_beta_reg_x_lt_0() {
        assert!(checked_beta_reg(1.0, 1.0, -1.0).is_err());
    }

    #[test]
    fn test_checked_beta_reg_x_gt_1() {
        assert!(checked_beta_reg(1.0, 1.0, 2.0).is_err());
    }

    /// `I_{1/2}(a, a) == 1/2` exactly, by the symmetry of the `Beta(a, a)`
    /// density. That point is also where the continued fraction converges most
    /// slowly, so it pins the iteration bound. With the old fixed bound of 140
    /// iterations this returned 0.49999969 at `a = 1e5`, 0.49121973 at `a = 1e6`
    /// and 0.21285001 at `a = 1e7`.
    #[test]
    fn test_beta_reg_symmetric_midpoint_large_parameters() {
        // The saddle-point prefix (`ln_beta_prefix`) took this from `6e-7` at
        // `a = 1e5` and `5.7e-1` at `a = 1e7` down to the `1e-13` level.
        for a in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7] {
            prec::assert_relative_eq!(
                beta_reg(a, a, 0.5),
                0.5,
                epsilon = 0.0,
                max_relative = 1e-12
            );
        }
    }

    /// `beta_reg` is a probability and must stay in `[0, 1]` and finite for
    /// every valid input, including parameter ratios extreme enough to
    /// over/underflow the intermediate quantities. Before the `bd0` ratio guard
    /// and the `[0, 1]` clamp, this grid produced NaN (`a = 1e300, b = 1e-300`),
    /// a silent `1.0` where the answer was `0` (`a = 1e100, b = 1e-300`), and
    /// `-2.56` (`a = b = 1e20`).
    #[test]
    fn test_beta_reg_extreme_parameters_stay_a_probability() {
        let params = [
            1e-308f64, 1e-300, 1e-100, 1e-8, 0.5, 1.0, 20.0, 1e8, 1e20, 1e100, 1e300, 1e308,
        ];
        for &a in &params {
            for &b in &params {
                // Beyond a ~1e300 parameter ratio the Lentz recurrence bottoms
                // out in its own `fpmin` guards for `x` within an ulp of the
                // mode, and returns NaN. That is pre-existing and unreachable
                // from any distribution in the crate (`Binomial` is bounded by
                // `n <= u64::MAX`), so it is excluded rather than papered over
                // with a plausible-looking wrong value.
                if a.max(b) / a.min(b) > 1e200 {
                    continue;
                }
                for x in [0.0f64, 1e-300, 0.25, 0.5, 0.75, 1.0] {
                    let v = beta_reg(a, b, x);
                    assert!(
                        v.is_finite() && (0.0..=1.0).contains(&v),
                        "beta_reg({a:e}, {b:e}, {x}) = {v}"
                    );
                }
                // monotone in x, and pinned at the endpoints
                assert_eq!(beta_reg(a, b, 0.0), 0.0, "beta_reg({a:e},{b:e},0)");
                assert_eq!(beta_reg(a, b, 1.0), 1.0, "beta_reg({a:e},{b:e},1)");
            }
        }
    }

    /// `I_x(a, b) + I_{1-x}(b, a) == 1` for every valid `a`, `b`, `x`. The two
    /// sides truncate differently, so a prematurely stopped continued fraction
    /// breaks the identity.
    #[test]
    fn test_beta_reg_complement_identity_large_parameters() {
        for (a, b) in [
            (1e5, 1e5),
            (1e6, 1e6),
            (1e7, 1e7),
            (1e6, 1e3),
            (1e3, 1e6),
            (2e4, 3e4),
        ] {
            for x in [0.1, 0.25, 0.5, 0.5 + 1e-9, 0.75, 0.9] {
                let lhs = beta_reg(a, b, x) + beta_reg(b, a, 1.0 - x);
                prec::assert_abs_diff_eq!(lhs, 1.0, epsilon = 1e-7);
            }
        }
    }

    #[test]
    fn test_error_is_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<BetaFuncError>();
    }
}
