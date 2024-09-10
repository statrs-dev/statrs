//! Provides the [gamma](https://en.wikipedia.org/wiki/Gamma_function) and
//! related functions

use crate::consts;
use crate::prec;
use std::f64;

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

impl std::fmt::Display for GammaFuncError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            GammaFuncError::AInvalid => write!(f, "a is infinite, zero or less than zero"),
            GammaFuncError::XInvalid => write!(f, "x is infinite, zero or less than zero"),
        }
    }
}

impl std::error::Error for GammaFuncError {}

/// Auxiliary variable when evaluating the `gamma_ln` function
const GAMMA_R: f64 = 10.900511;

/// Polynomial coefficients for approximating the `gamma_ln` function
const GAMMA_DK: &[f64] = &[
    2.48574089138753565546e-5,
    1.05142378581721974210,
    -3.45687097222016235469,
    4.51227709466894823700,
    -2.98285225323576655721,
    1.05639711577126713077,
    -1.95428773191645869583e-1,
    1.70970543404441224307e-2,
    -5.71926117404305781283e-4,
    4.63399473359905636708e-6,
    -2.71994908488607703910e-9,
];

/// Computes the logarithm of the gamma function
/// with an accuracy of 16 floating point digits.
/// The implementation is derived from
/// "An Analysis of the Lanczos Gamma Approximation",
/// Glendon Ralph Pugh, 2004 p. 116
pub fn ln_gamma(x: f64) -> f64 {
    if x < 0.5 {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| s + t.1 / (t.0 as f64 - x));

        consts::LN_PI
            - (f64::consts::PI * x).sin().ln()
            - s.ln()
            - consts::LN_2_SQRT_E_OVER_PI
            - (0.5 - x) * ((0.5 - x + GAMMA_R) / f64::consts::E).ln()
    } else {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| s + t.1 / (x + t.0 as f64 - 1.0));

        s.ln()
            + consts::LN_2_SQRT_E_OVER_PI
            + (x - 0.5) * ((x - 0.5 + GAMMA_R) / f64::consts::E).ln()
    }
}

/// Computes the gamma function with an accuracy
/// of 16 floating point digits. The implementation
/// is derived from "An Analysis of the Lanczos Gamma Approximation",
/// Glendon Ralph Pugh, 2004 p. 116
pub fn gamma(x: f64) -> f64 {
    if x < 0.5 {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| s + t.1 / (t.0 as f64 - x));

        f64::consts::PI
            / ((f64::consts::PI * x).sin()
                * s
                * consts::TWO_SQRT_E_OVER_PI
                * ((0.5 - x + GAMMA_R) / f64::consts::E).powf(0.5 - x))
    } else {
        let s = GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(GAMMA_DK[0], |s, t| s + t.1 / (x + t.0 as f64 - 1.0));

        s * consts::TWO_SQRT_E_OVER_PI * ((x - 0.5 + GAMMA_R) / f64::consts::E).powf(x - 0.5)
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

    if prec::almost_eq(a, 0.0, prec::DEFAULT_F64_ACC) {
        return Ok(1.0);
    }
    if prec::almost_eq(x, 0.0, prec::DEFAULT_F64_ACC) {
        return Ok(0.0);
    }

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
    if x <= 0.0 && ulps_eq!(x.floor(), x) {
        return f64::NEG_INFINITY;
    }
    if x < 0.0 {
        return digamma(1.0 - x) + f64::consts::PI / (-f64::consts::PI * x).tan();
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
    if x == 0.0 {
        0.0
    } else {
        x.signum()
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use super::*;

    use std::f64::consts;

    #[test]
    fn test_gamma() {
        assert!(super::gamma(f64::NAN).is_nan());
        assert_almost_eq!(super::gamma(1.000001e-35), 9.9999900000099999900000099999899999522784235098567139293e+34, 1e20);
        assert_almost_eq!(super::gamma(1.000001e-10), 9.99998999943278432519738283781280989934496494539074049002e+9, 1e-5);
        assert_almost_eq!(super::gamma(1.000001e-5), 99999.32279432557746387178953902739303931424932435387031653234, 1e-10);
        assert_almost_eq!(super::gamma(1.000001e-2), 99.43248512896257405886134437203369035261893114349805309870831, 1e-13);
        assert_almost_eq!(super::gamma(-4.8), -0.06242336135475955314181664931547009890495158793105543559676, 1e-13);
        assert_almost_eq!(super::gamma(-1.5), 2.363271801207354703064223311121526910396732608163182837618410, 1e-13);
        assert_almost_eq!(super::gamma(-0.5), -3.54490770181103205459633496668229036559509891224477425642761, 1e-13);
        assert_almost_eq!(super::gamma(1.0e-5 + 1.0e-16), 99999.42279322556767360213300482199406241771308740302819426480, 1e-9);
        assert_almost_eq!(super::gamma(0.1), 9.513507698668731836292487177265402192550578626088377343050000, 1e-14);
        assert_eq!(super::gamma(1.0 - 1.0e-14), 1.000000000000005772156649015427511664653698987042926067639529);
        assert_almost_eq!(super::gamma(1.0), 1.0, 1e-15);
        assert_almost_eq!(super::gamma(1.0 + 1.0e-14), 0.99999999999999422784335098477029953441189552403615306268023, 1e-15);
        assert_almost_eq!(super::gamma(1.5), 0.886226925452758013649083741670572591398774728061193564106903, 1e-14);
        assert_almost_eq!(super::gamma(consts::PI/2.0), 0.890560890381539328010659635359121005933541962884758999762766, 1e-15);
        assert_eq!(super::gamma(2.0), 1.0);
        assert_almost_eq!(super::gamma(2.5), 1.329340388179137020473625612505858887098162092091790346160355, 1e-13);
        assert_almost_eq!(super::gamma(3.0), 2.0, 1e-14);
        assert_almost_eq!(super::gamma(consts::PI), 2.288037795340032417959588909060233922889688153356222441199380, 1e-13);
        assert_almost_eq!(super::gamma(3.5), 3.323350970447842551184064031264647217745405230229475865400889, 1e-14);
        assert_almost_eq!(super::gamma(4.0), 6.0, 1e-13);
        assert_almost_eq!(super::gamma(4.5), 11.63172839656744892914422410942626526210891830580316552890311, 1e-12);
        assert_almost_eq!(super::gamma(5.0 - 1.0e-14), 23.99999999999963853175957637087420162718107213574617032780374, 1e-13);
        assert_almost_eq!(super::gamma(5.0), 24.0, 1e-12);
        assert_almost_eq!(super::gamma(5.0 + 1.0e-14), 24.00000000000036146824042363510111050137786752408660789873592, 1e-12);
        assert_almost_eq!(super::gamma(5.5), 52.34277778455352018114900849241819367949013237611424488006401, 1e-12);
        assert_almost_eq!(super::gamma(10.1), 454760.7514415859508673358368319076190405047458218916492282448, 1e-7);
        assert_almost_eq!(super::gamma(150.0 + 1.0e-12), 3.8089226376496421386707466577615064443807882167327097140e+260, 1e248);
    }

    #[test]
    fn test_ln_gamma() {
        assert!(super::ln_gamma(f64::NAN).is_nan());
        assert_eq!(super::ln_gamma(1.000001e-35), 80.59047725479209894029636783061921392709972287131139201585211);
        assert_almost_eq!(super::ln_gamma(1.000001e-10), 23.02584992988323521564308637407936081168344192865285883337793, 1e-14);
        assert_almost_eq!(super::ln_gamma(1.000001e-5), 11.51291869289055371493077240324332039045238086972508869965363, 1e-14);
        assert_eq!(super::ln_gamma(1.000001e-2), 4.599478872433667224554543378460164306444416156144779542513592);
        assert_almost_eq!(super::ln_gamma(0.1), 2.252712651734205959869701646368495118615627222294953765041739, 1e-14);
        assert_almost_eq!(super::ln_gamma(1.0 - 1.0e-14), 5.772156649015410852768463312546533565566459794933360600e-15, 1e-15);
        assert_almost_eq!(super::ln_gamma(1.0), 0.0, 1e-15);
        assert_almost_eq!(super::ln_gamma(1.0 + 1.0e-14), -5.77215664901524635936177848990288632404978978079827014e-15, 1e-15);
        assert_almost_eq!(super::ln_gamma(1.5), -0.12078223763524522234551844578164721225185272790259946836386, 1e-14);
        assert_almost_eq!(super::ln_gamma(consts::PI/2.0), -0.11590380084550241329912089415904874214542604767006895, 1e-14);
        assert_eq!(super::ln_gamma(2.0), 0.0);
        assert_almost_eq!(super::ln_gamma(2.5), 0.284682870472919159632494669682701924320137695559894729250145, 1e-13);
        assert_almost_eq!(super::ln_gamma(3.0), 0.693147180559945309417232121458176568075500134360255254120680, 1e-14);
        assert_almost_eq!(super::ln_gamma(consts::PI), 0.82769459232343710152957855845235995115350173412073715, 1e-13);
        assert_almost_eq!(super::ln_gamma(3.5), 1.200973602347074224816021881450712995770238915468157197042113, 1e-14);
        assert_almost_eq!(super::ln_gamma(4.0), 1.791759469228055000812477358380702272722990692183004705855374, 1e-14);
        assert_almost_eq!(super::ln_gamma(4.5), 2.453736570842442220504142503435716157331823510689763131380823, 1e-13);
        assert_almost_eq!(super::ln_gamma(5.0 - 1.0e-14), 3.178053830347930558470257283303394288448414225994179545985931, 1e-14);
        assert_almost_eq!(super::ln_gamma(5.0), 3.178053830347945619646941601297055408873990960903515214096734, 1e-14);
        assert_almost_eq!(super::ln_gamma(5.0 + 1.0e-14), 3.178053830347960680823625919312848824873279228348981287761046, 1e-13);
        assert_almost_eq!(super::ln_gamma(5.5), 3.957813967618716293877400855822590998551304491975006780729532, 1e-14);
        assert_almost_eq!(super::ln_gamma(10.1), 13.02752673863323795851370097886835481188051062306253294740504, 1e-14);
        assert_almost_eq!(super::ln_gamma(150.0 + 1.0e-12), 600.0094705553324354062157737572509902987070089159051628001813, 1e-12);
        assert_almost_eq!(super::ln_gamma(1.001e+7), 1.51342135323817913130119829455205139905331697084416059779e+8, 1e-13);
    }

    #[test]
    fn test_gamma_lr() {
        assert!(super::gamma_lr(f64::NAN, f64::NAN).is_nan());
        assert_almost_eq!(super::gamma_lr(0.1, 1.0), 0.97587265627367222115949155252812057714751052498477013, 1e-14);
        assert_eq!(super::gamma_lr(0.1, 2.0), 0.99432617602018847196075251078067514034772764693462125);
        assert_eq!(super::gamma_lr(0.1, 8.0), 0.99999507519205198048686442150578226823401842046310854);
        assert_almost_eq!(super::gamma_lr(1.5, 1.0), 0.42759329552912016600095238564127189392715996802703368, 1e-13);
        assert_almost_eq!(super::gamma_lr(1.5, 2.0), 0.73853587005088937779717792402407879809718939080920993, 1e-15);
        assert_eq!(super::gamma_lr(1.5, 8.0), 0.99886601571021467734329986257903021041757398191304284);
        assert_almost_eq!(super::gamma_lr(2.5, 1.0), 0.15085496391539036377410688601371365034788861473418704, 1e-13);
        assert_almost_eq!(super::gamma_lr(2.5, 2.0), 0.45058404864721976739416885516693969548484517509263197, 1e-14);
        assert_almost_eq!(super::gamma_lr(2.5, 8.0), 0.99315592607757956900093935107222761316136944145439676, 1e-15);
        assert_almost_eq!(super::gamma_lr(5.5, 1.0), 0.0015041182825838038421585211353488839717739161316985392, 1e-15);
        assert_almost_eq!(super::gamma_lr(5.5, 2.0), 0.030082976121226050615171484772387355162056796585883967, 1e-14);
        assert_almost_eq!(super::gamma_lr(5.5, 8.0), 0.85886911973294184646060071855669224657735916933487681, 1e-14);
        assert_almost_eq!(super::gamma_lr(100.0, 0.5), 0.0, 1e-188);
        assert_almost_eq!(super::gamma_lr(100.0, 1.5), 0.0, 1e-141);
        assert_almost_eq!(super::gamma_lr(100.0, 90.0), 0.1582209891864301681049696996709105316998233457433473, 1e-13);
        assert_almost_eq!(super::gamma_lr(100.0, 100.0), 0.5132987982791486648573142565640291634709251499279450, 1e-13);
        assert_almost_eq!(super::gamma_lr(100.0, 110.0), 0.8417213299399129061982996209829688531933500308658222, 1e-13);
        assert_almost_eq!(super::gamma_lr(100.0, 200.0), 1.0, 1e-14);
        assert_eq!(super::gamma_lr(500.0, 0.5), 0.0);
        assert_eq!(super::gamma_lr(500.0, 1.5), 0.0);
        assert_almost_eq!(super::gamma_lr(500.0, 200.0), 0.0, 1e-70);
        assert_almost_eq!(super::gamma_lr(500.0, 450.0), 0.0107172380912897415573958770655204965434869949241480, 1e-14);
        assert_almost_eq!(super::gamma_lr(500.0, 500.0), 0.5059471461707603580470479574412058032802735425634263, 1e-13);
        assert_almost_eq!(super::gamma_lr(500.0, 550.0), 0.9853855918737048059548470006900844665580616318702748, 1e-14);
        assert_almost_eq!(super::gamma_lr(500.0, 700.0), 1.0, 1e-15);
        assert_eq!(super::gamma_lr(1000.0, 10000.0), 1.0);
        assert_eq!(super::gamma_lr(1e+50, 1e+48), 0.0);
        assert_eq!(super::gamma_lr(1e+50, 1e+52), 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_lr_a_lower_bound() {
        super::gamma_lr(-1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_lr_a_upper_bound() {
        super::gamma_lr(f64::INFINITY, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_lr_x_lower_bound() {
        super::gamma_lr(1.0, -1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_lr_x_upper_bound() {
        super::gamma_lr(1.0, f64::INFINITY);
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
        assert!(super::gamma_li(f64::NAN, f64::NAN).is_nan());
        assert_almost_eq!(super::gamma_li(0.1, 1.0), 9.2839720283798852469443229940217320532607158711056334, 1e-14);
        assert_almost_eq!(super::gamma_li(0.1, 2.0), 9.4595297305559030536119885480983751098528458886962883, 1e-14);
        assert_almost_eq!(super::gamma_li(0.1, 8.0), 9.5134608464704033372127589212547718314010339263844976, 1e-13);
        assert_almost_eq!(super::gamma_li(1.5, 1.0), 0.37894469164098470380394366597039213790868855578083847, 1e-15);
        assert_almost_eq!(super::gamma_li(1.5, 2.0), 0.65451037345177732033319477475056262302270310457635612, 1e-14);
        assert_almost_eq!(super::gamma_li(1.5, 8.0), 0.88522195804210983776635107858848816480298923071075222, 1e-13);
        assert_almost_eq!(super::gamma_li(2.5, 1.0), 0.20053759629003473411039172879412733941722170263949, 1e-16);
        assert_almost_eq!(super::gamma_li(2.5, 2.0), 0.59897957413602228465664030130712917348327070206302442, 1e-15);
        assert_almost_eq!(super::gamma_li(2.5, 8.0), 1.3202422842943799358198434659248530581833764879301293, 1e-14);
        assert_almost_eq!(super::gamma_li(5.5, 1.0), 0.078729729026968321691794205337720556329618007004848672, 1e-16);
        assert_almost_eq!(super::gamma_li(5.5, 2.0), 1.5746265342113649473739798668921124454837064926448459, 1e-15);
        assert_almost_eq!(super::gamma_li(5.5, 8.0), 44.955595480196465884619737757794960132425035578313584, 1e-12);
    }

    #[test]
    #[should_panic]
    fn test_gamma_li_a_lower_bound() {
        super::gamma_li(-1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_li_a_upper_bound() {
        super::gamma_li(f64::INFINITY, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_li_x_lower_bound() {
        super::gamma_li(1.0, -1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_li_x_upper_bound() {
        super::gamma_li(1.0, f64::INFINITY);
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
        assert!(super::gamma_ur(f64::NAN, f64::NAN).is_nan());
        assert_almost_eq!(super::gamma_ur(0.1, 1.0), 0.0241273437263277773829694356333550393309597428392044, 1e-13);
        assert_almost_eq!(super::gamma_ur(0.1, 2.0), 0.0056738239798115280392474892193248596522723530653781, 1e-13);
        assert_almost_eq!(super::gamma_ur(0.1, 8.0), 0.0000049248079480195131355784942177317659815795368919702, 1e-13);
        assert_almost_eq!(super::gamma_ur(1.5, 1.0), 0.57240670447087983399904761435872810607284003197297, 1e-13);
        assert_almost_eq!(super::gamma_ur(1.5, 2.0), 0.26146412994911062220282207597592120190281060919079, 1e-13);
        assert_almost_eq!(super::gamma_ur(1.5, 8.0), 0.0011339842897853226567001374209697895824260180869567, 1e-13);
        assert_almost_eq!(super::gamma_ur(2.5, 1.0), 0.84914503608460963622589311398628634965211138526581, 1e-13);
        assert_almost_eq!(super::gamma_ur(2.5, 2.0), 0.54941595135278023260583114483306030451515482490737, 1e-13);
        assert_almost_eq!(super::gamma_ur(2.5, 8.0), 0.0068440739224204309990606489277723868386305585456026, 1e-13);
        assert_almost_eq!(super::gamma_ur(5.5, 1.0), 0.9984958817174161961578414788646511160282260838683, 1e-13);
        assert_almost_eq!(super::gamma_ur(5.5, 2.0), 0.96991702387877394938482851522761264483794320341412, 1e-13);
        assert_almost_eq!(super::gamma_ur(5.5, 8.0), 0.14113088026705815353939928144330775342264083066512, 1e-13);
        assert_almost_eq!(super::gamma_ur(100.0, 0.5), 1.0, 1e-14);
        assert_almost_eq!(super::gamma_ur(100.0, 1.5), 1.0, 1e-14);
        assert_almost_eq!(super::gamma_ur(100.0, 90.0), 0.8417790108135698318950303003290894683001766542566526, 1e-12);
        assert_almost_eq!(super::gamma_ur(100.0, 100.0), 0.4867012017208513351426857434359708365290748500720549, 1e-12);
        assert_almost_eq!(super::gamma_ur(100.0, 110.0), 0.1582786700600870938017003790170311468066499691341777, 1e-12);
        assert_almost_eq!(super::gamma_ur(100.0, 200.0), 0.0, 1e-14);
        assert_almost_eq!(super::gamma_ur(500.0, 0.5), 1.0, 1e-14);
        assert_almost_eq!(super::gamma_ur(500.0, 1.5), 1.0, 1e-14);
        assert_almost_eq!(super::gamma_ur(500.0, 200.0), 1.0, 1e-14);
        assert_almost_eq!(super::gamma_ur(500.0, 450.0), 0.9892827619087102584426041229344795034565130050758519, 1e-12);
        assert_almost_eq!(super::gamma_ur(500.0, 500.0), 0.4940528538292396419529520425587941967197264574365736, 1e-12);
        assert_almost_eq!(super::gamma_ur(500.0, 550.0), 0.0146144081262951940451529993099155334419383681297251, 1e-12);
        assert_almost_eq!(super::gamma_ur(500.0, 700.0), 0.0, 1e-14);
        assert_almost_eq!(super::gamma_ur(1000.0, 10000.0), 0.0, 1e-14);
        assert_almost_eq!(super::gamma_ur(1e+50, 1e+48), 1.0, 1e-14);
        assert_almost_eq!(super::gamma_ur(1e+50, 1e+52), 0.0, 1e-14);
    }

    #[test]
    #[should_panic]
    fn test_gamma_ur_a_lower_bound() {
        super::gamma_ur(-1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_ur_a_upper_bound() {
        super::gamma_ur(f64::INFINITY, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_ur_x_lower_bound() {
        super::gamma_ur(1.0, -1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_ur_x_upper_bound() {
        super::gamma_ur(1.0, f64::INFINITY);
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
        assert!(super::gamma_ui(f64::NAN, f64::NAN).is_nan());
        assert_almost_eq!(super::gamma_ui(0.1, 1.0), 0.2295356702888460382790772147651768201739736396141314, 1e-14);
        assert_almost_eq!(super::gamma_ui(0.1, 2.0), 0.053977968112828232195991347726857391060870217694027, 1e-15);
        assert_almost_eq!(super::gamma_ui(0.1, 8.0), 0.000046852198327948595220974570460669512682180005810156, 1e-19);
        assert_almost_eq!(super::gamma_ui(1.5, 1.0), 0.50728223381177330984514007570018045349008617228036, 1e-14);
        assert_almost_eq!(super::gamma_ui(1.5, 2.0), 0.23171655200098069331588896692000996837607162348484, 1e-15);
        assert_almost_eq!(super::gamma_ui(1.5, 8.0), 0.0010049674106481758827326630820844265957854973504417, 1e-17);
        assert_almost_eq!(super::gamma_ui(2.5, 1.0), 1.1288027918891022863632338837117315476809403894523, 1e-14);
        assert_almost_eq!(super::gamma_ui(2.5, 2.0), 0.73036081404311473581698531119872971361489139002877, 1e-14);
        assert_almost_eq!(super::gamma_ui(2.5, 8.0), 0.0090981038847570846537821465810058289147856041616617, 1e-17);
        assert_almost_eq!(super::gamma_ui(5.5, 1.0), 52.264048055526551859457214287080473123160514369109, 1e-12);
        assert_almost_eq!(super::gamma_ui(5.5, 2.0), 50.768151250342155233775028625526081234006425883469, 1e-12);
        assert_almost_eq!(super::gamma_ui(5.5, 8.0), 7.3871823043570542965292707346232335470650967978006, 1e-13);
    }

    #[test]
    #[should_panic]
    fn test_gamma_ui_a_lower_bound() {
        super::gamma_ui(-1.0, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_ui_a_upper_bound() {
        super::gamma_ui(f64::INFINITY, 1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_ui_x_lower_bound() {
        super::gamma_ui(1.0, -1.0);
    }

    #[test]
    #[should_panic]
    fn test_gamma_ui_x_upper_bound() {
        super::gamma_ui(1.0, f64::INFINITY);
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

    // TODO: precision testing could be more accurate
    #[test]
    fn test_digamma() {
        assert!(super::digamma(f64::NAN).is_nan());
        assert_almost_eq!(super::digamma(-1.5), 0.70315664064524318722569033366791109947350706200623256, 1e-14);
        assert_almost_eq!(super::digamma(-0.5), 0.036489973978576520559023667001244432806840395339565891, 1e-14);
        assert_almost_eq!(super::digamma(0.1), -10.423754940411076232100295314502760886768558023951363, 1e-14);
        assert_almost_eq!(super::digamma(1.0), -0.57721566490153286060651209008240243104215933593992359, 1e-14);
        assert_almost_eq!(super::digamma(1.5), 0.036489973978576520559023667001244432806840395339565888, 1e-14);
        assert_almost_eq!(super::digamma(consts::PI / 2.0), 0.10067337642740238636795561404029690452798358068944001, 1e-14);
        assert_almost_eq!(super::digamma(2.0), 0.42278433509846713939348790991759756895784066406007641, 1e-14);
        assert_almost_eq!(super::digamma(2.5), 0.70315664064524318722569033366791109947350706200623255, 1e-14);
        assert_almost_eq!(super::digamma(3.0), 0.92278433509846713939348790991759756895784066406007641, 1e-14);
        assert_almost_eq!(super::digamma(consts::PI), 0.97721330794200673329206948640618234364083460999432603, 1e-14);
        assert_almost_eq!(super::digamma(3.5), 1.1031566406452431872256903336679110994735070620062326, 1e-14);
        assert_almost_eq!(super::digamma(4.0), 1.2561176684318004727268212432509309022911739973934097, 1e-14);
        assert_almost_eq!(super::digamma(4.5), 1.3888709263595289015114046193821968137592213477205183, 1e-14);
        assert_almost_eq!(super::digamma(5.0), 1.5061176684318004727268212432509309022911739973934097, 1e-14);
        assert_almost_eq!(super::digamma(5.5), 1.6110931485817511237336268416044190359814435699427405, 1e-14);
        assert_almost_eq!(super::digamma(10.1), 2.2622143570941481235561593642219403924532310597356171, 1e-14);
    }

    #[test]
    fn test_inv_digamma() {
        assert!(super::inv_digamma(f64::NAN).is_nan());
        assert_eq!(super::inv_digamma(f64::NEG_INFINITY), 0.0);
        assert_almost_eq!(super::inv_digamma(-10.423754940411076232100295314502760886768558023951363), 0.1, 1e-15);
        assert_almost_eq!(super::inv_digamma(-0.57721566490153286060651209008240243104215933593992359), 1.0, 1e-14);
        assert_almost_eq!(super::inv_digamma(0.036489973978576520559023667001244432806840395339565888), 1.5, 1e-14);
        assert_almost_eq!(super::inv_digamma(0.10067337642740238636795561404029690452798358068944001), consts::PI / 2.0, 1e-14);
        assert_almost_eq!(super::inv_digamma(0.42278433509846713939348790991759756895784066406007641), 2.0, 1e-14);
        assert_almost_eq!(super::inv_digamma(0.70315664064524318722569033366791109947350706200623255), 2.5, 1e-14);
        assert_almost_eq!(super::inv_digamma(0.92278433509846713939348790991759756895784066406007641), 3.0, 1e-14);
        assert_almost_eq!(super::inv_digamma(0.97721330794200673329206948640618234364083460999432603), consts::PI, 1e-14);
        assert_almost_eq!(super::inv_digamma(1.1031566406452431872256903336679110994735070620062326), 3.5, 1e-14);
        assert_almost_eq!(super::inv_digamma(1.2561176684318004727268212432509309022911739973934097), 4.0, 1e-14);
        assert_almost_eq!(super::inv_digamma(1.3888709263595289015114046193821968137592213477205183), 4.5, 1e-14);
        assert_almost_eq!(super::inv_digamma(1.5061176684318004727268212432509309022911739973934097), 5.0, 1e-14);
        assert_almost_eq!(super::inv_digamma(1.6110931485817511237336268416044190359814435699427405), 5.5, 1e-14);
        assert_almost_eq!(super::inv_digamma(2.2622143570941481235561593642219403924532310597356171), 10.1, 1e-13);
    }

    #[test]
    fn test_error_is_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<GammaFuncError>();
    }
}
