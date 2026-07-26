//! Integration tests for `ln_cdf` / `ln_sf` (statrs-dev/statrs#338).
//!
//! Two properties are pinned:
//!  1. consistency: where `cdf`/`sf` are comfortably representable, the log
//!     variants agree with `cdf().ln()` to ~1e-13 relative;
//!  2. tail accuracy: far past the underflow point of the probability (where
//!     the default `cdf().ln()` is `-inf`), the overrides match mpmath
//!     references computed at 60 significant digits.

use approx::assert_relative_eq;
use statrs::distribution::*;

/// The log stays within `tol` (relative on the log, absolute near zero) of the
/// linear-domain value wherever the latter is representable and not too close
/// to a hard 0 or 1.
fn consistent<F: Fn(f64) -> (f64, f64)>(pairs: F, xs: &[f64]) {
    for &x in xs {
        let (lin, log) = pairs(x);
        if lin > 1e-290 && lin < 1.0 {
            assert_relative_eq!(lin.ln(), log, epsilon = 1e-12, max_relative = 1e-12);
        }
    }
}

#[test]
fn ln_cdf_ln_sf_consistent_with_linear_domain() {
    let n = Normal::new(0.5, 2.0).unwrap();
    consistent(
        |x| (n.cdf(x), n.ln_cdf(x)),
        &[-50.0, -8.0, -1.0, 0.5, 3.0, 40.0],
    );
    consistent(|x| (n.sf(x), n.ln_sf(x)), &[-40.0, -3.0, 0.5, 8.0, 50.0]);

    let g = Gamma::new(5.5, 2.0).unwrap();
    consistent(|x| (g.cdf(x), g.ln_cdf(x)), &[1e-6, 0.5, 2.75, 10.0, 100.0]);
    consistent(|x| (g.sf(x), g.ln_sf(x)), &[1e-6, 0.5, 2.75, 10.0, 100.0]);

    let c = ChiSquared::new(3.0).unwrap();
    consistent(|x| (c.cdf(x), c.ln_cdf(x)), &[0.01, 1.0, 3.0, 30.0, 300.0]);
    consistent(|x| (c.sf(x), c.ln_sf(x)), &[0.01, 1.0, 3.0, 30.0, 300.0]);

    let b = Beta::new(3.5, 8.0).unwrap();
    consistent(|x| (b.cdf(x), b.ln_cdf(x)), &[1e-8, 0.1, 0.3, 0.7, 0.999]);
    consistent(|x| (b.sf(x), b.ln_sf(x)), &[0.001, 0.1, 0.3, 0.7, 0.999]);

    let e = Exp::new(2.0).unwrap();
    consistent(|x| (e.cdf(x), e.ln_cdf(x)), &[1e-10, 0.5, 5.0, 300.0]);
    consistent(|x| (e.sf(x), e.ln_sf(x)), &[1e-10, 0.5, 5.0, 300.0]);

    let w = Weibull::new(1.5, 2.0).unwrap();
    consistent(|x| (w.cdf(x), w.ln_cdf(x)), &[0.001, 1.0, 5.0, 50.0]);
    consistent(|x| (w.sf(x), w.ln_sf(x)), &[0.001, 1.0, 5.0, 50.0]);

    let p = Pareto::new(1.0, 3.0).unwrap();
    consistent(|x| (p.cdf(x), p.ln_cdf(x)), &[1.001, 2.0, 100.0, 1e50]);
    consistent(|x| (p.sf(x), p.ln_sf(x)), &[1.001, 2.0, 100.0, 1e50]);

    let ln = LogNormal::new(0.0, 1.0).unwrap();
    consistent(
        |x| (ln.cdf(x), ln.ln_cdf(x)),
        &[1e-10, 0.1, 1.0, 10.0, 1e10],
    );
    consistent(|x| (ln.sf(x), ln.ln_sf(x)), &[1e-10, 0.1, 1.0, 10.0, 1e10]);

    let bi = Binomial::new(0.3, 1000).unwrap();
    let po = Poisson::new(50.0).unwrap();
    let ge = Geometric::new(0.05).unwrap();
    for k in [1u64, 10, 100, 250, 400, 900] {
        for (lin, log) in [
            (bi.cdf(k), bi.ln_cdf(k)),
            (bi.sf(k), bi.ln_sf(k)),
            (po.cdf(k), po.ln_cdf(k)),
            (po.sf(k), po.ln_sf(k)),
            (ge.cdf(k), ge.ln_cdf(k)),
            (ge.sf(k), ge.ln_sf(k)),
        ] {
            if lin > 1e-290 && lin < 1.0 {
                assert_relative_eq!(lin.ln(), log, epsilon = 1e-12, max_relative = 1e-12);
            }
        }
    }
}

/// References are mpmath at 60 significant digits. Every `cdf`/`sf` here is a
/// hard 0 in f64, so the default `cdf().ln()` returns `-inf`; the overrides
/// carry the probability's relative accuracy (~1e-15) into the log domain.
#[test]
fn ln_cdf_ln_sf_deep_tails_match_references() {
    let n = Normal::standard();
    assert_eq!(n.sf(39.0), 0.0, "premise: sf underflows here");
    assert_relative_eq!(n.ln_sf(39.0), -765.0831565643775, max_relative = 1e-14);
    assert_relative_eq!(n.ln_sf(100.0), -5005.524208694205, max_relative = 1e-14);
    assert_relative_eq!(n.ln_sf(1000.0), -500007.82669481216, max_relative = 1e-14);
    assert_relative_eq!(n.ln_cdf(-39.0), -765.0831565643775, max_relative = 1e-14);

    let c = ChiSquared::new(1.0).unwrap();
    assert_eq!(c.sf(1500.0), 0.0, "premise: sf underflows here");
    assert_relative_eq!(c.ln_sf(1500.0), -753.8830671053825, max_relative = 1e-14);
    assert_relative_eq!(c.ln_sf(5000.0), -2504.484587848451, max_relative = 1e-14);

    let g = Gamma::new(5.5, 2.0).unwrap();
    assert_relative_eq!(g.ln_sf(500.0), -972.8684095883499, max_relative = 1e-14);
    assert_relative_eq!(g.ln_cdf(1e-6), -77.83556232788861, max_relative = 1e-14);

    let po = Poisson::new(1000.0).unwrap();
    assert_relative_eq!(po.ln_cdf(100), -672.8586102872655, max_relative = 1e-13);

    let bi = Binomial::new(0.5, 10_000).unwrap();
    assert_eq!(bi.cdf(1000), 0.0, "premise: cdf underflows here");
    assert_relative_eq!(bi.ln_cdf(1000), -3684.8445400592873, max_relative = 1e-13);

    let be = Beta::new(3.5, 8.0).unwrap();
    assert_relative_eq!(be.ln_cdf(1e-30), -236.4583322197154, max_relative = 1e-14);

    let lo = LogNormal::new(0.0, 1.0).unwrap();
    assert_relative_eq!(lo.ln_cdf(1e-100), -26515.84871241671, max_relative = 1e-14);

    // exact closed forms
    let e = Exp::new(2.0).unwrap();
    assert_eq!(e.ln_sf(400.0), -800.0);
    assert_eq!(e.ln_sf(1e8), -2e8);
    let pa = Pareto::new(1.0, 3.0).unwrap();
    assert_relative_eq!(
        pa.ln_sf(1e100),
        -300.0 * core::f64::consts::LN_10,
        max_relative = 1e-15
    );
    let ge = Geometric::new(0.01).unwrap();
    assert_relative_eq!(
        ge.ln_sf(200_000),
        200_000.0 * (-0.01f64).ln_1p(),
        max_relative = 1e-15
    );
}

/// Distributions without an override fall back to `cdf().ln()`; check the
/// default exists and behaves.
#[test]
fn ln_cdf_default_impl_works() {
    let u = Uniform::new(0.0, 1.0).unwrap();
    assert_relative_eq!(u.ln_cdf(0.25), 0.25f64.ln(), max_relative = 1e-15);
    assert_relative_eq!(u.ln_sf(0.25), 0.75f64.ln(), max_relative = 1e-15);
    let t = StudentsT::new(0.0, 1.0, 5.0).unwrap();
    assert_relative_eq!(t.ln_cdf(1.0), t.cdf(1.0).ln(), max_relative = 1e-15);
}

/// Boundary semantics: logs of exact 0 and 1.
#[test]
fn ln_cdf_ln_sf_boundaries() {
    let n = Normal::standard();
    assert_eq!(n.ln_cdf(f64::NEG_INFINITY), f64::NEG_INFINITY);
    assert_eq!(n.ln_cdf(f64::INFINITY), 0.0);
    assert_eq!(n.ln_sf(f64::NEG_INFINITY), 0.0);
    assert_eq!(n.ln_sf(f64::INFINITY), f64::NEG_INFINITY);

    let bi = Binomial::new(0.3, 10).unwrap();
    assert_eq!(bi.ln_cdf(10), 0.0);
    assert_eq!(bi.ln_sf(10), f64::NEG_INFINITY);

    let e = Exp::new(2.0).unwrap();
    assert_eq!(e.ln_cdf(-1.0), f64::NEG_INFINITY);
    assert_eq!(e.ln_sf(-1.0), 0.0);

    let w = Weibull::new(1.5, 2.0).unwrap();
    assert_eq!(w.ln_cdf(-1.0), f64::NEG_INFINITY);
    assert_eq!(w.ln_sf(-1.0), 0.0);

    let pa = Pareto::new(1.0, 3.0).unwrap();
    assert_eq!(pa.ln_cdf(0.5), f64::NEG_INFINITY);
    assert_eq!(pa.ln_sf(0.5), 0.0);
}
