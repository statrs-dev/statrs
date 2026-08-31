//! Cross-cutting checks that `Median` agrees with the cdf it is defined by.
//!
//! Several distributions have no closed-form median and define it as
//! `inverse_cdf(0.5)` (statrs-dev/statrs#276). For those, `cdf(median()) == 0.5`
//! is a reference-free correctness check: it needs no external tables, and it
//! fails loudly if the underlying inverse is bracketed wrongly or fails to
//! converge.

use statrs::distribution::{
    Beta, Chi, ContinuousCDF, DiscreteCDF, Erlang, FisherSnedecor, Gamma, Hypergeometric,
    InverseGamma, NegativeBinomial,
};
use statrs::statistics::{Distribution, Median, Min};

/// For a continuous distribution the median is the exact point where the cdf
/// crosses one half.
fn assert_continuous<D: ContinuousCDF<f64, f64> + Median<f64>>(d: &D, label: &str) {
    let m = d.median();
    assert!(m.is_finite(), "{label}: median was not finite ({m})");
    let c = d.cdf(m);
    assert!(
        (c - 0.5).abs() < 1e-10,
        "{label}: cdf(median) = {c}, expected 0.5 (median = {m})"
    );
}

/// For a discrete distribution the median is the smallest `k` with
/// `cdf(k) >= 0.5`, so the value below it must fall short.
fn assert_discrete<D: DiscreteCDF<u64, f64> + Median<f64> + Min<u64>>(d: &D, label: &str) {
    let m = d.median();
    assert!(m.is_finite() && m >= 0.0, "{label}: bad median {m}");
    let k = m as u64;
    let c = d.cdf(k);
    assert!(c >= 0.5, "{label}: cdf({k}) = {c}, expected >= 0.5");
    if k > d.min() {
        let below = d.cdf(k - 1);
        assert!(
            below < 0.5,
            "{label}: cdf({}) = {below} already >= 0.5, so {k} is not the median",
            k - 1
        );
    }
}

#[test]
fn beta_median_matches_cdf() {
    for (a, b) in [
        (1.0, 1.0),
        (2.0, 2.0),
        (0.5, 0.5),
        (2.0, 5.0),
        (5.0, 2.0),
        (0.1, 0.1),
        (1e3, 1e3),
        (1.0, 100.0),
        (100.0, 1.0),
        (0.5, 50.0),
    ] {
        let d = Beta::new(a, b).unwrap();
        assert_continuous(&d, &format!("Beta({a}, {b})"));
    }
    // Symmetric parameters put the median exactly at one half.
    for a in [0.5, 1.0, 2.0, 10.0, 1e3] {
        let m = Beta::new(a, a).unwrap().median();
        assert!(
            (m - 0.5).abs() < 1e-12,
            "Beta({a}, {a}) median = {m}, expected 0.5 by symmetry"
        );
    }
}

#[test]
fn gamma_and_erlang_medians_match_cdf() {
    for (shape, rate) in [
        (1.0, 1.0),
        (2.0, 1.0),
        (0.5, 2.0),
        (10.0, 0.1),
        (1e3, 1.0),
        (1.0, 1e3),
        (0.1, 1.0),
    ] {
        let d = Gamma::new(shape, rate).unwrap();
        assert_continuous(&d, &format!("Gamma({shape}, {rate})"));
    }

    for (k, rate) in [(1u64, 1.0), (2, 1.0), (5, 2.0), (20, 0.5)] {
        let d = Erlang::new(k, rate).unwrap();
        assert_continuous(&d, &format!("Erlang({k}, {rate})"));
        // Erlang delegates to Gamma, so the two must agree exactly.
        let g = Gamma::new(k as f64, rate).unwrap();
        assert_eq!(d.median(), g.median(), "Erlang({k}, {rate}) != Gamma");
    }

    // The gamma median is known to lie in [shape - 1/3, shape] for shape >= 1
    // at unit rate. Check the bracket independently of the cdf.
    for shape in [1.0, 2.0, 5.0, 50.0, 1e3] {
        let m = Gamma::new(shape, 1.0).unwrap().median();
        assert!(
            m > shape - 1.0 / 3.0 - 1e-9 && m < shape,
            "Gamma({shape}, 1) median {m} outside [shape - 1/3, shape]"
        );
    }
}

#[test]
fn chi_median_matches_cdf() {
    for k in [1u64, 2, 3, 5, 10, 100] {
        let d = Chi::new(k).unwrap();
        assert_continuous(&d, &format!("Chi({k})"));
        // The median must sit between the distribution's own bounds.
        assert!(d.median() > 0.0);
    }
}

#[test]
fn inverse_gamma_median_matches_cdf() {
    for (shape, rate) in [(1.0, 1.0), (2.0, 1.0), (3.0, 2.0), (10.0, 0.5), (1.5, 3.0)] {
        let d = InverseGamma::new(shape, rate).unwrap();
        assert_continuous(&d, &format!("InverseGamma({shape}, {rate})"));
    }
}

#[test]
fn fisher_snedecor_median_matches_cdf() {
    for (d1, d2) in [
        (1.0, 1.0),
        (2.0, 2.0),
        (5.0, 10.0),
        (10.0, 5.0),
        (100.0, 100.0),
        (1.0, 50.0),
    ] {
        let d = FisherSnedecor::new(d1, d2).unwrap();
        assert_continuous(&d, &format!("FisherSnedecor({d1}, {d2})"));
    }
}

#[test]
fn hypergeometric_median_matches_cdf() {
    for (pop, succ, draws) in [
        (10u64, 5u64, 5u64),
        (50, 10, 20),
        (100, 50, 10),
        (20, 19, 10),
        (7, 1, 3),
        (1000, 500, 100),
    ] {
        let d = Hypergeometric::new(pop, succ, draws).unwrap();
        assert_discrete(&d, &format!("Hypergeometric({pop}, {succ}, {draws})"));
    }
}

#[test]
fn negative_binomial_median_matches_cdf() {
    for (r, p) in [
        (1.0, 0.5),
        (5.0, 0.5),
        (1.0, 0.1),
        (10.0, 0.9),
        (2.5, 0.3),
        (100.0, 0.5),
    ] {
        let d = NegativeBinomial::new(r, p).unwrap();
        assert_discrete(&d, &format!("NegativeBinomial({r}, {p})"));
    }
}

/// Where a distribution coincides with one that *does* have a closed-form
/// median, the numerical result must agree with it. This bounds the accuracy of
/// the search against an exact value rather than against the cdf it inverts.
///
/// The agreement is to ~1e-13 relative, not to the last bit: these medians come
/// from a root-find, so they are not correctly rounded the way a closed form is.
#[test]
fn numerical_medians_agree_with_closed_form_equivalents() {
    use statrs::distribution::{Exp, Normal};

    // Gamma(1, rate) is Exp(rate), whose median is the closed form ln(2)/rate.
    for rate in [0.5, 1.0, 2.0, 10.0] {
        let g = Gamma::new(1.0, rate).unwrap().median();
        let e = Exp::new(rate).unwrap().median();
        assert!(
            (g - e).abs() <= 1e-13 * e,
            "Gamma(1, {rate}) median {g} != Exp({rate}) median {e}"
        );
        // and both against ln(2)/rate directly
        let exact = std::f64::consts::LN_2 / rate;
        assert!((g - exact).abs() <= 1e-13 * exact);
    }

    // Erlang(1, rate) is likewise Exp(rate).
    for rate in [0.5, 1.0, 3.0] {
        let er = Erlang::new(1, rate).unwrap().median();
        let e = Exp::new(rate).unwrap().median();
        assert!((er - e).abs() <= 1e-13 * e, "Erlang(1, {rate}) != Exp");
    }

    // Chi(1) is the half-normal, so its median is the normal 0.75 quantile.
    let chi1 = Chi::new(1).unwrap().median();
    let q75 = Normal::new(0.0, 1.0).unwrap().inverse_cdf(0.75);
    assert!(
        (chi1 - q75).abs() <= 1e-12 * q75,
        "Chi(1) median {chi1} != N(0,1) 0.75-quantile {q75}"
    );

    // Beta(1, 1) is uniform on (0, 1).
    let u = Beta::new(1.0, 1.0).unwrap().median();
    assert!((u - 0.5).abs() <= 1e-13, "Beta(1,1) median {u} != 0.5");

    // InverseGamma(1, 1) has cdf exp(-1/x), so its median is 1 / ln(2).
    let ig = InverseGamma::new(1.0, 1.0).unwrap().median();
    let exact = 1.0 / std::f64::consts::LN_2;
    assert!(
        (ig - exact).abs() <= 1e-13 * exact,
        "InverseGamma(1,1) median {ig} != 1/ln(2) = {exact}"
    );
}

/// A median must lie between the distribution's mean-adjacent landmarks in the
/// obvious cases, and always inside its own support.
#[test]
fn medians_lie_within_support() {
    let b = Beta::new(2.0, 5.0).unwrap();
    assert!(b.median() > 0.0 && b.median() < 1.0);

    let g = Gamma::new(3.0, 1.5).unwrap();
    assert!(g.median() > 0.0);
    // For a right-skewed gamma the median sits below the mean.
    assert!(g.median() < g.mean().unwrap());

    let ig = InverseGamma::new(4.0, 2.0).unwrap();
    assert!(ig.median() > 0.0 && ig.median() < ig.mean().unwrap());
}
