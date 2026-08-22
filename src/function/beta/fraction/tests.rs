use super::*;

#[test]
fn double_double_fraction_handles_zero_initial_denominator() {
    let fraction = beta_continued_fraction_dd(1.0, 3.0, (0.5, 0.0)).unwrap();

    assert!(fraction.0.is_finite());
    assert!(fraction.1.is_finite());
    assert!(fraction.0 + fraction.1 > 0.0);
}
