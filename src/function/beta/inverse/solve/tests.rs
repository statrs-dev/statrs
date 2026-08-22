use super::*;

#[test]
fn log_value_computes_missing_accurate_beta_parts() {
    let a = 2.5;
    let b = 0.5;
    let x = 0.8;
    let log_beta = ln_beta_inverse_parts(a, b);
    let expected = inverse_beta_log_value_parts(
        a,
        b,
        x,
        log_beta,
        Some(ln_beta_inverse_accurate_parts(a, b)),
    )
    .unwrap();
    let actual = inverse_beta_log_value_parts(a, b, x, log_beta, None).unwrap();

    assert_eq!(actual, expected);
}
