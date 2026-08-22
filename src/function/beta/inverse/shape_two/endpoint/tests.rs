use super::overlap_result;

#[test]
fn overlap_policy_is_monotone_and_ties_to_even() {
    let cases = [(0_u64, 0_u64), (1, 2), (2, 2), (3, 4)];
    let results = cases.map(|(lower, expected)| {
        let result = overlap_result(lower);
        assert_eq!(result.to_bits(), expected);
        result
    });
    assert!(results.windows(2).all(|pair| pair[0] <= pair[1]));
}
