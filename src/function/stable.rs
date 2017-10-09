//! Provides [numerically
//! stable](https://en.wikipedia.org/wiki/Numerical_stability) functions

use function::internal;
use std::f64;

/// Calculates a numerically stable `exp(x) - 1`
///
/// # Remarks
///
/// Returns `f64::NAN` if `pow` is `f64::NAN`
pub fn exp_minus_one(pow: f64) -> f64 {
    pow.exp_m1()
}

#[cfg_attr(rustfmt, rustfmt_skip)]
#[cfg(test)]
mod test {
    use std::f64;

    #[test]
    fn test_exp_minus_one() {
        assert!(super::exp_minus_one(f64::NAN).is_nan());
        assert_eq!(super::exp_minus_one(f64::INFINITY), f64::INFINITY);
        assert_eq!(super::exp_minus_one(f64::NEG_INFINITY), -1.0);
        assert_eq!(super::exp_minus_one(-50.0), -0.99999999999999999999980712501520360822169826571834729874);
        assert_almost_eq!(super::exp_minus_one(-2.5), -0.91791500137610120483047132553284019216219587898456, 1e-15);
        assert_almost_eq!(super::exp_minus_one(-0.1), -0.09516258196404042683575094055356337880529463901959, 1e-8);
        assert_eq!(super::exp_minus_one(0.0), 0.0);
        assert_almost_eq!(super::exp_minus_one(0.1), 0.1051709180756476248117078264902466682245471947375187, 1e-8);
        assert_eq!(super::exp_minus_one(2.5), 11.182493960703473438070175951167966183182767790063161);
        assert_eq!(super::exp_minus_one(50.0), 5.1847055285870724640864533229334853848274691005838464e21);
    }
}
