//! Provides the [logistic](http://en.wikipedia.org/wiki/Logistic_function) and
//! related functions

use crate::error::StatsError;
use crate::Result;

/// Computes the logistic function
pub fn logistic(p: f64) -> f64 {
    1.0 / ((-p).exp() + 1.0)
}

/// Computes the logit function
///
/// # Panics
///
/// If `p < 0.0` or `p > 1.0`
pub fn logit(p: f64) -> f64 {
    checked_logit(p).unwrap()
}

/// Computes the logit function
///
/// # Errors
///
/// If `p < 0.0` or `p > 1.0`
pub fn checked_logit(p: f64) -> Result<f64> {
    if !(0.0..=1.0).contains(&p) {
        Err(StatsError::ArgIntervalIncl("p", 0.0, 1.0))
    } else {
        Ok((p / (1.0 - p)).ln())
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use std::f64;

    #[test]
    fn test_logistic() {
        assert_eq!(super::logistic(f64::NEG_INFINITY), 0.0);
        assert_eq!(super::logistic(-11.512915464920228103874353849992239636376994324587), 0.00001);
        assert_almost_eq!(super::logistic(-6.9067547786485535272274487616830597875179908939086), 0.001, 1e-18);
        assert_almost_eq!(super::logistic(-2.1972245773362193134015514347727700402304323440139), 0.1, 1e-16);
        assert_eq!(super::logistic(0.0), 0.5);
        assert_almost_eq!(super::logistic(2.1972245773362195801634726294284168954491240598975), 0.9, 1e-15);
        assert_almost_eq!(super::logistic(6.9067547786485526081487245019905638981131702804661), 0.999, 1e-15);
        assert_eq!(super::logistic(11.512915464924779098232747799811946290419057060965), 0.99999);
        assert_eq!(super::logistic(f64::INFINITY), 1.0);
    }

    #[test]
    fn test_logit() {
        assert_eq!(super::logit(0.0), f64::NEG_INFINITY);
        assert_eq!(super::logit(0.00001), -11.512915464920228103874353849992239636376994324587);
        assert_eq!(super::logit(0.001), -6.9067547786485535272274487616830597875179908939086);
        assert_eq!(super::logit(0.1), -2.1972245773362193134015514347727700402304323440139);
        assert_eq!(super::logit(0.5), 0.0);
        assert_eq!(super::logit(0.9), 2.1972245773362195801634726294284168954491240598975);
        assert_eq!(super::logit(0.999), 6.9067547786485526081487245019905638981131702804661);
        assert_eq!(super::logit(0.99999), 11.512915464924779098232747799811946290419057060965);
        assert_eq!(super::logit(1.0), f64::INFINITY);
    }

    #[test]
    #[should_panic]
    fn test_logit_p_lt_0() {
        super::logit(-1.0);
    }

    #[test]
    #[should_panic]
    fn test_logit_p_gt_1() {
        super::logit(2.0);
    }

    #[test]
    fn test_checked_logit_p_lt_0() {
        assert!(super::checked_logit(-1.0).is_err());
    }

    #[test]
    fn test_checked_logit_p_gt_1() {
        assert!(super::checked_logit(2.0).is_err());
    }
}
