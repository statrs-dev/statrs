use crate::distribution::ContinuousCDF;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum AndersonDarlingError {
    SampleSizeInvalid,
}

impl core::fmt::Display for AndersonDarlingError {
    /// Formats the error for display.
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            AndersonDarlingError::SampleSizeInvalid => {
                write!(f, "Sample size `n` must be greater than 0.")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AndersonDarlingError {}

pub fn anderson_darling<T: ContinuousCDF<f64, f64>>(
    f_obs: &[f64],
    dist: &T,
) -> Result<(f64, f64), AndersonDarlingError> {
    let n = f_obs.len();
    if n == 0 {
        return Err(AndersonDarlingError::SampleSizeInvalid);
    }
    let mut f_obs = f_obs.to_vec();
    f_obs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let n_float = n as f64;
    let beta: f64 = (1.0 / n_float)
        * (0..n)
            .map(|i| {
                (2.0 * (i + 1) as f64 - 1.0)
                    * (f64::ln(dist.cdf(f_obs[i])) + f64::ln(1.0 - dist.cdf(f_obs[n - 1 - i])))
            })
            .sum::<f64>();
    let a_squared = -n_float - beta;
    let a_squared_adjusted = a_squared * (1.0 + (0.75 / n_float) + (2.25 / n_float.powi(2)));
    let p_value = if a_squared_adjusted >= 0.6 {
        (1.2937 - 5.709 * a_squared_adjusted + 0.0186 * a_squared_adjusted.powi(2)).exp()
    } else if a_squared_adjusted >= 0.34 {
        (0.9177 - 4.279 * a_squared_adjusted - 1.38 * a_squared_adjusted.powi(2)).exp()
    } else if a_squared_adjusted >= 0.2 {
        1.0 - (-8.318 + 42.796 * a_squared_adjusted - 59.938 * a_squared_adjusted.powi(2)).exp()
    } else {
        1.0 - (-13.436 + 101.14 * a_squared_adjusted - 223.73 * a_squared_adjusted.powi(2)).exp()
    };

    Ok((a_squared, p_value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::{Gamma, Normal};

    #[test]
    fn test_normality_good_fit() {
        let data = vec![5.2, 4.9, 5.5, 4.8, 5.0, 5.1, 5.3, 4.7, 5.4, 4.9, 5.2, 5.0];
        let n = data.len();
        let n_float = n as f64;
        let mean = data.iter().sum::<f64>() / n_float;
        let std_dev =
            (data.iter().map(|&val| (val - mean).powi(2)).sum::<f64>() / (n_float - 1.0)).sqrt();
        let normal_dist = Normal::new(mean, std_dev).unwrap();

        let (stat, p_value) = anderson_darling(&data, &normal_dist).unwrap();

        assert!(stat < 0.5, "Statistic should be low for a good fit");
        assert!(p_value > 0.05, "P-value should be high for a good fit");
    }

    #[test]
    fn test_normality_poor_fit() {
        let data = vec![1.0, 1.2, 1.5, 1.9, 2.0, 2.1, 2.2, 2.3, 5.0, 8.0, 12.0];
        let n = data.len();
        let n_float = n as f64;
        let mean = data.iter().sum::<f64>() / n_float;
        let std_dev =
            (data.iter().map(|&val| (val - mean).powi(2)).sum::<f64>() / (n_float - 1.0)).sqrt();
        let normal_dist = Normal::new(mean, std_dev).unwrap();

        let (stat, p_value) = anderson_darling(&data, &normal_dist).unwrap();

        assert!(stat > 0.5, "Statistic should be high for a poor fit");
        assert!(p_value < 0.05, "P-value should be low for a poor fit");
    }

    #[test]
    fn test_gamma_distribution_good_fit() {
        let data = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0];
        let gamma_dist = Gamma::new(2.0, 1.0).unwrap();
        let (stat, p_value) = anderson_darling(&data, &gamma_dist).unwrap();
        assert!(stat < 1.0, "Statistic should be low for a good gamma fit");
        assert!(p_value > 0.05, "P-value should be high for a good fit");
    }

    #[test]
    fn test_gamma_distribution_bad_fit() {
        let data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let gamma_dist = Gamma::new(2.0, 1.0).unwrap();
        let (stat, p_value) = anderson_darling(&data, &gamma_dist).unwrap();
        assert!(stat > 1.0, "Statistic should be high for a bad gamma fit");
        assert!(p_value < 0.05, "P-value should be low for a bad fit");
    }

    #[test]
    fn test_sample_size_invalid() {
        let data: Vec<f64> = vec![];
        let normal_dist = Normal::new(0.0, 1.0).unwrap();
        let result = anderson_darling(&data, &normal_dist);
        assert!(result.is_err());
    }
}
