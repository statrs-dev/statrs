#![no_std]

extern crate alloc;

use alloc::vec;
use statrs::distribution::{Categorical, Continuous, Empirical, Multinomial, MultivariateNormal};
use statrs::generate::log_spaced;
use statrs::statistics::{Data, Distribution, MeanN, OrderStatistics, RankTieBreaker};
use statrs::stats_tests::{NaNPolicy, f_oneway::f_oneway};

#[global_allocator]
static ALLOCATOR: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    core::arch::wasm32::unreachable()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() <= tolerance);
}

#[unsafe(no_mangle)]
pub extern "C" fn verify() {
    let categorical = Categorical::new(&[1.0, 2.0, 3.0]).unwrap();
    assert_close(categorical.mean().unwrap(), 4.0 / 3.0, 1e-12);

    let empirical = vec![1.0, 2.0, 3.0].into_iter().collect::<Empirical>();
    assert_close(empirical.mean().unwrap(), 2.0, 1e-12);

    assert_eq!(log_spaced(3, 0.0, 2.0), [1.0, 10.0, 100.0]);

    let mut data = Data::new([3.0, 1.0, 2.0, 2.0]);
    assert_eq!(data.ranks(RankTieBreaker::Average), [4.0, 1.0, 2.5, 2.5]);

    let multinomial = Multinomial::new(vec![1.0, 3.0], 4).unwrap();
    let multinomial_mean = multinomial.mean().unwrap();
    assert_close(multinomial_mean[0], 1.0, 1e-12);
    assert_close(multinomial_mean[1], 3.0, 1e-12);

    let normal = MultivariateNormal::new(vec![0.0, 0.0], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let point = vec![0.0, 0.0].into();
    assert_close(
        normal.pdf(&point),
        1.0 / (2.0 * core::f64::consts::PI),
        1e-12,
    );

    let (statistic, p_value) = f_oneway(
        vec![
            vec![6.0, 8.0, 4.0, 5.0, 3.0, 4.0],
            vec![8.0, 12.0, 9.0, 11.0, 6.0, 8.0],
            vec![13.0, 9.0, 11.0, 8.0, 7.0, 12.0],
        ],
        NaNPolicy::Error,
    )
    .unwrap();
    assert_close(statistic, 9.264705882352942, 1e-12);
    assert_close(p_value, 0.0023987773293929317, 1e-12);
}
