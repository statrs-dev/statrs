use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use statrs::function::beta::{beta_reg, inv_beta_reg};
use std::hint::black_box;

fn bench_beta_reg(c: &mut Criterion) {
    let mut group = c.benchmark_group("beta_reg");
    for (name, a, b, x) in [
        ("typical", 2.0, 5.0, 0.3),
        (
            "large_symmetric_adjacent",
            1e8,
            1e8,
            f64::from_bits(0.5_f64.to_bits() + 1),
        ),
        (
            "moderate_fraction",
            25.32628846940565,
            3.1028101710805442,
            0.9276950604606229,
        ),
    ] {
        group.bench_with_input(
            BenchmarkId::new("cdf", name),
            &(a, b, x),
            |bencher, input| {
                bencher
                    .iter(|| beta_reg(black_box(input.0), black_box(input.1), black_box(input.2)));
            },
        );
    }
    group.finish();
}

fn bench_inv_beta_reg(c: &mut Criterion) {
    let mut group = c.benchmark_group("inv_beta_reg");
    for (name, a, b, probability) in [
        ("typical", 2.0, 5.0, 0.3),
        ("nontermination_regression", 200.0, 2.0, 1e-60),
        ("panic_regression", 200.0, 2.0, 1e-165),
        ("tiny_quantile", 0.1, 500.0, 1e-30),
    ] {
        group.bench_with_input(
            BenchmarkId::new("quantile", name),
            &(a, b, probability),
            |bencher, input| {
                bencher.iter(|| {
                    inv_beta_reg(black_box(input.0), black_box(input.1), black_box(input.2))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_beta_reg, bench_inv_beta_reg);
criterion_main!(benches);
