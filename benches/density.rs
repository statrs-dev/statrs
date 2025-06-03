extern crate criterion;
extern crate rand;
extern crate statrs;
use criterion::{criterion_group, criterion_main, Criterion};

fn generate_1d(n_samples: usize) -> Vec<[f64; 1]> {
    (0..n_samples).map(|_| [rand::random()]).collect()
}

fn generate_3d(n_samples: usize) -> Vec<[f64; 3]> {
    (0..n_samples).map(|_| rand::random()).collect()
}

fn bench_density(c: &mut Criterion) {
    let samples = generate_1d(100_000);
    let mut group = c.benchmark_group("density");
    group.bench_function("knn_density_1d", |b| {
        b.iter(|| {
            let _f = statrs::density::knn::knn_pdf([0.], &samples, None);
        });
    });
    let samples = generate_3d(100_000);
    group.bench_function("knn_density_3d", |b| {
        b.iter(|| {
            let _f = statrs::density::knn::knn_pdf([0., 0., 0.], &samples, None);
        });
    });
}

criterion_group!(benches, bench_density);

criterion_main!(benches);
