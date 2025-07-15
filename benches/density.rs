extern crate criterion;
extern crate rand;
extern crate statrs;
use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::{Vector1, Vector3};
use rand::distributions::Standard;

fn generate<T>(n_samples: usize) -> Vec<T>
where
    Standard: rand::distributions::Distribution<T>,
{
    (0..n_samples).map(|_| rand::random()).collect()
}
fn bench_density(c: &mut Criterion) {
    let samples = generate(100_00);
    // generate_1d(100_000);
    let mut group = c.benchmark_group("density");
    group.bench_function("knn_density_1d", |b| {
        b.iter(|| {
            let _f = statrs::density::knn::knn_pdf([0.], &samples, None);
        });
    });
    let samples = generate(100_000);
    group.bench_function("knn_density_3d", |b| {
        b.iter(|| {
            let _f = statrs::density::knn::knn_pdf([0., 0., 0.], &samples, None);
        });
    });

    let samples = generate(100_000);
    group.bench_function("kde_density_1d", |b| {
        b.iter(|| {
            let _f = statrs::density::kde::kde_pdf(Vector1::new(0.), &samples, None);
        });
    });
    let samples = generate(100_000);
    group.bench_function("kde_density_3d", |b| {
        b.iter(|| {
            let _f = statrs::density::kde::kde_pdf(Vector3::new(0., 0., 0.), &samples, None);
        });
    });
}

criterion_group!(benches, bench_density);

criterion_main!(benches);
