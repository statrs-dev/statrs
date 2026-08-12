extern crate criterion;
extern crate rand;
extern crate statrs;
use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::{Vector1, Vector3};
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::StandardUniform;
use rand::rngs::StdRng;

fn generate<T>(n_samples: usize) -> Vec<T>
where
    StandardUniform: rand::distr::Distribution<T>,
{
    let mut rng = StdRng::seed_from_u64(42);
    (0..n_samples).map(|_| rng.random()).collect()
}

fn bench_density(c: &mut Criterion) {
    let samples = generate(100_000);
    let mut group = c.benchmark_group("density");
    group.bench_function("knn_density_1d", |b| {
        b.iter(|| {
            let _f = statrs::density::knn::knn_pdf(&[0.], &samples, None);
        });
    });

    let samples = generate(100_000);
    group.bench_function("knn_density_3d", |b| {
        b.iter(|| {
            let _f = statrs::density::knn::knn_pdf(&[0., 0., 0.], &samples, None);
        });
    });

    let samples = generate(100_000);
    group.bench_function("kde_density_1d", |b| {
        b.iter(|| {
            let _f = statrs::density::kde::kde_pdf(&Vector1::new(0.), &samples, None);
        });
    });

    let samples = generate(100_000);
    group.bench_function("kde_density_3d", |b| {
        b.iter(|| {
            let _f = statrs::density::kde::kde_pdf(&Vector3::new(0., 0., 0.), &samples, None);
        });
    });
    group.finish();

    // The free functions above rebuild the k-d tree on every call. These
    // measure the same work with the tree hoisted out of the loop, which is the
    // point of `DensityEstimator`.
    let samples: Vec<[f64; 1]> = generate(100_000);
    let grid: Vec<[f64; 1]> = (0..200).map(|i| [i as f64 / 200.0]).collect();
    let mut group = c.benchmark_group("density_grid_200");
    group.sample_size(20);
    group.bench_function("knn_1d_free_fns", |b| {
        b.iter(|| {
            for g in &grid {
                let _f = statrs::density::knn::knn_pdf(g, &samples, Some(0.05));
            }
        });
    });
    group.bench_function("knn_1d_prepared_estimator", |b| {
        b.iter(|| {
            let est = statrs::density::DensityEstimator::new(&samples).unwrap();
            for g in &grid {
                let _f = est.knn_pdf(g, Some(0.05));
            }
        });
    });
    group.finish();
}

criterion_group!(benches, bench_density);

criterion_main!(benches);
