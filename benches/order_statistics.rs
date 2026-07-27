extern crate statrs;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::prelude::*;
use rand::rngs::StdRng;
use statrs::statistics::*;
use std::hint::black_box;

fn bench_order_statistic(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<_> = (0..100).map(|x| x as f64).collect();
    let order = black_box(rng.random_range(1..=data.len()));
    let percentile = black_box(rng.random_range(0..=100));
    let tau = black_box(rng.random_range(0.0..1.0));
    let mut to_random_owned = |data: &[f64]| -> Data<Vec<f64>> {
        let mut owned = data.to_vec();
        owned.shuffle(&mut rng);
        Data::new(owned)
    };
    let mut group = c.benchmark_group("order statistic");
    group.throughput(Throughput::ElementsAndBytes {
        elements: data.len() as u64,
        bytes: std::mem::size_of_val(data.as_slice()) as u64,
    });
    group.bench_function("order_statistic", |b| {
        b.iter_batched(
            || to_random_owned(&data),
            |mut data| data.order_statistic(order),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("median", |b| {
        b.iter_batched(
            || to_random_owned(&data),
            |data| data.median(),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("quantile", |b| {
        b.iter_batched(
            || to_random_owned(&data),
            |mut data| data.quantile(tau),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("percentile", |b| {
        b.iter_batched(
            || to_random_owned(&data),
            |mut data| data.percentile(percentile),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("lower_quartile", |b| {
        b.iter_batched(
            || to_random_owned(&data),
            |mut data| data.lower_quartile(),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("upper_quartile", |b| {
        b.iter_batched(
            || to_random_owned(&data),
            |mut data| data.upper_quartile(),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("interquartile_range", |b| {
        b.iter_batched(
            || to_random_owned(&data),
            |mut data| data.interquartile_range(),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("ranks: RankTieBreaker::First", |b| {
        b.iter_batched(
            || to_random_owned(&data),
            |mut data| data.ranks(RankTieBreaker::First),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("ranks: RankTieBreaker::Average", |b| {
        b.iter_batched(
            || to_random_owned(&data),
            |mut data| data.ranks(RankTieBreaker::Average),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("ranks: RankTieBreaker::Min", |b| {
        b.iter_batched(
            || to_random_owned(&data),
            |mut data| data.ranks(RankTieBreaker::Min),
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

/// Selection cost across input sizes and shapes.
///
/// Kept separate from the fixed-size group above so the two can evolve
/// independently, and written against only the public `Data` API so the same
/// file compiles on any branch. To compare two implementations:
///
/// ```text
/// git checkout main    && cargo bench --bench order_statistics -- --save-baseline main
/// git checkout <branch> && cargo bench --bench order_statistics -- --baseline main
/// ```
///
/// The shapes matter as much as the sizes. A median-of-three pivot is fine on
/// random data and degrades on input built to defeat it, so a random-only
/// benchmark would understate the difference between a quickselect and an
/// introselect with an O(n) fallback.
fn bench_selection_scaling(c: &mut Criterion) {
    fn shaped(shape: &str, n: usize) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(0xB0A7);
        match shape {
            // shuffled, the ordinary case
            "random" => {
                let mut v: Vec<f64> = (0..n).map(|i| i as f64).collect();
                v.shuffle(&mut rng);
                v
            }
            // already ordered, both directions
            "sorted" => (0..n).map(|i| i as f64).collect(),
            "reversed" => (0..n).rev().map(|i| i as f64).collect(),
            // ascends then descends: the classic median-of-three adversary,
            // since the first, middle and last elements are unrepresentative
            "organ_pipe" => (0..n)
                .map(|i| if i < n / 2 { i } else { n - i } as f64)
                .collect(),
            // few distinct values, so partitions are heavily unbalanced
            "duplicates" => {
                let distinct = ((n as f64).sqrt() as usize).max(1);
                let mut v: Vec<f64> = (0..n).map(|i| (i % distinct) as f64).collect();
                v.shuffle(&mut rng);
                v
            }
            other => panic!("unknown shape {other}"),
        }
    }

    let mut group = c.benchmark_group("selection scaling");
    for &n in &[1_024usize, 65_536, 1_048_576] {
        for shape in ["random", "sorted", "reversed", "organ_pipe", "duplicates"] {
            let data = shaped(shape, n);
            group.throughput(Throughput::Elements(n as u64));
            group.bench_function(format!("median/{shape}/{n}"), |b| {
                b.iter_batched(
                    || Data::new(data.clone()),
                    |data| black_box(data.median()),
                    BatchSize::LargeInput,
                )
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_order_statistic, bench_selection_scaling);
criterion_main!(benches);
