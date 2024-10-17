use criterion::{criterion_group, criterion_main, Criterion};
use either::Either;
use rand::prelude::*;
use twovec::*;
use std::hint::black_box;

pub fn criterion_benchmark(c: &mut Criterion) {
    // got this number from a random number generator. locking it in to improve bench consistency?
    let mut rand = SmallRng::seed_from_u64(5156341831701782996);

    // random order of random values
    let mut twovec = TwoVec::<u8, f32>::new();
    for _ in 0..10000 {
        if rand.gen_bool(0.5) {
            twovec.push_a(rand.gen());
        } else {
            twovec.push_b(rand.gen());
        }
    }

    let mut vec = Vec::<Either<u8, f32>>::new();
    for _ in 0..10000 {
        if rand.gen_bool(0.5) {
            vec.push(Either::Left(rand.gen()));
        } else {
            vec.push(Either::Right(rand.gen()));
        }
    }

    c.bench_function("1k Push TwoVec", |b| {
        b.iter(|| {
            let mut bench_vec = TwoVec::<u8, f32>::new();

            for v in &vec {
                let val = black_box(*v);
                match val {
                    Either::Left(x) => bench_vec.push(black_box(x)),
                    Either::Right(x) => bench_vec.push(black_box(x)),
                }
            }
        })
    });

    c.bench_function("1k Push Vec", |b| {
        b.iter(|| {
            let mut bench_vec = Vec::<Either<u8,f32>>::new();
            for v in &vec {
                let val = black_box(*v);
                match val {
                    Either::Left(x) => bench_vec.push(black_box(Either::Left(x))),
                    Either::Right(x) => bench_vec.push(black_box(Either::Right(x))),
                }
            }
        })
    });

    c.bench_function("Read Twovec 1k", |b| {
        b.iter(|| {
            for i in 0..1000 {
                match twovec.get_either(i).unwrap() {
                    Either::Left(x) => {
                        black_box(x);
                    }
                    Either::Right(x) => {
                        black_box(x);
                    }
                }
            }
        })
    });

    c.bench_function("Read Vec 1k", |b| {
        b.iter(|| {
            for i in 0..1000 {
                match vec[i] {
                    Either::Left(x) => {
                        black_box(x);
                    }
                    Either::Right(x) => {
                        black_box(x);
                    }
                }
            }
        })
    });

    c.bench_function("Read Twovec 10k", |b| {
        b.iter(|| {
            for i in 0..10000 {
                match twovec.get_either(i).unwrap() {
                    Either::Left(x) => {
                        black_box(x);
                    }
                    Either::Right(x) => {
                        black_box(x);
                    }
                }
            }
        })
    });

    c.bench_function("Read Vec 10k", |b| {
        b.iter(|| {
            for i in 0..10000 {
                match vec[i] {
                    Either::Left(x) => {
                        black_box(x);
                    }
                    Either::Right(x) => {
                        black_box(x);
                    }
                }
            }
        })
    });

    c.bench_function("Compare", |b| {
        b.iter(
            || {
                for (lhs, rhs) in std::iter::zip(twovec.into_iter(), vec.iter()) {
                    if lhs == *rhs {
                        black_box(lhs);
                    } else {
                        black_box(rhs);
                    }
                }
            },
        )
    });

    c.bench_function("Remove TwoVec",|b| {
        b.iter_batched(
            || twovec.clone(),
            |mut twovec| {
                while !twovec.is_empty() {
                    let i = rand.gen_range(0..twovec.len());
                    let val = twovec.remove_either(i);
                    black_box(val);
                }
            },
            criterion::BatchSize::SmallInput
        )
    });

    c.bench_function("Remove Vec",|b| {
        b.iter_batched(
            || vec.clone(),
            |mut vec| {
                while !vec.is_empty() {
                    let i = rand.gen_range(0..vec.len());
                    let val = vec.remove(i);
                    black_box(val);
                }
            },
            criterion::BatchSize::SmallInput
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
