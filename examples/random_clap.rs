extern crate statrs;

use nalgebra as na;
use rand::{distributions, thread_rng, Rng};
use statrs::distribution::{Binomial, Continuous, Multinomial, Normal};

use std::fmt::Display;
use std::io::{self, Write};

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name="random_clapping", author, version="0.0.1", about, long_about = None)]
struct Args {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// for sampling
    Sample {
        #[arg(value_name = "SAMPLE COUNT")]
        count: Option<usize>,
        #[command(subcommand)]
        dist: DistributionAsCommand,
    },
    /// for evaluating distribution function density
    Density { arg: String, dist: String },
}

#[derive(Subcommand, Debug)]
enum DistributionAsCommand {
    /// the multinomial distribution
    Multinomial {
        #[arg(value_name = "trial counts")]
        n: u64,
        #[arg(value_name = "success probabilities")]
        p: Vec<f64>,
    },
    /// the binomial distribution
    Binomial {
        #[arg(value_name = "trial counts")]
        n: u64,
        #[arg(value_name = "success probability")]
        p: f64,
    },
    /// the normal distribution
    Normal {
        #[arg(value_name = "mean")]
        mu: f64,
        #[arg(value_name = "standard deviation")]
        sigma: f64,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Commands::Sample { count, dist } => {
            let count = count.unwrap_or(10);

            match dist {
                DistributionAsCommand::Multinomial { n, p } => {
                    // store as iter of Vec
                    let it = thread_rng()
                        .sample_iter(Multinomial::new(p, n)?)
                        .map(|v| Into::<Vec<_>>::into(v.data));
                    // print as int
                    print_multivariate_samples(
                        count,
                        it.map(|v| v.into_iter().map(|x| x as usize)),
                    )?
                }
                DistributionAsCommand::Binomial { n, p } => {
                    let samples = thread_rng().sample_iter(Binomial::new(p, n)?);
                    print_samples(count, samples)?;
                }
                DistributionAsCommand::Normal { mu, sigma } => {
                    let samples = thread_rng().sample_iter(Normal::new(mu, sigma)?);
                    print_samples(count, samples)?
                }
            }
        }
        x @ Commands::Density { .. } => println!("{x:#?}"),
    }
    println!("");
    Ok(())
}

mod util {
    use std::fmt::Display;
    use std::io::{self, BufWriter, Write};
    pub(super) fn write_interspersed<I, T, W>(
        handle: &mut BufWriter<W>,
        it: I,
        sep: &str,
    ) -> io::Result<()>
    where
        I: IntoIterator<Item = T>,
        T: Display,
        W: Write,
    {
        let mut it = it.into_iter();
        if let Some(i) = it.next() {
            write!(handle, "{i}")?;
            for i in it {
                write!(handle, "{sep}{i}")?;
            }
        }
        Ok(())
    }
}

fn print_multivariate_samples<T, S>(
    count: usize,
    samples: impl IntoIterator<Item = T>,
) -> io::Result<()>
where
    T: IntoIterator<Item = S>,
    S: Display,
{
    let mut handle = io::BufWriter::new(io::stdout());

    for s in samples.into_iter().take(count) {
        util::write_interspersed(&mut handle, s.into_iter(), ", ")?;
        writeln!(&mut handle, "")?;
    }
    Ok(())
}

fn print_samples<T>(count: usize, samples: impl IntoIterator<Item = T>) -> io::Result<()>
where
    T: Display,
{
    let mut handle = io::BufWriter::new(io::stdout());
    util::write_interspersed(&mut handle, samples.into_iter().take(count), "\n")?;
    Ok(())
}
