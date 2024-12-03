extern crate statrs;

use nalgebra as na;
use rand::{thread_rng, Rng};
use statrs::distribution::{Binomial, Continuous, Discrete, Multinomial, Normal};
use statrs::statistics::Mode;

use std::fmt::Display;
use std::io::{self, BufWriter, Write};
use std::str::{FromStr, Split};

use anyhow::Result;
use clap::{ArgAction, Parser, Subcommand};

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
    Density {
        /// sample to evaluate density at, default=distribution's mode
        #[arg(short, long = "arg", action = ArgAction::Append, value_name = "SAMPLE", help = "sample(s) evaluate at, (space-delimited string for multivariate)")]
        args: Vec<String>,
        #[command(subcommand)]
        dist: DistributionAsCommand,
    },
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
        #[arg(value_name = "success probability", default_value = "0.5")]
        p: f64,
    },
    /// the normal distribution
    Normal {
        #[arg(value_name = "mean", default_value = "0.0")]
        mu: f64,
        #[arg(value_name = "standard deviation", default_value = "1.0")]
        sigma: f64,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Commands::Sample { count, dist } => run_command_sample(count, dist),
        Commands::Density { args, dist } => run_command_density(&args, dist),
    }?;
    println!();
    Ok(())
}

fn run_command_density(args_str: &[String], dist: DistributionAsCommand) -> Result<()> {
    let densities = match dist {
        DistributionAsCommand::Multinomial { n, p } => {
            let dist = Multinomial::new(p, n)?;
            if !args_str.is_empty() {
                let mut densities = Vec::with_capacity(args_str.len());

                for arg_str in args_str {
                    let arg = parse_str_split_to_vec(arg_str.split(' '));
                    if arg.len() == dist.p().len() {
                        densities.push(dist.pmf(&arg.into()));
                    } else {
                        anyhow::bail!("dimension mismatch after parsing `--arg {arg_str}`");
                    }
                }

                densities
            } else {
                vec![dist.pmf(&dist.mode())]
            }
        }
        DistributionAsCommand::Binomial { n, p } => {
            let dist = Binomial::new(p, n)?;
            if !args_str.is_empty() {
                args_str
                    .iter()
                    .map_while(|s| match s.parse() {
                        Ok(x) => Some(x),
                        Err(e) => {
                            eprintln!("could not parse argment, got {e}");
                            None
                        }
                    })
                    .map(|x| dist.pmf(x))
                    .collect()
            } else {
                vec![dist.pmf(dist.mode().unwrap())]
            }
        }
        DistributionAsCommand::Normal { mu, sigma } => {
            let dist = Normal::new(mu, sigma)?;
            if !args_str.is_empty() {
                args_str
                    .iter()
                    .map_while(|s| match s.parse() {
                        Ok(x) => Some(x),
                        Err(e) => {
                            eprintln!("could not parse argment, got {e}");
                            None
                        }
                    })
                    .map(|x| dist.pdf(x))
                    .collect()
            } else {
                vec![dist.pdf(dist.mode().unwrap())]
            }
        }
    };

    util::write_interspersed(&mut BufWriter::new(io::stdout()), densities, ", ")?;

    Ok(())
}

fn parse_str_split_to_vec<T, E>(sp: Split<char>) -> Vec<T>
where
    T: FromStr<Err = E>,
    E: Display + std::error::Error,
{
    sp.map_while(|si| match si.parse::<T>() {
        Ok(x) => Some(x),
        Err(e) => {
            eprintln!("could not parse argment, got {e}");
            None
        }
    })
    .collect()
}

fn run_command_sample(count: Option<usize>, dist: DistributionAsCommand) -> Result<()> {
    let count = count.unwrap_or(10);

    match dist {
        // multinomial should print `count` of Vec<uint>
        DistributionAsCommand::Multinomial { n, p } => {
            let sample_iter = thread_rng().sample_iter(Multinomial::new(p, n)?);
            print_multivariate_samples(
                count,
                sample_iter.map(|v: na::DVector<u64>| {
                    let vec: Vec<_> = v.into_iter().cloned().collect();
                    vec
                }),
            )?;
        }
        // binomial should print `count` of uint
        DistributionAsCommand::Binomial { n, p } => {
            let sample_iter = thread_rng().sample_iter::<u64, Binomial>(Binomial::new(p, n)?);
            print_samples(count, sample_iter)?;
        }
        // normal should print `count` of float
        DistributionAsCommand::Normal { mu, sigma } => {
            let sample_iter = thread_rng().sample_iter(Normal::new(mu, sigma)?);
            print_samples(count, sample_iter)?
        }
    }

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
        writeln!(&mut handle)?;
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
