extern crate statrs;

use anyhow::{anyhow, Context, Result as AnyhowResult};
use statrs::distribution::{
    Continuous, Discrete, Gamma, GammaError, NegativeBinomial, Normal, ParametrizationError,
};

pub fn main() -> AnyhowResult<()> {
    gamma_pdf(1.0, 1.0, 1.0).map(|x| println!("val = {}", x))?;
    // val = 0.36787944117144233

    normal_pdf(1.0, 1.0, 1.0).map(|x| println!("val = {}", x))?;
    // val = 0.39894228040143265

    gamma_pdf_with_negative_shape_correction(-0.5, 1.0, 1.0).map(|x| println!("val = {}", x))?;
    // without shape correction would emit, the below
    //  Error: failed creating gamma(-0.5,1)
    //
    // Caused by:
    // 0: shape must be finite, positive, and not nan
    // 1: expected positive, got -0.5
    // after re-attempt, output is
    //  Error: gamma provided invalid shape
    // attempting to correct shape to 0.5
    // val = 0.2075537487102974

    nb_pmf(1, 1.0, 1).map(|x| println!("val = {}", x))?;
    //  Error: failed creating nb(1,1)
    //
    // Caused by:
    // mean of 0 is degenerate

    nb_pmf(1, 0., 1).map(|x| println!("val = {}", x))?;
    //  Error: failed creating nb(1,0)
    //
    // Caused by:
    // mean of inf is degenerate

    normal_pdf(1.0, f64::INFINITY, 1.0).map(|x| println!("val = {}", x))?;
    //  Error: failed creating normal(1, inf)
    //
    // Caused by:
    // variance of inf is degenrate

    normal_pdf(1.0, 0.0, 1.0).map(|x| println!("val = {}", x))?;
    //  Error: failed creating normal(1, 0)
    //
    // Caused by:
    // variance of 0 is degenerate

    Ok(())
}

pub fn gamma_pdf(shape: f64, rate: f64, x: f64) -> AnyhowResult<f64> {
    Ok(Gamma::new(shape, rate)
        .context(format!("failed creating gamma({},{})", shape, rate))?
        .pdf(x))
}

pub fn gamma_pdf_with_negative_shape_correction(
    shape: f64,
    rate: f64,
    x: f64,
) -> AnyhowResult<f64> {
    match gamma_pdf(shape, rate, x) {
        Ok(x) => Ok(x),
        Err(ee) => {
            if let GammaError::InvalidShape(e) = ee.downcast::<GammaError>()? {
                eprintln!("Error: gamma provided invalid shape");
                if let ParametrizationError::ExpectedPositive(shape) = e {
                    eprintln!("\tattempting to correct shape to {}", shape.abs());
                    // fails again for 0 and INF
                    gamma_pdf(shape.abs(), rate, x)
                } else {
                    Err(anyhow!("cannot recover valid shape from this error"))
                }
            } else {
                Err(anyhow!(
                    "cannot recover both valid shape and rate from this error"
                ))
            }
        }
    }
}

pub fn nb_pmf(r: u64, p: f64, x: u64) -> AnyhowResult<f64> {
    Ok(NegativeBinomial::new(r, p)
        .context(format!("failed creating nb({},{})", r, p))?
        .pmf(x))
}

pub fn normal_pdf(location: f64, scale: f64, x: f64) -> AnyhowResult<f64> {
    Ok(Normal::new(location, scale)
        .context(format!("failed creating normal({}, {})", location, scale))?
        .pdf(x))
}
