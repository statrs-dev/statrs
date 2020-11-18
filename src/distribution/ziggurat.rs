use super::ziggurat_tables;
use rand::distributions::Open01;
use rand::Rng;

pub fn sample_std_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    #[inline]
    fn pdf(x: f64) -> f64 {
        (-x * x / 2.0).exp()
    }

    #[inline]
    fn zero_case<R: Rng + ?Sized>(rng: &mut R, u: f64) -> f64 {
        let mut x = 1.0f64;
        let mut y = 0.0f64;
        while -2.0 * y < x * x {
            let x_: f64 = rng.sample(Open01);
            let y_: f64 = rng.sample(Open01);

            x = x_.ln() / ziggurat_tables::ZIG_NORM_R;
            y = y_.ln();
        }
        if u < 0.0 {
            x - ziggurat_tables::ZIG_NORM_R
        } else {
            ziggurat_tables::ZIG_NORM_R - x
        }
    }

    ziggurat(
        rng,
        true,
        &ziggurat_tables::ZIG_NORM_X,
        &ziggurat_tables::ZIG_NORM_F,
        pdf,
        zero_case,
    )
}

pub fn sample_exp_1<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    #[inline]
    fn pdf(x: f64) -> f64 {
        (-x).exp()
    }

    #[inline]
    fn zero_case<R: Rng + ?Sized>(rng: &mut R, _u: f64) -> f64 {
        ziggurat_tables::ZIG_EXP_R - rng.gen::<f64>().ln()
    }

    ziggurat(
        rng,
        false,
        &ziggurat_tables::ZIG_EXP_X,
        &ziggurat_tables::ZIG_EXP_F,
        pdf,
        zero_case,
    )
}

// Ziggurat method for sampling a random number based on the ZIGNOR
// variant from Doornik 2005. Code borrowed from
// https://github.com/rust-lang-nursery/rand/blob/master/src/distributions/mod.
// rs#L223
#[inline(always)]
fn ziggurat<R: Rng + ?Sized, P, Z>(
    rng: &mut R,
    symmetric: bool,
    x_tab: ziggurat_tables::ZigTable,
    f_tab: ziggurat_tables::ZigTable,
    mut pdf: P,
    mut zero_case: Z,
) -> f64
where
    P: FnMut(f64) -> f64,
    Z: FnMut(&mut R, f64) -> f64,
{
    const SCALE: f64 = (1u64 << 53) as f64;
    loop {
        let bits: u64 = rng.gen();
        let i = (bits & 0xff) as usize;
        let f = (bits >> 11) as f64 / SCALE;

        // u is either U(-1, 1) or U(0, 1) depending on if this is a
        // symmetric distribution or not.
        let u = if symmetric { 2.0 * f - 1.0 } else { f };
        let x = u * x_tab[i];

        let test_x = if symmetric { x.abs() } else { x };

        // algebraically equivalent to |u| < x_tab[i+1]/x_tab[i] (or u <
        // x_tab[i+1]/x_tab[i])
        if test_x < x_tab[i + 1] {
            return x;
        }
        if i == 0 {
            return zero_case(rng, u);
        }
        // algebraically equivalent to f1 + DRanU()*(f0 - f1) < 1
        if f_tab[i + 1] + (f_tab[i] - f_tab[i + 1]) * rng.gen::<f64>() < pdf(x) {
            return x;
        }
    }
}
