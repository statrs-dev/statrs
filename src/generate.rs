//! Provides utility functions for generating data sequences

use euclid::Modulus;
use std::f64::consts;
use std::iter::Take;

/// Generates a base 10 log spaced vector of the given length between the
/// specified decade exponents (inclusive). Equivalent to MATLAB logspace
///
/// # Examples
///
/// ```
/// use statrs::generate;
///
/// let x = generate::log_spaced(5, 0.0, 4.0);
/// assert_eq!(x, [1.0, 10.0, 100.0, 1000.0, 10000.0]);
/// ```
pub fn log_spaced(length: usize, start_exp: f64, stop_exp: f64) -> Vec<f64> {
    match length {
        0 => Vec::new(),
        1 => vec![10f64.powf(stop_exp)],
        _ => {
            let step = (stop_exp - start_exp) / (length - 1) as f64;
            let mut vec = (0..length)
                .map(|x| 10f64.powf(start_exp + (x as f64) * step))
                .collect::<Vec<f64>>();
            vec[length - 1] = 10f64.powf(stop_exp);
            vec
        }
    }
}

/// Infinite iterator returning floats that form a periodic wave
pub struct InfinitePeriodic {
    amplitude: f64,
    step: f64,
    phase: f64,
    k: f64,
}

impl InfinitePeriodic {
    /// Constructs a new infinite periodic wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::InfinitePeriodic;
    ///
    /// let x = InfinitePeriodic::new(8.0, 2.0, 10.0, 1.0,
    /// 2).take(10).collect::<Vec<f64>>();
    /// assert_eq!(x, [6.0, 8.5, 1.0, 3.5, 6.0, 8.5, 1.0, 3.5, 6.0, 8.5]);
    /// ```
    pub fn new(sampling_rate: f64, frequency: f64, amplitude: f64, phase: f64, delay: i64) -> InfinitePeriodic {

        let step = frequency / sampling_rate * amplitude;
        InfinitePeriodic {
            amplitude: amplitude,
            step: step,
            phase: (phase - delay as f64 * step).modulus(amplitude),
            k: 0.0,
        }
    }

    /// Constructs a default infinite periodic wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::InfinitePeriodic;
    ///
    /// let x = InfinitePeriodic::default(8.0,
    /// 2.0).take(10).collect::<Vec<f64>>();
    /// assert_eq!(x, [0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5, 0.75, 0.0, 0.25]);
    /// ```
    pub fn default(sampling_rate: f64, frequency: f64) -> InfinitePeriodic {
        Self::new(sampling_rate, frequency, 1.0, 0.0, 0)
    }
}

impl Iterator for InfinitePeriodic {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        let mut x = self.phase + self.k * self.step;
        if x >= self.amplitude {
            x %= self.amplitude;
            self.phase = x;
            self.k = 0.0;
        }
        self.k += 1.0;
        Some(x)
    }
}

/// Finite iterator returning floats that form a periodic wave
pub struct Periodic {
    internal: Take<InfinitePeriodic>,
}

impl Periodic {
    /// Constructs a new periodic wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::Periodic;
    ///
    /// let x = Periodic::new(10, 8.0, 2.0, 10.0, 1.0, 2).collect::<Vec<f64>>();
    /// assert_eq!(x, [6.0, 8.5, 1.0, 3.5, 6.0, 8.5, 1.0, 3.5, 6.0, 8.5]);
    /// ```
    #[deprecated(since = "0.9.0", note = "please use `InfinitePeriodic::new` and `take` instead")]
    pub fn new(length: usize, sampling_rate: f64, frequency: f64, amplitude: f64, phase: f64, delay: i64) -> Periodic {
        Periodic { internal: InfinitePeriodic::new(sampling_rate, frequency, amplitude, phase, delay).take(length) }
    }

    /// Constructs a default periodic wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::Periodic;
    ///
    /// let x = Periodic::default(10, 8.0, 2.0).collect::<Vec<f64>>();
    /// assert_eq!(x, [0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5, 0.75, 0.0, 0.25]);
    /// ```
    #[deprecated(since = "0.9.0", note = "please use `InfinitePeriodic::default` and `take` instead")]
    pub fn default(length: usize, sampling_rate: f64, frequency: f64) -> Periodic {
        Periodic { internal: InfinitePeriodic::default(sampling_rate, frequency).take(length) }
    }
}

impl Iterator for Periodic {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        self.internal.next()
    }
}

/// Creates a vector of `f64` points representing a periodic wave with an
/// amplitude of `1.0`, phase of `0.0`, and delay of `0`.
///
/// # Examples
///
/// ```
/// use statrs::generate;
///
/// let x = generate::periodic(10, 8.0, 2.0);
/// assert_eq!(x, [0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5, 0.75, 0.0, 0.25]);
/// ```
#[deprecated(since = "0.8.0", note = "please use `Periodic::default` instead")]
pub fn periodic(length: usize, sampling_rate: f64, frequency: f64) -> Vec<f64> {
    InfinitePeriodic::default(sampling_rate, frequency)
        .take(length)
        .collect::<Vec<f64>>()
}

/// Creates a vector of `f64` points representing a periodic wave.
///
/// # Examples
///
/// ```
/// use statrs::generate;
///
/// let x = generate::periodic_custom(10, 8.0, 2.0, 10.0, 1.0, 2);
/// assert_eq!(x, [6.0, 8.5, 1.0, 3.5, 6.0, 8.5, 1.0, 3.5, 6.0, 8.5]);
/// ```
#[deprecated(since = "0.8.0", note = "please use `Periodic::new` instead")]
pub fn periodic_custom(length: usize, sampling_rate: f64, frequency: f64, amplitude: f64, phase: f64, delay: i64) -> Vec<f64> {

    let step = frequency / sampling_rate * amplitude;
    let mut phase = (phase - delay as f64 * step).modulus(amplitude);
    let mut data = vec![0.0; length];
    let mut k = 0.0;

    for d in &mut data {
        let mut x = phase + k * step;
        if x >= amplitude {
            x %= amplitude;
            phase = x;
            k = 0.0;
        }
        *d = x;
        k += 1.0;
    }
    data
}

/// Infinite iterator returning floats that form a sinusoidal wave
pub struct InfiniteSinusoidal {
    amplitude: f64,
    mean: f64,
    step: f64,
    phase: f64,
    i: usize,
}

impl InfiniteSinusoidal {
    /// Constructs a new infinite sinusoidal wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::InfiniteSinusoidal;
    ///
    /// let x = InfiniteSinusoidal::new(8.0, 2.0, 1.0, 5.0, 2.0,
    /// 1).take(10).collect::<Vec<f64>>();
    /// assert_eq!(x,
    ///     [5.416146836547142, 5.909297426825682, 4.583853163452858,
    ///     4.090702573174318, 5.416146836547142, 5.909297426825682,
    ///     4.583853163452858, 4.090702573174318, 5.416146836547142,
    ///     5.909297426825682]);
    /// ```
    pub fn new(sampling_rate: f64, frequency: f64, amplitude: f64, mean: f64, phase: f64, delay: i64) -> InfiniteSinusoidal {

        let pi2 = consts::PI * 2.0;
        let step = frequency / sampling_rate * pi2;
        InfiniteSinusoidal {
            amplitude: amplitude,
            mean: mean,
            step: step,
            phase: (phase - delay as f64 * step) % pi2,
            i: 0,
        }
    }

    /// Constructs a default infinite sinusoidal wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::InfiniteSinusoidal;
    ///
    /// let x = InfiniteSinusoidal::default(8.0, 2.0,
    /// 1.0).take(10).collect::<Vec<f64>>();
    /// assert_eq!(x,
    ///     [0.0, 1.0, 0.00000000000000012246467991473532,
    ///     -1.0, -0.00000000000000024492935982947064, 1.0,
    ///     0.00000000000000036739403974420594, -1.0,
    ///     -0.0000000000000004898587196589413, 1.0]);
    /// ```
    pub fn default(sampling_rate: f64, frequency: f64, amplitude: f64) -> InfiniteSinusoidal {
        Self::new(sampling_rate, frequency, amplitude, 0.0, 0.0, 0)
    }
}

impl Iterator for InfiniteSinusoidal {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        let x = self.mean + self.amplitude * (self.phase + self.i as f64 * self.step).sin();
        self.i += 1;
        if self.i == 1000 {
            self.i = 0;
            self.phase = (self.phase + 1000.0 * self.step) % (consts::PI * 2.0);
        }
        Some(x)
    }
}

/// Finite iterator returning floats that form a sinusoidal wave
pub struct Sinusoidal {
    internal: Take<InfiniteSinusoidal>,
}

impl Sinusoidal {
    /// Constructs a new sinusoidal wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::Sinusoidal;
    ///
    /// let x = Sinusoidal::new(10, 8.0, 2.0, 1.0, 5.0, 2.0,
    /// 1).collect::<Vec<f64>>();
    /// assert_eq!(x,
    ///     [5.416146836547142, 5.909297426825682, 4.583853163452858,
    ///     4.090702573174318, 5.416146836547142, 5.909297426825682,
    ///     4.583853163452858, 4.090702573174318, 5.416146836547142,
    ///     5.909297426825682]);
    /// ```
    #[deprecated(since = "0.9.0", note = "please use `InfiniteSinusoidal::new` and `take` instead")]
    pub fn new(length: usize, sampling_rate: f64, frequency: f64, amplitude: f64, mean: f64, phase: f64, delay: i64) -> Sinusoidal {
        Sinusoidal { internal: InfiniteSinusoidal::new(sampling_rate, frequency, amplitude, mean, phase, delay).take(length) }
    }

    /// Constructs a default sinusoidal wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::Sinusoidal;
    ///
    /// let x = Sinusoidal::default(10, 8.0, 2.0, 1.0).collect::<Vec<f64>>();
    /// assert_eq!(x,
    ///     [0.0, 1.0, 0.00000000000000012246467991473532,
    ///     -1.0, -0.00000000000000024492935982947064, 1.0,
    ///     0.00000000000000036739403974420594, -1.0,
    ///     -0.0000000000000004898587196589413, 1.0]);
    /// ```
    #[deprecated(since = "0.9.0", note = "please use `InfiniteSinusoidal::default` and `take` instead")]
    pub fn default(length: usize, sampling_rate: f64, frequency: f64, amplitude: f64) -> Sinusoidal {
        Sinusoidal { internal: InfiniteSinusoidal::default(sampling_rate, frequency, amplitude).take(length) }
    }
}

impl Iterator for Sinusoidal {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        self.internal.next()
    }
}

/// Creates a vector of `f64` points representing a Sine wave with a mean of
/// `0.0`, phase of `0.0`, and delay of `0`.
///
/// # Examples
///
/// ```
/// use statrs::generate;
///
/// let x = generate::sinusoidal(10, 8.0, 2.0, 1.0);
/// assert_eq!(x,
///     [0.0, 1.0, 0.00000000000000012246467991473532,
///     -1.0, -0.00000000000000024492935982947064, 1.0,
///     0.00000000000000036739403974420594, -1.0,
///     -0.0000000000000004898587196589413, 1.0]);
/// ```
#[deprecated(since = "0.8.0", note = "please use `Sinusoidal::default` instead")]
pub fn sinusoidal(length: usize, sampling_rate: f64, frequency: f64, amplitude: f64) -> Vec<f64> {
    InfiniteSinusoidal::new(sampling_rate, frequency, amplitude, 0.0, 0.0, 0)
        .take(length)
        .collect::<Vec<f64>>()
}

/// Creates a vector of `f64` points representing a Sine wave.
///
/// # Examples
///
/// ```
/// use statrs::generate;
///
/// let x = generate::sinusoidal_custom(10, 8.0, 2.0, 1.0, 5.0, 2.0, 1);
/// assert_eq!(x,
///     [5.416146836547142, 5.909297426825682, 4.583853163452858,
///     4.090702573174318, 5.416146836547142, 5.909297426825682,
///     4.583853163452858, 4.090702573174318, 5.416146836547142,
///     5.909297426825682]);
/// ```
#[deprecated(since = "0.8.0", note = "please use `Sinusoidal::new` instead")]
pub fn sinusoidal_custom(length: usize, sampling_rate: f64, frequency: f64, amplitude: f64, mean: f64, phase: f64, delay: i64) -> Vec<f64> {

    let pi2 = consts::PI * 2.0;
    let step = frequency / sampling_rate * pi2;
    let phase = (phase - delay as f64 * step) % pi2;
    (0..length)
        .map(|i| mean + amplitude * (phase + i as f64 * step).sin())
        .collect()
}

/// Infinite iterator returning floats forming a square wave starting
/// with the high phase
pub struct InfiniteSquare {
    periodic: InfinitePeriodic,
    high_duration: f64,
    high_value: f64,
    low_value: f64,
}

impl InfiniteSquare {
    /// Constructs a new infinite square wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::InfiniteSquare;
    ///
    /// let x = InfiniteSquare::new(3, 7, 1.0, -1.0,
    /// 1).take(12).collect::<Vec<f64>>();
    /// assert_eq!(x, [-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    /// -1.0, 1.0])
    /// ```
    pub fn new(high_duration: i64, low_duration: i64, high_value: f64, low_value: f64, delay: i64) -> InfiniteSquare {

        let duration = (high_duration + low_duration) as f64;
        InfiniteSquare {
            periodic: InfinitePeriodic::new(1.0, 1.0 / duration, duration, 0.0, delay),
            high_duration: high_duration as f64,
            high_value: high_value,
            low_value: low_value,
        }
    }
}

impl Iterator for InfiniteSquare {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        self.periodic.next().and_then(|x| {
            if x < self.high_duration {
                Some(self.high_value)
            } else {
                Some(self.low_value)
            }
        })
    }
}

/// Finite iterator returning floats forming a square wave starting
/// with the high phase
pub struct Square {
    internal: Take<InfiniteSquare>,
}

impl Square {
    /// Constructs a new square wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::Square;
    ///
    /// let x = Square::new(12, 3, 7, 1.0, -1.0, 1).collect::<Vec<f64>>();
    /// assert_eq!(x, [-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    /// -1.0, 1.0])
    /// ```
    #[deprecated(since = "0.9.0", note = "please use `InfiniteSquare::new` and `take` instead")]
    pub fn new(length: usize, high_duration: i64, low_duration: i64, high_value: f64, low_value: f64, delay: i64) -> Square {
        Square { internal: InfiniteSquare::new(high_duration, low_duration, high_value, low_value, delay).take(length) }
    }
}

impl Iterator for Square {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        self.internal.next()
    }
}

/// Infinite iterator returning floats forming a triangle wave starting with
/// the raise phase from the lowest sample
pub struct InfiniteTriangle {
    periodic: InfinitePeriodic,
    raise_duration: f64,
    raise: f64,
    fall: f64,
    high_value: f64,
    low_value: f64,
}

impl InfiniteTriangle {
    /// Constructs a new infinite triangle wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use statrs::generate::InfiniteTriangle;
    ///
    /// # fn main() {
    /// let x = InfiniteTriangle::new(4, 7, 1.0, -1.0,
    /// 1).take(12).collect::<Vec<f64>>();
    /// let expected: [f64; 12] = [-0.714, -1.0, -0.5, 0.0, 0.5, 1.0, 0.714,
    /// 0.429, 0.143, -0.143, -0.429, -0.714];
    /// for (&left, &right) in x.iter().zip(expected.iter()) {
    ///     assert_almost_eq!(left, right, 1e-3);
    /// }
    /// # }
    /// ```
    pub fn new(raise_duration: i64, fall_duration: i64, high_value: f64, low_value: f64, delay: i64) -> InfiniteTriangle {

        let duration = (raise_duration + fall_duration) as f64;
        let height = high_value - low_value;
        InfiniteTriangle {
            periodic: InfinitePeriodic::new(1.0, 1.0 / duration, duration, 0.0, delay),
            raise_duration: raise_duration as f64,
            raise: height / raise_duration as f64,
            fall: height / fall_duration as f64,
            high_value: high_value,
            low_value: low_value,
        }
    }
}

impl Iterator for InfiniteTriangle {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        self.periodic.next().and_then(|x| {
            if x < self.raise_duration {
                Some(self.low_value + x * self.raise)
            } else {
                Some(self.high_value - (x - self.raise_duration) * self.fall)
            }
        })
    }
}


/// Finite iterator returning floats forming a triangle wave
/// starting with the raise phase from the lowest sample
pub struct Triangle {
    internal: Take<InfiniteTriangle>,
}

impl Triangle {
    /// Constructs a new triangle wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// #[macro_use]
    /// extern crate statrs;
    ///
    /// use statrs::generate::Triangle;
    ///
    /// # fn main() {
    /// let x = Triangle::new(12, 4, 7, 1.0, -1.0, 1).collect::<Vec<f64>>();
    /// let expected: [f64; 12] = [-0.714, -1.0, -0.5, 0.0, 0.5, 1.0, 0.714,
    /// 0.429, 0.143, -0.143, -0.429, -0.714];
    /// for (&left, &right) in x.iter().zip(expected.iter()) {
    ///     assert_almost_eq!(left, right, 1e-3);
    /// }
    /// # }
    /// ```
    #[deprecated(since = "0.9.0", note = "please use `InfiniteTriangle::new` and `take` instead")]
    pub fn new(length: usize, raise_duration: i64, fall_duration: i64, high_value: f64, low_value: f64, delay: i64) -> Triangle {
        Triangle { internal: InfiniteTriangle::new(raise_duration, fall_duration, high_value, low_value, delay).take(length) }
    }
}

impl Iterator for Triangle {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        self.internal.next()
    }
}

/// Infinite iterator returning floats forming a sawtooth wave
/// starting with the lowest sample
pub struct InfiniteSawtooth {
    periodic: InfinitePeriodic,
    low_value: f64,
}

impl InfiniteSawtooth {
    /// Constructs a new infinite sawtooth wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::InfiniteSawtooth;
    ///
    /// let x = InfiniteSawtooth::new(5, 1.0, -1.0,
    /// 1).take(12).collect::<Vec<f64>>();
    /// assert_eq!(x, [1.0, -1.0, -0.5, 0.0, 0.5, 1.0, -1.0, -0.5, 0.0, 0.5,
    /// 1.0, -1.0]);
    /// ```
    pub fn new(period: i64, high_value: f64, low_value: f64, delay: i64) -> InfiniteSawtooth {

        let height = high_value - low_value;
        let period = period as f64;
        InfiniteSawtooth {
            periodic: InfinitePeriodic::new(
                1.0,
                1.0 / period,
                height * period / (period - 1.0),
                0.0,
                delay,
            ),
            low_value: low_value as f64,
        }
    }
}

impl Iterator for InfiniteSawtooth {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        self.periodic.next().and_then(|x| {
            Some(x + self.low_value)
        })
    }
}

/// Finite iterator returning floats forming a sawtooth wave
/// starting with the lowest sample
pub struct Sawtooth {
    internal: Take<InfiniteSawtooth>,
}

impl Sawtooth {
    /// Constructs a new sawtooth wave generator
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::generate::Sawtooth;
    ///
    /// let x = Sawtooth::new(12, 5, 1.0, -1.0, 1).collect::<Vec<f64>>();
    /// assert_eq!(x, [1.0, -1.0, -0.5, 0.0, 0.5, 1.0, -1.0, -0.5, 0.0, 0.5,
    /// 1.0, -1.0]);
    /// ```
    #[deprecated(since = "0.9.0", note = "please use `InfiniteSawtooth::new` and `take` instead")]
    pub fn new(length: usize, period: i64, high_value: f64, low_value: f64, delay: i64) -> Sawtooth {
        Sawtooth { internal: InfiniteSawtooth::new(period, high_value, low_value, delay).take(length) }
    }
}

impl Iterator for Sawtooth {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        self.internal.next()
    }
}
