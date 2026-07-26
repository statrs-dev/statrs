use statrs::function::erf::ln_erfc;
fn main() {
    for i in 0..=800 {
        let x = -6.0 + (206.0) * (i as f64) / 800.0; // covers <0.5, rational range, boundary at 110, asymptotic
        println!("{:016x}\t{:016x}", x.to_bits(), ln_erfc(x).to_bits());
    }
    for x in [1e3f64, 1e5, 1e10, 1e77, 1.3e154, 1.4e154, 1e200] {
        println!("{:016x}\t{:016x}", x.to_bits(), ln_erfc(x).to_bits());
    }
}
