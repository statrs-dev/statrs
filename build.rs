//! Detects whether the target has a hardware fused multiply-add.
//!
//! `f64::mul_add` is always correct, but on a target without the instruction it
//! calls a software routine that is far slower than doing the same job with
//! Veltkamp's split. `src/prec.rs` picks between the two on `statrs_hardware_fma`.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(statrs_hardware_fma)");
    println!("cargo::rerun-if-changed=build.rs");

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features: Vec<String> = std::env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap_or_default()
        .split(',')
        .map(str::to_owned)
        .collect();
    let has = |f: &str| features.iter().any(|x| x == f);

    // Each arm is "the FMA instruction is present", not "the architecture
    // usually has it": AArch64 without NEON is softfloat, and baseline x86-64
    // needs -C target-feature=+fma or a -C target-cpu implying it.
    let hardware_fma = match arch.as_str() {
        "aarch64" | "arm64ec" => has("neon"),
        "x86" | "x86_64" => has("fma"),
        "arm" => has("vfp4"),
        "riscv32" | "riscv64" => has("d") || has("f"),
        "powerpc" | "powerpc64" => has("vsx") || has("altivec"),
        "loongarch32" | "loongarch64" => has("d") || has("f"),
        "mips" | "mips64" | "mips32r6" | "mips64r6" => has("fp64"),
        "wasm32" | "wasm64" => has("relaxed-simd"),
        _ => false,
    };

    if hardware_fma {
        println!("cargo::rustc-cfg=statrs_hardware_fma");
    }
}
