// (C) Copyright John Maddock 2006.
// (C) Copyright Matt Borland 2024.
// SPDX-License-Identifier: MIT AND BSL-1.0
// Use, modification and distribution are subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE-BOOST.md or copy at
// https://www.boost.org/LICENSE_1_0.txt)
// Adapted from the 53-bit erfc approximation in Boost.Math 1.90 erf.hpp.

#[cfg(all(not(feature = "std"), not(test)))]
use super::Float;
use super::two_sum;

fn evaluate(argument: f64, coefficients: &[f64]) -> f64 {
    coefficients.iter().rev().fold(0.0, |value, coefficient| {
        value.mul_add(argument, *coefficient)
    })
}

pub(super) fn normal_tail(argument: (f64, f64)) -> f64 {
    let (argument_value, argument_error) = two_sum(argument.0, argument.1);
    let value = if argument_value < 0.5 {
        let squared = argument_value * argument_value;
        let numerator = evaluate(
            squared,
            &[
                0.08343058921465318,
                -0.33816513445936094,
                -0.050999073514677746,
                -0.007727583458021333,
                -0.0003227801209646057,
            ],
        );
        let denominator = evaluate(
            squared,
            &[
                1.0,
                0.455004033050794,
                0.08752226001422525,
                0.008585719250744063,
                0.000370900071787748,
            ],
        );
        let erf = argument_value * (1.0449485778808594 + numerator / denominator);
        0.5 - 0.5 * erf
    } else {
        normal_tail_rational(argument_value)
    };
    let derivative = -(-argument_value * argument_value).exp() / core::f64::consts::PI.sqrt();
    let corrected = two_sum(value, derivative * argument_error);
    corrected.0 + corrected.1
}

fn normal_tail_rational(argument: f64) -> f64 {
    let (offset, numerator, denominator, constant) = if argument < 1.5 {
        (
            argument - 0.5,
            &[
                -0.09809059221628124,
                0.17811466584112034,
                0.19100369579677543,
                0.08889003689678845,
                0.01950490012512188,
                0.0018042453829701422,
            ][..],
            &[
                1.0,
                1.8475907098300222,
                1.4262800484551132,
                0.5780528048899024,
                0.12385097467900864,
                0.011338523357700142,
                0.0000033751147248309468,
            ][..],
            0.40593576431274414,
        )
    } else if argument < 2.5 {
        (
            argument - 1.5,
            &[
                -0.024350047620769844,
                0.03865403750357072,
                0.04394818964209516,
                0.01756794363118021,
                0.0032396240629084213,
                0.00023583911559688072,
            ][..],
            &[
                1.0,
                1.5399149494855245,
                0.9824037091579202,
                0.32573292478244445,
                0.056392183742047816,
                0.004103697239789046,
            ][..],
            0.5067281723022461,
        )
    } else {
        (
            argument - 3.5,
            &[
                0.0029527671653097166,
                0.013738442589635533,
                0.008408076155555854,
                0.0021282562091461865,
                0.00025026996154479463,
                0.000011321240664884757,
            ][..],
            &[
                1.0,
                1.0421781416693842,
                0.4425976594815631,
                0.09584927263010614,
                0.010598290648487653,
                0.0004794112695217145,
            ][..],
            0.5405750274658203,
        )
    };
    let rational = constant + evaluate(offset, numerator) / evaluate(offset, denominator);
    let high = (argument * 67_108_864.0).trunc() / 67_108_864.0;
    let low = argument - high;
    let squared = argument * argument;
    let squared_error = (high * high - squared) + 2.0 * high * low + low * low;
    0.5 * rational * (-squared).exp() * (-squared_error).exp() / argument
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundaries_match_500_digit_references() {
        let cases = [
            (0x3fdfffffffffffff_u64, 0x3fceb02147ce245d_u64),
            (0x3fe0000000000000, 0x3fceb02147ce245c),
            (0x3fe0000000000001, 0x3fceb02147ce245a),
            (0x3ff7ffffffffffff, 0x3f915aaa8ec85209),
            (0x3ff8000000000000, 0x3f915aaa8ec85205),
            (0x3ff8000000000001, 0x3f915aaa8ec85201),
            (0x4003ffffffffffff, 0x3f2aab859b20acb0),
            (0x4004000000000000, 0x3f2aab859b20ac9e),
            (0x4004000000000001, 0x3f2aab859b20ac8d),
        ];
        for (argument, expected) in cases {
            let actual = normal_tail((f64::from_bits(argument), 0.0)).to_bits();
            assert!(
                actual.abs_diff(expected) <= 4,
                "argument={argument:#018x}, actual={actual:#018x}, expected={expected:#018x}"
            );
        }
    }

    #[test]
    fn double_double_arguments_match_500_digit_references() {
        let cases = [
            (
                0x3fd0000000000000_u64,
                2.0_f64.powi(-55),
                0x3fd728558ee694fb_u64,
            ),
            (0x3fd0000000000000, -2.0_f64.powi(-55), 0x3fd728558ee694fc),
            (0x3ff8000000000000, 2.0_f64.powi(-53), 0x3f915aaa8ec85203),
            (
                0x400408614bd1f138,
                f64::from_bits(0xbcb38087e4245eb0),
                0x3f2a178104a215c2,
            ),
            (
                0x4006a2ae09a6ce40,
                f64::from_bits(0xbc8cd2b297d889bc),
                0x3f0081598e5e54ed,
            ),
        ];
        for (argument, error, expected) in cases {
            let actual = normal_tail((f64::from_bits(argument), error)).to_bits();
            assert!(
                actual.abs_diff(expected) <= 4,
                "argument={argument:#018x}, error={error:?}, actual={actual:#018x}, expected={expected:#018x}"
            );
        }
    }
}
