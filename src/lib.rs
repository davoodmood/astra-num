//! # Astra-Num
//!
//! `astra-num` is a Rust library providing an easy-to-use wrapper around
//! [`num-bigint`](https://crates.io/crates/num-bigint), [`num-integer`](https://crates.io/crates/num-integer),
//! and [`num-traits`](https://crates.io/crates/num-traits). It offers additional utilities
//! for handling incredibly large (astronomical) values, cryptography, and more.

use num_bigint::{BigInt, Sign, ToBigInt, ParseBigIntError};
use num_traits::{FromPrimitive, ToPrimitive, Zero, One, Num, Signed};
use num_integer::Integer;
use std::ops::{
    Add, Sub, Mul, Div, Rem, Neg,
    BitAnd, BitOr, BitXor, Not,
    AddAssign, SubAssign, MulAssign, DivAssign, RemAssign
};
use std::str::FromStr;
use std::fmt;
use serde::{Deserialize, Deserializer, Serialize};

#[cfg(feature = "crypto_utils")]
use rand::prelude::*;

/// Represents errors that can occur when working with [`BigNum`].
#[derive(Debug)]
pub enum BigNumError {
    /// Wrapper for parse errors from `num_bigint`.
    ParseBigIntError(ParseBigIntError),
    /// Generic conversion error with an accompanying message.
    ConversionError(String),
}

impl From<ParseBigIntError> for BigNumError {
    fn from(err: ParseBigIntError) -> Self {
        BigNumError::ParseBigIntError(err)
    }
}

/// Represents errors specifically related to parsing [`BigInt`]-formatted values.
#[derive(Debug)]
pub enum BigIntError {
    /// Returned when the string format is invalid.
    InvalidFormat,
    // Additional variants can be added as needed.
}

impl fmt::Display for BigIntError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BigIntError::InvalidFormat => write!(f, "Invalid BigInt format"),
        }
    }
}

impl std::error::Error for BigIntError {}

/// A wrapper struct around `BigInt` providing additional utilities
/// such as modular exponentiation, easy conversions, and more.
/// 
/// # Example
/// 
/// ```
/// use astra_num::BigNum;
/// 
/// let a = BigNum::from(123u32);
/// let b = BigNum::from(456u32);
/// let c = a + b;
/// println!("{}", c); // 579
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BigNum(pub BigInt);

impl BigNum {
    /// Constructs a new [`BigNum`] directly from a [`BigInt`].
    pub fn is(value: BigInt) -> Self {
        BigNum(value)
    }

    /// Creates a [`BigNum`] from a decimal string.
    ///
    /// # Errors
    ///
    /// Returns [`BigIntError::InvalidFormat`] if parsing fails.
    pub fn from_str(s: &str) -> Result<Self, BigIntError> {
        match BigInt::from_str(s) {
            Ok(bigint) => Ok(BigNum(bigint)),
            Err(_) => Err(BigIntError::InvalidFormat),
        }
    }

    /// Converts this [`BigNum`] to a decimal string.
    pub fn to_string(&self) -> String {
        self.0.to_string()
    }

    /// Creates a [`BigNum`] from a little-endian byte slice.
    pub fn from_bytes_le(bytes: &[u8]) -> Self {
        BigNum(BigInt::from_bytes_le(Sign::Plus, bytes))
    }

    /// Converts this [`BigNum`] into a tuple of (Sign, `Vec<u8>`) in little-endian byte order.
    pub fn to_bytes_le(&self) -> (Sign, Vec<u8>) {
        self.0.to_bytes_le()
    }

    /// Creates a [`BigNum`] by parsing a byte slice with the given `radix` in **big-endian** order.
    ///
    /// # Errors
    ///
    /// Returns [`BigNumError::ConversionError`] if parsing fails for the given `radix`.
    pub fn from_radix_be(bytes: &[u8], radix: u32) -> Result<BigNum, BigNumError> {
        match BigInt::parse_bytes(bytes, radix) {
            Some(bigint) => Ok(BigNum(bigint)),
            None => Err(BigNumError::ConversionError(
                "Invalid digit found for the given radix".into(),
            )),
        }
    }

    /// Converts this [`BigNum`] to an `f64`, returning `None` if it doesn't fit.
    pub fn to_f64(&self) -> Option<f64> {
        self.0.to_f64()
    }

    /// Creates a [`BigNum`] from an `f64`. May lose precision or return an error for NaN/infinity.
    ///
    /// # Errors
    ///
    /// Returns [`BigNumError::ConversionError`] if the value is not representable as an integer.
    pub fn from_f64(value: f64) -> Result<BigNum, BigNumError> {
        value.to_bigint()
            .map(BigNum)
            .ok_or(BigNumError::ConversionError(
                "Failed to convert f64 to BigInt".into(),
            ))
    }

    /// Adds two [`BigNum`]s by reference and returns the result.
    pub fn add(&self, other: &BigNum) -> BigNum {
        BigNum(&self.0 + &other.0)
    }

    /// Subtracts two [`BigNum`]s by reference and returns the result.
    pub fn sub(&self, other: &BigNum) -> BigNum {
        BigNum(&self.0 - &other.0)
    }

    /// Multiplies two [`BigNum`]s by reference and returns the result.
    pub fn mul(&self, other: &BigNum) -> BigNum {
        BigNum(&self.0 * &other.0)
    }

    /// Performs integer division (floor) between two [`BigNum`]s by reference and returns the result.
    /// 
    /// # Panics
    ///
    /// May panic if `other` is zero.
    pub fn div(&self, other: &Self) -> Self {
        BigNum(&self.0 / &other.0)
    }

    /// Returns the remainder of dividing this [`BigNum`] by another.
    ///
    /// # Panics
    ///
    /// May panic if `other` is zero.
    pub fn rem(&self, other: &Self) -> Self {
        BigNum(&self.0 % &other.0)
    }

    /// Returns the integer square root (floor) of this [`BigNum`].
    pub fn sqrt(&self) -> BigNum {
        BigNum(self.0.sqrt())
    }

    /// Computes a more precise integer square root using Newton's method.
    ///
    /// # Errors
    ///
    /// Returns an error string if the number is negative (since real sqrt is not defined for negative values).
    pub fn sqrt_precise(&self) -> Result<BigNum, String> {
        if self.0.sign() == Sign::Minus {
            return Err("Square root of negative numbers is not supported.".into());
        }

        if self.0 < BigInt::from(2) {
            return Ok(self.clone());
        }

        let two: BigInt = BigInt::from(2);
        let mut x0 = self.0.clone();
        let mut x1 = (&self.0 / &two) + 1u32;

        while x1 < x0 {
            x0 = x1.clone();
            x1 = ((&self.0 / &x1) + &x1) / &two;
        }
        Ok(BigNum(x0))
    }

    /// Returns the number of bits needed to represent this [`BigNum`].
    pub fn bits(&self) -> u64 {
        self.0.bits()
    }

    /// Raises this [`BigNum`] to the power of `exp` using exponentiation by squaring.
    pub fn pow(&self, exp: u32) -> Self {
        BigNum(self.0.pow(exp))
    }

    /// Computes `(self^exponent) mod modulus` using modular exponentiation.
    pub fn mod_pow(&self, exponent: &Self, modulus: &Self) -> BigNum {
        BigNum(self.0.modpow(&exponent.0, &modulus.0))
    }

    /// Computes the bitwise AND of `self` & `other`.
    pub fn bitand(&self, other: &Self) -> Self {
        BigNum(&self.0 & &other.0)
    }

    /// Computes the bitwise OR of `self` | `other`.
    pub fn bitor(&self, other: &Self) -> Self {
        BigNum(&self.0 | &other.0)
    }

    /// Computes the bitwise XOR of `self` ^ `other`.
    pub fn bitxor(&self, other: &Self) -> Self {
        BigNum(&self.0 ^ &other.0)
    }

    /// Computes the bitwise NOT of `self`.
    pub fn not(&self) -> Self {
        BigNum(!&self.0)
    }

    /// Checks if this [`BigNum`] is zero.
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    /// Returns an immutable reference to the underlying `BigInt`.
    pub fn inner(&self) -> &BigInt {
        &self.0
    }

    // ================================================================
    // Cryptographic Utilities (enabled with `crypto_utils` feature)
    // ================================================================
    #[cfg(feature = "crypto_utils")]
    /// Performs the Miller-Rabin primality test for a given number of `rounds`.
    /// 
    /// The more rounds used, the higher the confidence that the number is prime.
    pub fn probable_prime(&self, rounds: usize) -> bool {
        let n = &self.0;

        if *n == BigInt::one() || *n == BigInt::from(2) {
            return true;
        }
        if n.is_even() {
            return false;
        }

        let mut rng = rand::thread_rng();
        let (d, r) = n.sub(1).unwrap().decompose();

        'outer: for _ in 0..rounds {
            let a = rng.gen_bigint_range(&BigInt::from(2), &(n - 1));
            let mut x = a.modpow(&d, n);

            if x == BigInt::one() || x == n - 1 {
                continue;
            }

            for _ in 0..r {
                x = x.modpow(&BigInt::from(2), n);

                if x == n - 1 {
                    continue 'outer;
                }
            }
            return false;
        }
        true
    }

    #[cfg(feature = "crypto_utils")]
    /// Generates a random prime [`BigNum`] of the given bit size.
    ///
    /// Uses 20 Miller-Rabin rounds for primality checks by default.
    pub fn generate_prime(bits: usize) -> BigNum {
        let mut rng = rand::thread_rng();
        loop {
            let candidate = rng.gen_bigint(bits);
            if candidate.probable_prime(20) {
                return BigNum(candidate);
            }
        }
    }
}

// ================================================================
// Trait Implementations
// ================================================================

impl fmt::Display for BigNum {
    /// Allows printing a [`BigNum`] with `format!` or `println!`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&BigInt> for BigNum {
    fn from(value: &BigInt) -> Self {
        BigNum(value.clone())
    }
}

impl From<&BigNum> for BigNum {
    fn from(value: &BigNum) -> Self {
        value.clone()
    }
}

impl From<i32> for BigNum {
    fn from(num: i32) -> Self {
        BigNum(BigInt::from(num))
    }
}

impl From<u32> for BigNum {
    fn from(num: u32) -> Self {
        BigNum(BigInt::from(num))
    }
}

impl From<u64> for BigNum {
    fn from(num: u64) -> Self {
        BigNum(BigInt::from(num))
    }
}

// Operator Overloads
impl Add for BigNum {
    type Output = BigNum;
    fn add(self, other: Self) -> Self::Output {
        BigNum(self.0 + other.0)
    }
}

impl Sub for BigNum {
    type Output = BigNum;
    fn sub(self, other: Self) -> Self::Output {
        BigNum(self.0 - other.0)
    }
}

impl Mul for BigNum {
    type Output = BigNum;
    fn mul(self, other: Self) -> Self::Output {
        BigNum(self.0 * other.0)
    }
}

impl Mul for &BigNum {
    type Output = BigNum;
    fn mul(self, other: &BigNum) -> BigNum {
        BigNum(&self.0 * &other.0)
    }
}

impl Mul<BigNum> for &BigNum {
    type Output = BigNum;
    fn mul(self, mut other: BigNum) -> BigNum {
        other.0 *= &self.0;
        other
    }
}

impl Mul<&BigNum> for BigNum {
    type Output = BigNum;
    fn mul(mut self, other: &BigNum) -> BigNum {
        self.0 *= &other.0;
        self
    }
}

impl Div for BigNum {
    type Output = BigNum;
    fn div(self, other: Self) -> Self::Output {
        BigNum(self.0 / other.0)
    }
}

impl Rem for BigNum {
    type Output = BigNum;
    fn rem(self, other: Self) -> Self::Output {
        BigNum(self.0 % other.0)
    }
}

impl Neg for BigNum {
    type Output = BigNum;
    fn neg(self) -> Self::Output {
        BigNum(-self.0)
    }
}

impl AddAssign for BigNum {
    fn add_assign(&mut self, other: Self) {
        self.0 += other.0;
    }
}

impl SubAssign for BigNum {
    fn sub_assign(&mut self, other: Self) {
        self.0 -= other.0;
    }
}

impl MulAssign for BigNum {
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

impl DivAssign for BigNum {
    fn div_assign(&mut self, other: Self) {
        self.0 /= other.0;
    }
}

impl RemAssign for BigNum {
    fn rem_assign(&mut self, other: Self) {
        self.0 %= other.0;
    }
}

impl PartialOrd for BigNum {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigNum {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl BitAnd for BigNum {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        BigNum(self.0 & rhs.0)
    }
}

impl BitOr for BigNum {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        BigNum(self.0 | rhs.0)
    }
}

impl BitXor for BigNum {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        BigNum(self.0 ^ rhs.0)
    }
}

impl Not for BigNum {
    type Output = Self;
    fn not(self) -> Self::Output {
        BigNum(!self.0)
    }
}

// Serde Serialization & Deserialization
impl Serialize for BigNum {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for BigNum {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        BigNum::from_str(&s).map_err(serde::de::Error::custom)
    }
}

// Zero and One
impl Zero for BigNum {
    fn zero() -> Self {
        BigNum(BigInt::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl One for BigNum {
    fn one() -> Self {
        BigNum(BigInt::one())
    }
}

// Integer trait implementation
impl Integer for BigNum {
    fn div_floor(&self, other: &Self) -> Self {
        BigNum(&self.0 / &other.0)
    }

    fn mod_floor(&self, other: &Self) -> Self {
        BigNum(self.0.mod_floor(&other.0))
    }

    fn gcd(&self, other: &Self) -> Self {
        BigNum(self.0.gcd(&other.0))
    }

    fn lcm(&self, other: &Self) -> Self {
        BigNum(self.0.lcm(&other.0))
    }

    fn divides(&self, other: &Self) -> bool {
        if other.is_zero() {
            return self.is_zero();
        }
        let r = &other.0 % &self.0;
        r.is_zero()
    }

    fn is_multiple_of(&self, other: &Self) -> bool {
        if self.is_zero() {
            return other.is_zero();
        }
        let r = &self.0 % &other.0;
        r.is_zero()
    }

    fn is_even(&self) -> bool {
        let two = BigInt::from(2);
        (&self.0 % two).is_zero()
    }

    fn is_odd(&self) -> bool {
        !self.is_even()
    }

    fn div_rem(&self, other: &Self) -> (Self, Self) {
        let (div, rem) = self.0.div_rem(&other.0);
        (BigNum(div), BigNum(rem))
    }
}

// Num trait implementation
impl Num for BigNum {
    type FromStrRadixErr = ParseBigIntError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        BigInt::from_str_radix(str, radix).map(BigNum)
    }
}

// Signed trait implementation
impl Signed for BigNum {
    fn abs(&self) -> Self {
        BigNum(self.0.abs())
    }

    fn abs_sub(&self, other: &Self) -> Self {
        if *self <= *other {
            BigNum::zero()
        } else {
            self.sub(other)
        }
    }

    fn signum(&self) -> Self {
        BigNum(self.0.signum())
    }

    fn is_positive(&self) -> bool {
        self.0.is_positive()
    }

    fn is_negative(&self) -> bool {
        self.0.is_negative()
    }
}

// FromPrimitive trait for convenience
impl FromPrimitive for BigNum {
    fn from_i64(n: i64) -> Option<Self> {
        Some(BigNum(BigInt::from(n)))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(BigNum(BigInt::from(n)))
    }
}



// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_str_and_to_string() {
        let bn = BigNum::from_str("12345").expect("Failed to parse BigNum from str");
        assert_eq!(bn.to_string(), "12345");
    }

    #[test]
    fn test_addition() {
        let a = BigNum::from(2i32);
        let b = BigNum::from(3i32);
        let sum = a + b;
        assert_eq!(sum.to_string(), "5");
    }

    #[test]
    fn test_subtraction() {
        let a = BigNum::from(10u32);
        let b = BigNum::from(3u32);
        let difference = a - b;
        assert_eq!(difference.to_string(), "7");
    }

    #[test]
    fn test_multiplication() {
        let a = BigNum::from(6u32);
        let b = BigNum::from(7u32);
        let product = a * b;
        assert_eq!(product.to_string(), "42");
    }

    #[test]
    fn test_division() {
        let a = BigNum::from(24u32);
        let b = BigNum::from(4u32);
        let quotient = a / b;
        assert_eq!(quotient.to_string(), "6");
    }

    #[test]
    fn test_remainder() {
        let a = BigNum::from(29u32);
        let b = BigNum::from(4u32);
        let remainder = a % b;
        assert_eq!(remainder.to_string(), "1");
    }

    #[test]
    fn test_is_zero() {
        let zero_num = BigNum::zero();
        assert!(zero_num.is_zero());

        let non_zero = BigNum::from(5u32);
        assert!(!non_zero.is_zero());
    }

    #[test]
    fn test_sqrt() {
        // sqrt(49) = 7
        let a = BigNum::from(49u32);
        let r = a.sqrt();
        assert_eq!(r.to_string(), "7");
    }

    #[test]
    fn test_sqrt_precise() {
        // sqrt_precise(100) = 10
        let a = BigNum::from(100u32);
        let r = a.sqrt_precise().expect("Failed to compute sqrt_precise");
        assert_eq!(r.to_string(), "10");
    }

    #[test]
    fn test_pow() {
        // 2^10 = 1024
        let base = BigNum::from(2u32);
        let result = base.pow(10);
        assert_eq!(result.to_string(), "1024");
    }

    #[test]
    fn test_bits() {
        // 15 in binary is 1111 => 4 bits
        let a = BigNum::from(15u32);
        assert_eq!(a.bits(), 4);
    }
}