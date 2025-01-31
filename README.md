# Astra-Num

Astra-Num is a Rust library providing an easy-to-use wrapper around [num-bigint], [num-integer], and [num-traits]. It offers additional utilities for handling incredibly large (astronomical) values, cryptographic operations (optional), and more. This library is suitable for projects that require arbitrary-precision arithmetic or advanced integer manipulation.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Example](#basic-example)
  - [Arithmetic Operations](#arithmetic-operations)
  - [Conversion Examples](#conversion-examples)
  - [Crypto Utilities (Feature: `crypto_utils`)](#crypto-utilities-feature-crypto_utils)
- [Documentation](#documentation)
- [Testing](#testing)
- [License](#license)

---

## Features

1. **Arbitrary-precision arithmetic** based on [`num_bigint::BigInt`].
2. **Trait implementations** for [`num_traits`]—allows you to use familiar methods like `Zero`, `One`, `Num`, and `Signed`.
3. **Additional arithmetic methods**: 
   - `mod_pow` for efficient modular exponentiation.
   - `sqrt`, `sqrt_precise` for integer square roots.
   - Bitwise operations like AND, OR, XOR, NOT.
4. **Optional cryptographic utilities** (enabled via the `crypto_utils` feature) including:
   - **Miller-Rabin primality check** for probable prime testing.
   - **Random prime generation** of a given bit size.
5. **Serde support** for easy serialization/deserialization of `BigNum` as decimal strings.

---

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
astra-num = "0.1.0"
```

If you want the cryptographic utilities (prime checking, prime generation), enable the `crypto_utils` feature:

```toml
[dependencies]
astra-num = { version = "0.1.0", features = ["crypto_utils"] }
```

Then import in your Rust code:
```rust
use astra_num::BigNum;
```

## Usage
### Basic Example
```rust
use astra_num::BigNum;

fn main() {
    let a = BigNum::from(123u32);
    let b = BigNum::from(456u32);
    let result = a + b;
    println!("Sum = {}", result); // 579
}
```
### Arithmetic Operations
BigNum supports common operations through both method calls and Rust’s operator overloads:

```rust
use astra_num::BigNum;

// addition: operator
let a = BigNum::from(100u32);
let b = BigNum::from(25u32);
let sum = a.clone() + b.clone(); // 125

// subtraction: method
let diff = a.sub(&b);           // 75

// multiplication: operator
let product = a * b;            // 2500

// division & remainder
let quotient = BigNum::from(24u32) / BigNum::from(4u32); // 6
let remainder = BigNum::from(29u32) % BigNum::from(4u32);// 1

println!("Sum = {}", sum);
println!("Diff = {}", diff);
println!("Product = {}", product);
println!("Quotient = {}", quotient);
println!("Remainder = {}", remainder);

```

### Conversion Examples

```rust
use astra_num::BigNum;
use num_bigint::Sign;

// from decimal string
let bn_decimal = BigNum::from_str("999999999").expect("Invalid format");
println!("Decimal parse: {}", bn_decimal); // 999999999

// from/to little-endian bytes
let data = &[0x01, 0x02, 0x03]; // example
let bn_le = BigNum::from_bytes_le(data);
let (sign, bytes) = bn_le.to_bytes_le();
assert_eq!(sign, Sign::Plus);
assert_eq!(bytes, data);

// from f64
let approximate = BigNum::from_f64(12345.0).unwrap();
println!("From f64 = {}", approximate); // 12345

// to f64 (returns an Option<f64>)
let maybe_float = approximate.to_f64();
if let Some(f) = maybe_float {
    println!("As float = {}", f); // 12345.0
} else {
    println!("Value too large for f64!");
}

```
### Crypto Utilities (Feature: `crypto_utils`)

```rust
// In Cargo.toml:
// [dependencies]
// astra-num = { version = "0.1.0", features = ["crypto_utils"] }

use astra_num::BigNum;

fn main() {
    // 1. Probable prime check using Miller-Rabin
    let num = BigNum::from(7919u32);
    let is_prime = num.probable_prime(20); 
    println!("Is 7919 prime? {}", is_prime); // Likely true

    // 2. Generate a random prime of bit length 128
    let prime_128 = BigNum::generate_prime(128);
    println!("128-bit prime = {}", prime_128);
}

```
**Security Notice:** While these utilities are educational and convenient, cryptography requires careful review and best practices (e.g., secure random number generation, safe prime generation parameters, etc.). For production-grade cryptographic applications, consider established, audited libraries.


## Documentation

Comprehensive documentation for **astra-num** is hosted on [docs.rs](https://docs.rs/astra-num).

To build it locally, run:

```bash
cargo doc --open
```
This command will generate and open documentation in your web browser.

## Testing
I included a suite of unit tests in the `lib.rs` file. You can run them with:

```bash
cargo test
```
This will:

- Build the library in test mode, and
- Execute all the tests, reporting any failures or errors.

## License
This project is licensed under the **MIT license**. You are free to use, modify, and distribute this software, subject to the terms of the license.