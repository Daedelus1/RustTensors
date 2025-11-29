# Rust Tensors

A lightweight, safe tensor library for Rust.
This includes an interface to make any dimensional arrays (e.g., matrices, lists, triple-nested arrays, etc).
There is no unsafe code and no non-test dependencies.
This crate includes an implementation of a matrix and matrix address type, and you can use these to make your own
tensors.

## Usage

To use, simply add the following to your cargo.toml file:

```toml
[dependencies]
rust_tensors = "0.3.0"
```

This will allow you to use the traits to make your own arbitrary dimensional arrays.

