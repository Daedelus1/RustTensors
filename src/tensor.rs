use crate::address_iterator::AddressIterator;
use crate::adressable::Addressable;
use std::ops::{Add, Index, IndexMut, Sub};

pub trait Tensor<
    T,
    V: Copy + From<u8> + Add<Output = V> + Sub<Output = V> + PartialOrd,
    A: Addressable<V, DIMENSION>,
    const DIMENSION: usize,
>: Index<A, Output = T> + IndexMut<A, Output = T>
{
    fn smallest_contained_address(&self) -> A;
    fn largest_contained_address(&self) -> A;
    /// Attempts to get a reference of the value at the given address. Will return Err if the address
    /// is not contained in the matrix.
    ///
    /// # Arguments
    ///
    /// * `address`: The address of the value to be retrieved
    ///
    /// Returns: Result<&T, String>, the Result-wrapped reference to the value.
    fn get(&self, address: A) -> Result<&T, String> {
        if self.contains_address(address) {
            Ok(&self[address])
        } else {
            Err(format!(
                "Cannot retrieve value at {address:?}, index out of bounds."
            ))
        }
    }
    /// Attempts to get a mutable reference of the value at the given address. Will return Err if the
    /// address is not contained in the matrix.
    ///
    /// # Arguments
    ///
    /// * `address`: The address of the value to be retrieved
    ///
    /// Returns: Result<&T, String>, the Result-wrapped reference to the value.
    fn get_mut(&mut self, address: A) -> Result<&mut T, String> {
        if self.contains_address(address) {
            Ok(&mut self[address])
        } else {
            Err(format!(
                "Cannot retrieve value at {address:?}, index out of bounds."
            ))
        }
    }
    /// Evaluates whether an address is valid and has an associated value in the tensor.
    ///
    /// # Arguments
    ///
    /// * `address`: The address to be evaluated
    ///
    /// Returns: bool, A boolean which is true if and only if the address is valid and has an
    /// associated value.
    fn contains_address(&self, address: A) -> bool {
        !(0..DIMENSION).any(|d| {
            address.get_value_at_dimension_index(d)
                < self
                    .smallest_contained_address()
                    .get_value_at_dimension_index(d)
                || address.get_value_at_dimension_index(d)
                    > self
                        .largest_contained_address()
                        .get_value_at_dimension_index(d)
        })
    }
    /// Creates an iterator over the addresses within the bounds of the tensor.
    ///
    /// The iterator will traverse all addresses starting from the smallest contained address
    /// and ending at the largest contained address, inclusive.
    ///
    /// # Returns
    ///
    /// An instance of `AddressIterator<V, A, DIMENSION>`, initialized to iterate between
    /// the smallest and largest contained addresses of the current object.
    fn address_iter(&self) -> AddressIterator<V, A, DIMENSION> {
        AddressIterator::<V, A, DIMENSION>::new(
            self.smallest_contained_address().into(),
            self.largest_contained_address().into(),
        )
    }
}
