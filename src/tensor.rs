use crate::address_iterator::AddressValueIterator;
use crate::adressable::Addressable;
use crate::{address_iterator::AddressIterator, adressable::AddressValue};
use std::ops::{Index, IndexMut};

pub trait Tensor<'a, T: 'a, V: AddressValue, A: Addressable<V, DIMENSION>, const DIMENSION: usize>:
    Index<A, Output = T> + IndexMut<A, Output = T>
{
    fn smallest_contained_address(&self) -> A;
    fn largest_contained_address(&self) -> A;
    /// Attempts to get a reference of the value at the given address. Will return `None` if the address
    /// is not contained in the matrix.
    ///
    /// # Arguments
    ///
    /// * `address`: The address of the value to be retrieved
    ///
    /// Returns: `Option<&T>`, A reference to the value if it exists.
    fn get(&self, address: A) -> Option<&T> {
        if self.contains_address(address) {
            Some(&self[address])
        } else {
            None
        }
    }
    /// Attempts to get a mutable reference of the value at the given address. Will return `None` if the
    /// address is not contained in the matrix.
    ///
    /// # Arguments
    ///
    /// * `address`: The address of the value to be retrieved
    ///
    /// Returns: `Option<&mut T>`, A mutable reference to the value if it exists.
    fn get_mut(&mut self, address: A) -> Option<&mut T> {
        if self.contains_address(address) {
            Some(&mut self[address])
        } else {
            None
        }
    }
    /// Evaluates whether an address is valid and has an associated value in the tensor.
    ///
    /// # Arguments
    ///
    /// * `address`: The address to be evaluated
    ///
    /// Returns: `bool`, A boolean which is true if and only if the address is valid and has an
    /// associated value.
    fn contains_address(&self, address: A) -> bool {
        (0..DIMENSION).all(|d| {
            address.get_value_at_dimension_index(d)
                >= self
                    .smallest_contained_address()
                    .get_value_at_dimension_index(d)
                && address.get_value_at_dimension_index(d)
                    <= self
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

    fn address_value_iter(&'a self) -> AddressValueIterator<'a, T, V, A, Self, DIMENSION>
    where
        Self: Sized,
    {
        AddressValueIterator::<'a, T, V, A, Self, DIMENSION>::new(&self)
    }
}
