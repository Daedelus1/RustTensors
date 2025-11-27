use crate::address_iterator::AddressIterator;
use crate::adressable::Addressable;
use std::ops::{Add, Index, IndexMut, Sub};

pub trait Tensor<
    T,
    V: Copy + From<u8> + Add<Output = V> + Sub<Output = V> + PartialOrd,
    A: Addressable<V, DIMENSION>,
    I: Iterator,
    const DIMENSION: usize,
>: Index<A, Output = T> + IndexMut<A, Output = T>
{
    fn contains_address(&self, address: A) -> bool;
    fn smallest_contained_address(&self) -> A;
    fn largest_contained_address(&self) -> A;
    fn get(&self, address: A) -> Result<&T, String> {
        if self.contains_address(address) {
            Ok(&self[address])
        } else {
            Err(format!(
                "Cannot retrieve value at {address:?}, index out of bounds."
            ))
        }
    }
    fn get_mut(&mut self, address: A) -> Result<&mut T, String> {
        if self.contains_address(address) {
            Ok(&mut self[address])
        } else {
            Err(format!(
                "Cannot retrieve value at {address:?}, index out of bounds."
            ))
        }
    }
    fn address_iter(&self) -> AddressIterator<V, A, DIMENSION> {
        let upper_bounds_exclusive = self.largest_contained_address().into();
        for mut i in upper_bounds_exclusive {
            i = i + 1.into();
        }
        AddressIterator::<V, A, DIMENSION>::new(
            self.smallest_contained_address().into().clone(),
            upper_bounds_exclusive.clone(),
        )
    }
}
