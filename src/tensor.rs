use crate::adressable::Addressable;
use std::ops::{Index, IndexMut};

pub trait Tensor<T, V: Copy, A: Addressable<V, DIMENSION>, I: Iterator, const DIMENSION: usize>:
    Index<A, Output = T> + IndexMut<A, Output = T>
{
    fn contains_address(&self, address: A) -> bool;
    fn address_iter(&self) -> I;
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
}
