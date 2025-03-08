use crate::address_bound::{AddressBound, AddressIterator};
use crate::addressable::Addressable;
use std::io::Error;

/// The framework used to make a tensor or an N dimensional array
pub trait Tensor<T, A: Addressable> {
    /// Creates a new instance of a tensor with the given inclusive bounds and an address value mapper closure
    fn new<F>(bounds: AddressBound<A>, address_value_converter: F) -> Self
    where
        F: Fn(A) -> T;
    /// Attempts to retrieve a reference to the value at the address.
    /// Returns None if the address is outside the given bound
    fn get(&self, address: &A) -> Option<&T>;
    fn get_mut(&mut self, address: &A) -> Option<&mut T>;
    /// Attempts to reassign the value at the given index. Returns Err if the given address is out of bounds.
    fn set(&mut self, address: &A, value: T) -> Result<(), Error>;
    /// Returns a reference to the bounds of the tensor
    fn bounds(&self) -> &AddressBound<A>;
    /// Returns an iterator who generates addresses in row-major order.
    /// Can't give an address who is out of bounds.
    fn address_iterator<'a>(&'a self) -> AddressIterator<A>
    where
        T: 'a,
    {
        self.bounds().iter()
    }
    /// Returns a new tensor, which is a copy of this tensor with the values mapped by the given function.
    fn transform<F, TNew, TENew>(&self, mapping_function: F) -> TENew
    where
        F: Fn(&T) -> TNew,
        TENew: Tensor<TNew, A>,
    {
        Tensor::new(self.bounds().clone(), |address| {
            mapping_function(self.get(&address).unwrap())
        })
    }
    /// Returns a new tensor, which is a copy of this tensor with the values mapped by the given function.
    fn transform_by_address<F, TNew, TENew>(&self, mapping_function: F) -> TENew
    where
        F: Fn(&A, &T) -> TNew,
        TENew: Tensor<TNew, A>,
    {
        Tensor::new(self.bounds().clone(), |address| {
            mapping_function(&address, self.get(&address).unwrap())
        })
    }
    /// Alters every value in the tensor by some mapping function.
    /// Note: the mapping function must have the same input value and output value type.
    fn transform_in_place<F>(&mut self, mapping_function: F)
    where
        F: Fn(&mut T),
    {
        self.bounds().iter().for_each(|address| {
            mapping_function(self.get_mut(&address).unwrap());
        });
    }

    /// Alters every value in the tensor by some mapping function.
    /// Note: the mapping function must have the same input value and output value type.
    fn transform_by_address_in_place<F>(&mut self, mapping_function: F)
    where
        F: Fn(&A, &mut T),
    {
        self.bounds().iter().for_each(|address| {
            mapping_function(&address, self.get_mut(&address).unwrap());
        });
    }

    /// A canonical means of testing equality.
    /// Implementors may choose to use other more efficient means, but this is always viable.
    #[allow(unused)]
    fn eq(&self, other: &Self) -> bool
    where
        T: PartialEq,
    {
        if self.bounds() != other.bounds() {
            return false;
        }
        self.address_iterator()
            .all(|address| self.get(&address).unwrap() == other.get(&address).unwrap())
    }
}
