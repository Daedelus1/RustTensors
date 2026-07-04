use std::{
    fmt::Debug,
    ops::{Add, Sub},
};

use crate::generic_tensor_address::GenericTensorAddress;

pub trait Addressable<V: AddressValue, const RANK: usize>:
    Copy + Clone + Debug + From<GenericTensorAddress<RANK, V>> + Into<GenericTensorAddress<RANK, V>>
{
    fn get_value_at_rank(&self, index: usize) -> V;
}

pub trait AddressValue:
    Copy + From<usize> + Into<usize> + Add<Output = Self> + Sub<Output = Self> + PartialOrd + Debug
{
}

impl<T> AddressValue for T where
    T: Copy
        + From<usize>
        + Into<usize>
        + Add<Output = Self>
        + Sub<Output = Self>
        + PartialOrd
        + Debug
{
}
