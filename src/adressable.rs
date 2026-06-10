use std::{
    fmt::Debug,
    ops::{Add, Mul, Sub},
};

use crate::generic_tensor_address::GenericTensorAddress;

pub trait Addressable<V: AddressValue, const RANK: usize>:
    Copy + Clone + Debug + From<GenericTensorAddress<RANK, V>> + Into<GenericTensorAddress<RANK, V>>
{
    fn get_value_at_rank(&self, index: usize) -> V;
}

pub trait AddressValue:
    Copy + From<u8> + Add<Output = Self> + Sub<Output = Self> + PartialOrd + Debug + TryInto<usize>
{
}

impl<
    T: Copy + From<u8> + Add<Output = Self> + Sub<Output = Self> + PartialOrd + Debug + TryInto<usize>,
> AddressValue for T
{
}
