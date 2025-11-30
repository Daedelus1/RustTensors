use std::{
    fmt::Debug,
    ops::{Add, Sub},
};

pub trait Addressable<V, const DIMENSION: usize>:
    Copy + Clone + Debug + From<[V; DIMENSION]> + Into<[V; DIMENSION]>
{
    fn get_value_at_dimension_index(&self, index: usize) -> V;
}

pub trait AddressValue:
    Copy + From<u8> + Add<Output = Self> + Sub<Output = Self> + PartialOrd
{
}

impl<T: Copy + From<u8> + Add<Output = Self> + Sub<Output = Self> + PartialOrd> AddressValue for T {}
