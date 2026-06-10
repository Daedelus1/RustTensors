use std::ops::{Index, IndexMut};

use crate::{
    adressable::{AddressValue, Addressable},
    error::Error,
    matrix_address::MatrixAddress,
};

#[derive(Debug, Clone, Copy)]
pub struct GenericTensorAddress<const RANK: usize, V: AddressValue = usize> {
    data: [V; RANK],
}

impl<const RANK: usize, V: AddressValue> From<[V; RANK]> for GenericTensorAddress<RANK, V> {
    fn from(value: [V; RANK]) -> Self {
        GenericTensorAddress::new(value)
    }
}

impl<const RANK: usize, V: AddressValue> GenericTensorAddress<RANK, V> {
    pub fn new(values: [V; RANK]) -> GenericTensorAddress<RANK, V> {
        GenericTensorAddress { data: values }
    }
}

impl<const RANK: usize, V: AddressValue> Addressable<V, RANK> for GenericTensorAddress<RANK, V> {
    fn get_value_at_rank(&self, index: usize) -> V {
        return self.data[index];
    }
}

impl<const RANK: usize, V: AddressValue> Index<usize> for GenericTensorAddress<RANK, V> {
    type Output = V;

    fn index(&self, rank_index: usize) -> &Self::Output {
        return &self.data[rank_index];
    }
}
impl<const RANK: usize, V: AddressValue> IndexMut<usize> for GenericTensorAddress<RANK, V> {
    fn index_mut(&mut self, rank_index: usize) -> &mut Self::Output {
        return &mut self.data[rank_index];
    }
}

impl TryFrom<MatrixAddress> for GenericTensorAddress<2> {
    type Error = Error;

    fn try_from(value: MatrixAddress) -> Result<Self, Self::Error> {
        if value.x < 0 || value.y < 0 {
            return Err(Error::AddressOutOfBounds(
                "Cannot represent address with negative indices as a GenericTensorArray!"
                    .to_owned(),
            ));
        }
        return Ok(GenericTensorAddress {
            data: [value.x as usize, value.y as usize],
        });
    }
}
