use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{Index, IndexMut},
};

use crate::{
    adressable::AddressValue, error::Error, generic_tensor_address::GenericTensorAddress,
    tensor::Tensor,
};

pub struct GenericTensor<T, V: AddressValue, const RANK: usize> {
    data: Vec<T>,
    dimensions: [V; RANK],
}

impl<T, V: AddressValue, const RANK: usize> GenericTensor<T, V, RANK> {
    fn index_address(&self, address: GenericTensorAddress<RANK, V>) -> usize
    where
        V::Error: Debug,
    {
        let mut scalar = 1usize;
        let mut index = 0usize;
        for i in 0..RANK {
            index = address[i]
                .try_into()
                .expect("All Rank indices must be positive or zero to index")
                * (scalar);
            scalar *= self.dimensions[i]
                .try_into()
                .expect("All Rank indices must be positive or zero to index");
        }
        return index;
    }
}

impl<T, const RANK: usize> GenericTensor<T, usize, RANK> {
    fn index_address(&self, address: GenericTensorAddress<RANK, usize>) -> usize {}
}

impl<T, V: AddressValue, const RANK: usize> Index<GenericTensorAddress<RANK, V>>
    for GenericTensor<T, V, RANK>
{
    type Output = T;

    fn index(&self, index: GenericTensorAddress<RANK, V>) -> &Self::Output {
        todo!()
    }
}
impl<T, V: AddressValue, const RANK: usize> IndexMut<GenericTensorAddress<RANK, V>>
    for GenericTensor<T, V, RANK>
{
    fn index_mut(&mut self, index: GenericTensorAddress<RANK, V>) -> &mut Self::Output {
        todo!()
    }
}

impl<'a, T: 'a, V: AddressValue, const RANK: usize>
    Tensor<'a, T, V, GenericTensorAddress<RANK, V>, RANK> for GenericTensor<T, V, RANK>
{
    fn smallest_contained_address(&self) -> GenericTensorAddress<RANK, V> {
        return [0.into(); RANK].into();
    }

    fn largest_contained_address(&self) -> GenericTensorAddress<RANK, V> {
        self.dimensions.into()
    }
}
