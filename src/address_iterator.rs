use crate::adressable::{AddressValue, Addressable};
use crate::generic_tensor_address::GenericTensorAddress;
use crate::tensor::Tensor;
use std::marker::PhantomData;

pub struct AddressIterator<V: AddressValue, A: Addressable<V, RANK>, const RANK: usize> {
    lower_bounds_inclusive: GenericTensorAddress<RANK, V>,
    upper_bounds_inclusive: GenericTensorAddress<RANK, V>,
    current_position: GenericTensorAddress<RANK, V>,
    is_first_pass: bool,
    _marker: PhantomData<A>,
}

pub struct AddressValueIterator<
    'a,
    T: 'a,
    V: AddressValue,
    A: Addressable<V, RANK>,
    TENSOR: Tensor<'a, T, V, A, RANK>,
    const RANK: usize,
> {
    address_iterator: AddressIterator<V, A, RANK>,
    tensor: &'a TENSOR,
    _marker: PhantomData<T>,
}

impl<V: AddressValue, A: Addressable<V, RANK>, const RANK: usize> AddressIterator<V, A, RANK> {
    pub(crate) fn new(
        lower_bounds_inclusive: GenericTensorAddress<RANK, V>,
        upper_bounds_inclusive: GenericTensorAddress<RANK, V>,
    ) -> Self {
        let lower_bounds_copy: GenericTensorAddress<RANK, V> = lower_bounds_inclusive;
        Self {
            lower_bounds_inclusive,
            upper_bounds_inclusive,
            current_position: lower_bounds_copy,
            is_first_pass: true,
            _marker: PhantomData,
        }
    }
}

impl<'a, T, V, A, TENSOR, const RANK: usize> AddressValueIterator<'a, T, V, A, TENSOR, RANK>
where
    T: 'a,
    V: AddressValue,
    A: Addressable<V, RANK>,
    TENSOR: Tensor<'a, T, V, A, RANK>,
{
    pub(crate) fn new(tensor: &'a TENSOR) -> Self {
        Self {
            address_iterator: AddressIterator::new(
                tensor.smallest_contained_address().into(),
                tensor.largest_contained_address().into(),
            ),
            tensor,
            _marker: PhantomData,
        }
    }
}

impl<V: AddressValue, A: Addressable<V, RANK>, const RANK: usize> Iterator
    for AddressIterator<V, A, RANK>
{
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_first_pass {
            self.is_first_pass = false;
            return Some(self.current_position.into());
        }
        for dimension_index in 0..RANK {
            if self.current_position[dimension_index] < self.upper_bounds_inclusive[dimension_index]
            {
                self.current_position[dimension_index] =
                    self.current_position[dimension_index] + 1.into();
                return Some(self.current_position.into());
            } else {
                self.current_position[dimension_index] =
                    self.lower_bounds_inclusive[dimension_index];
            }
        }
        None
    }
}

impl<'a, T, V, A, TENSOR, const RANK: usize> Iterator
    for AddressValueIterator<'a, T, V, A, TENSOR, RANK>
where
    T: 'a,
    V: AddressValue,
    A: Addressable<V, RANK>,
    TENSOR: Tensor<'a, T, V, A, RANK>,
{
    type Item = (A, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(address) = self.address_iterator.next() {
            Some((address, &self.tensor[address]))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::address_iterator::AddressIterator;
    use crate::generic_tensor_address::GenericTensorAddress;
    use crate::matrix::Matrix;
    use crate::matrix_address::MatrixAddress;
    use crate::tensor::Tensor;

    // Working address iterator from the previous version
    pub struct MatrixAddressIterator {
        pub(crate) x: usize,
        pub(crate) y: usize,
        pub(crate) width: usize,
        pub(crate) height: usize,
    }
    impl Iterator for MatrixAddressIterator {
        type Item = MatrixAddress;

        fn next(&mut self) -> Option<Self::Item> {
            if self.x >= self.width - 1 {
                if self.y >= self.height - 1 {
                    return None;
                }
                self.x = 0;
                self.y += 1;
            } else {
                self.x += 1;
            }
            Some(MatrixAddress {
                x: self.x,
                y: self.y,
            })
        }
    }

    #[test]
    fn address_iterator_test() {
        let (width, height) = (1000, 2000);
        let mut matrix_address_iterator = MatrixAddressIterator {
            x: 0,
            y: 0,
            width,
            height,
        };
        let mut address_iterator = AddressIterator::<usize, MatrixAddress, 2>::new(
            GenericTensorAddress::new([0, 0]),
            GenericTensorAddress::new([width - 1, height - 1]),
        );
        assert_eq!(address_iterator.next(), Some(MatrixAddress { x: 0, y: 0 }));
        loop {
            let a = matrix_address_iterator.next();
            let b = address_iterator.next();
            assert_eq!(a, b);
            if a.is_none() {
                break;
            }
        }
    }

    #[test]
    fn address_value_iterator_test() {
        let (width, height) = (1000, 2000);
        let matrix = Matrix::new(width, height, |address| address.y * width + address.x).unwrap();
        let address_iter = matrix.address_iter();
        let address_value_iter = matrix.address_value_iter();
        address_iter
            .zip(address_value_iter)
            .for_each(|(a1, (a2, value))| {
                assert_eq!(a1, a2);
                assert_eq!(*value, a2.y * width + a2.x);
            })
    }

    #[test]
    fn transform_test() {
        let (width, height) = (1000, 1000);
        let matrix = Matrix::new(width, height, |_| 0u8).unwrap();
        let matrix = matrix.transform(|address, _value| address.y * width + address.x);
        matrix
            .address_value_iter()
            .for_each(|(address, value)| assert_eq!(address.y * width + address.x, *value));
    }
}
