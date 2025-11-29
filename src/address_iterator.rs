use crate::adressable::Addressable;
use std::marker::PhantomData;
use std::ops::{Add, Sub};

pub struct AddressIterator<V: Copy + From<u8>, A: Addressable<V, DIMENSION>, const DIMENSION: usize>
{
    lower_bounds_inclusive: [V; DIMENSION],
    upper_bounds_inclusive: [V; DIMENSION],
    current_position: [V; DIMENSION],
    _marker: PhantomData<A>,
}

impl<
    V: Copy + From<u8> + Add<Output = V> + Sub<Output = V> + PartialOrd,
    A: Addressable<V, DIMENSION>,
    const DIMENSION: usize,
> AddressIterator<V, A, DIMENSION>
{
    pub(crate) fn new(
        lower_bounds_inclusive: [V; DIMENSION],
        upper_bounds_inclusive: [V; DIMENSION],
    ) -> Self {
        let mut lower_bounds_copy: [V; DIMENSION] = lower_bounds_inclusive;
        lower_bounds_copy[0] = lower_bounds_copy[0] - 1.into();
        Self {
            lower_bounds_inclusive,
            upper_bounds_inclusive,
            current_position: lower_bounds_copy,
            _marker: PhantomData,
        }
    }
}

impl<
    V: Copy + From<u8> + Add<Output = V> + Sub<Output = V> + PartialOrd,
    A: Addressable<V, DIMENSION>,
    const DIMENSION: usize,
> Iterator for AddressIterator<V, A, DIMENSION>
{
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        for dimension_index in 0..DIMENSION {
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

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;
    use crate::matrix_address::MatrixAddress;
    use crate::tensor::Tensor;

    // Working address iterator from the previous version
    pub struct MatrixAddressIterator {
        pub(crate) x: i32,
        pub(crate) y: i32,
        pub(crate) width: usize,
        pub(crate) height: usize,
    }
    impl Iterator for MatrixAddressIterator {
        type Item = MatrixAddress;

        fn next(&mut self) -> Option<Self::Item> {
            if self.x >= self.width as i32 - 1 {
                if self.y >= self.height as i32 - 1 {
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
        let matrix_address_iterator = MatrixAddressIterator {
            x: -1,
            y: 0,
            width,
            height,
        };
        let matrix = Matrix::new(width, height, |_| 0);
        for (true_address, new_address) in matrix_address_iterator.zip(matrix.address_iter()) {
            assert_eq!(true_address, new_address);
        }
    }
}
