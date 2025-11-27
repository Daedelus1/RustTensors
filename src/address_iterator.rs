use crate::adressable::Addressable;
use std::marker::PhantomData;
use std::ops::{Add, Sub};

pub struct AddressIterator<V: Copy + From<u8>, A: Addressable<V, DIMENSION>, const DIMENSION: usize>
{
    lower_bounds_inclusive: [V; DIMENSION],
    upper_bounds_exclusive: [V; DIMENSION],
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
        upper_bounds_exclusive: [V; DIMENSION],
    ) -> Self {
        let mut lower_bounds_copy: [V; DIMENSION] = lower_bounds_inclusive;
        lower_bounds_copy[0] = lower_bounds_copy[0] - 1.into();
        Self {
            lower_bounds_inclusive,
            upper_bounds_exclusive,
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
            if self.current_position[dimension_index] + 1.into()
                < self.upper_bounds_exclusive[dimension_index]
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
    #[test]
    fn address_iterator_test() {
        // let iter = AddressIterator::<MatrixAddress, i32, 2>::new([0, 0], [3, 4]);
        // for address in iter {
        //     println!("{address:?}");
        // }
    }
}
