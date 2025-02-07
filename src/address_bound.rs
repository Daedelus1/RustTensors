use crate::addressable::Addressable;

#[derive(Clone, PartialEq, Debug)]
pub struct AddressBound<A: Addressable> {
    pub smallest_possible_position: A,
    pub largest_possible_position: A,
}

impl<A: Addressable> AddressBound<A> {
    pub fn contains_address(&self, address: &A) -> bool {
        for d in 0..A::get_dimension_count() {
            if address.get_item_at_dimension_index(d)
                < self
                    .smallest_possible_position
                    .get_item_at_dimension_index(d)
                || address.get_item_at_dimension_index(d)
                    > self
                        .largest_possible_position
                        .get_item_at_dimension_index(d)
            {
                return false;
            }
        }
        true
    }

    pub fn iter(&self) -> AddressIterator<A> {
        AddressIterator {
            bounds: AddressBound::new(
                self.smallest_possible_position,
                self.largest_possible_position,
            ),
            abacus: Vec::new(),
        }
    }

    pub fn new(smallest_possible_position: A, largest_possible_position: A) -> AddressBound<A> {
        AddressBound {
            smallest_possible_position,
            largest_possible_position,
        }
    }

    pub fn index_address(&self, address: &A) -> Option<usize> {
        if !self.contains_address(&address) {
            return None;
        }
        let mut out: usize = 0;

        for d in (0..A::get_dimension_count()).rev() {
            out *= (self
                .largest_possible_position
                .get_item_at_dimension_index(d)
                - self
                    .smallest_possible_position
                    .get_item_at_dimension_index(d)
                + 1) as usize;
            out += (address.get_item_at_dimension_index(d)
                - self
                    .smallest_possible_position
                    .get_item_at_dimension_index(d)) as usize;
        }

        Some(out)
    }
    pub fn get_address_from_index(&self, index: usize) -> Result<A, &str> {
        let mut index = index;
        let mut values: Vec<i64> = Vec::new();
        for d in 0..A::get_dimension_count() {
            let min_value = *(self
                .smallest_possible_position
                .get_item_at_dimension_index(d));
            let max_value = *(self
                .largest_possible_position
                .get_item_at_dimension_index(d));
            let breadth = (max_value - min_value + 1) as usize;
            let value = (index % breadth) as i64 + min_value;
            values.push(value);
            index /= breadth;
        }
        if index != 0 {
            return Err("Index is too large.");
        }
        Ok(A::new_from_value_vec(values))
    }
}

pub struct AddressIterator<A: Addressable> {
    bounds: AddressBound<A>,
    abacus: Vec<i64>,
}

impl<A: Addressable> Iterator for AddressIterator<A> {
    type Item = A;

    fn next(&mut self) -> Option<Self::Item> {
        if self.abacus.len() == 0 {
            let dims = A::get_dimension_count();
            self.abacus = vec![0; dims as usize];
            for i in 0..dims {
                self.abacus[i as usize] = *self
                    .bounds
                    .smallest_possible_position
                    .get_item_at_dimension_index(i);
            }
            return Some(A::new_from_value_vec(self.abacus.clone()));
        }
        for dimension in 0..A::get_dimension_count() {
            if self.abacus[dimension as usize]
                >= *self
                    .bounds
                    .largest_possible_position
                    .get_item_at_dimension_index(dimension)
            {
                if dimension == A::get_dimension_count() - 1 {
                    return None;
                }
                self.abacus[dimension as usize] = *self
                    .bounds
                    .smallest_possible_position
                    .get_item_at_dimension_index(dimension);
                continue;
            } else {
                self.abacus[dimension as usize] += 1;
            }
            break;
        }
        Some(A::new_from_value_vec(self.abacus.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix_address::MatrixAddress;
    use proptest::proptest;

    #[test]
    fn iteration_visual_test() {
        let bounds = AddressBound {
            smallest_possible_position: MatrixAddress { x: 50, y: 50 },
            largest_possible_position: MatrixAddress { x: 69, y: 100 },
        };
        assert!(bounds.iter().is_sorted());
    }
    proptest! {
        #[test]
        fn indexing_test(x1 in 0i64..1000, y1 in 0i64..1000, x2 in 0i64..1000, y2 in 0i64..1000) {
            if x2 < x1 || y2 < y1 {
                return Ok(());
            }
            let bounds = AddressBound {
                smallest_possible_position: MatrixAddress { x: x1, y: y1 },
                largest_possible_position: MatrixAddress { x: x2, y: y2 },
            };
            bounds.iter()
            .enumerate()
            .for_each(|(index, address)| {
                assert_eq!(bounds.index_address(&address).unwrap(), ((address.y - y1) * (x2-x1 + 1) + (address.x - x1)) as usize);
                assert_eq!(bounds.get_address_from_index(index).expect("Index out of bounds"), address);
            });
        }
        #[test]
        fn address_iteration_test(x1 in 0i64..1000, y1 in 0i64..1000, x2 in 0i64..1000, y2 in 0i64..1000) {
            if x2 < x1 || y2 < y1 {
                return Ok(());
            }
            let bounds = AddressBound {
                smallest_possible_position: MatrixAddress { x: x1, y: y1 },
                largest_possible_position: MatrixAddress { x: x2, y: y2 },
            };
            bounds.iter().for_each(|address| assert!(bounds.contains_address(&address)));
            assert_eq!((x2 - x1 + 1) * (y2 - y1 + 1), bounds.iter().collect::<Vec<_>>().len().try_into().unwrap())
        }
        #[test]
        fn contains_test(x1 in 0i64..1000, y1 in 0i64..1000, x2 in 0i64..1000, y2 in 0i64..1000, x3 in 0i64..1000, y3 in 0i64..1000) {
           if x2 < x1 || y2 < y1 {
                return Ok(());
            }
            let bounds = AddressBound {
                smallest_possible_position: MatrixAddress { x: x1, y: y1 },
                largest_possible_position: MatrixAddress { x: x2, y: y2 },
            };
            assert_eq!(bounds.contains_address(&MatrixAddress{x: x3,y: y3}), x3 >= x1 && x3 <= x2 && y3 >= y1 && y3 <= y2);
        }
    }
}
