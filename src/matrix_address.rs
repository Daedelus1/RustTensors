use crate::addressable::Addressable;
use std::cmp::Ordering;
use std::fmt::{Display, Formatter};

#[derive(Eq, PartialEq, Ord, Copy, Clone, Debug, Hash)]
pub struct MatrixAddress {
    pub x: i64,
    pub y: i64,
}

impl PartialOrd<Self> for MatrixAddress {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.y.cmp(&other.y) {
            Ordering::Less => Some(Ordering::Less),
            Ordering::Equal => match self.x.cmp(&other.x) {
                Ordering::Less => Some(Ordering::Less),
                Ordering::Equal => Some(Ordering::Equal),
                Ordering::Greater => Some(Ordering::Greater),
            },
            Ordering::Greater => Some(Ordering::Greater),
        }
    }
}

impl Addressable for MatrixAddress {
    fn get_dimension_count() -> u32 {
        2
    }

    fn new_from_value_vec(values: Vec<i64>) -> MatrixAddress {
        MatrixAddress {
            x: values[0],
            y: values[1],
        }
    }

    fn get_item_at_dimension_index(&self, dimension_index: u32) -> &i64 {
        match dimension_index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Unexpected Dimension Index Accessed"),
        }
    }

    fn get_mut_item_at_dimension_index(&mut self, dimension_index: u32) -> &mut i64 {
        match dimension_index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Unexpected Dimension Index Accessed"),
        }
    }
}

impl Display for MatrixAddress {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

#[allow(unused)]
impl MatrixAddress {
    fn new(x: i64, y: i64) -> MatrixAddress {
        MatrixAddress { x, y }
    }
}

#[cfg(test)]
mod tests {
    use crate::addressable::Addressable;
    use crate::matrix_address::MatrixAddress;
    use proptest::{prop_assert_eq, proptest};

    proptest! {
        #[test]
        fn doesnt_crash(x1 in -1_000_000_000i64..1_000_000_000i64, y1 in -1_000_000_000i64..1_000_000_000i64, x2 in -1_000_000_000i64..1_000_000_000i64, y2 in -1_000_000_000i64..1_000_000_000i64) {
            let a1 = MatrixAddress{x: x1, y: y1};
            let a2 = MatrixAddress{x: x2,y: y2};
            a1.add(&a2);
            a2.add(&a1);
            a1.subtract(&a2);
            a2.subtract(&a1);
            a1.difference(&a2);
            a2.difference(&a1);
            a1.distance(&a2);
            a2.distance(&a1);
            a1.scale(x2 as f64);
        }

        #[test]
        fn operation_consistency(x1 in -1_000_000_000i64..1_000_000_000i64, y1 in -1_000_000_000i64..1_000_000_000i64, x2 in -1_000_000_000i64..1_000_000_000i64, y2 in -1_000_000_000i64..1_000_000_000i64) {
            let a1 = MatrixAddress{x: x1, y: y1};
            let a2 = MatrixAddress{x: x2,y: y2};
            prop_assert_eq!(a1.add(&a2).subtract(&a2), a1);
            prop_assert_eq!(a1.subtract(&a2).add(&a2), a1);
            let mut a1_copy = a1.clone();
            a1_copy.add_in_place(&a2);
            a1_copy.subtract_in_place(&a2);
            prop_assert_eq!(a1_copy, a1);
            let mut a2_copy = a2.clone();
            a2_copy.add_in_place(&a1);
            a2_copy.subtract_in_place(&a1);
            prop_assert_eq!(a2_copy, a2)
        }

        #[test]
        fn operation_accuracy(x1 in -1_000_000i64..1_000_000i64, y1 in -1_000_000i64..1_000_000i64, x2 in -1_000_000i64..1_000_000i64, y2 in -1_000_000i64..1_000_000i64) {
            let a1 = MatrixAddress{x: x1, y: y1};
            let a2 = MatrixAddress{x: x2,y: y2};
            prop_assert_eq!(a1.add(&a2), MatrixAddress{x: x1 + x2,y: y1 + y2});
            prop_assert_eq!(a1.subtract(&a2), MatrixAddress{x: x1 - x2,y: y1 - y2});
            prop_assert_eq!(a1.scale(x2 as f64), MatrixAddress{x: x1 * x2,y: y1 * x2});
            prop_assert_eq!(a1.distance(&a2), (((x1 - x2).pow(2) as f64) + ((y1 - y2).pow(2) as f64)).sqrt());
            let mut a1 = MatrixAddress{x: x1, y: y1};
            let a2 = MatrixAddress{x: x2,y: y2};
            a1.add_in_place(&a2);
            prop_assert_eq!(a1, MatrixAddress{x: x1 + x2,y: y1 + y2});
            let mut a1 = MatrixAddress{x: x1, y: y1};
            let a2 = MatrixAddress{x: x2,y: y2};
            a1.subtract_in_place(&a2);
            prop_assert_eq!(a1, MatrixAddress{x: x1 - x2,y: y1 - y2});
            let mut a1 = MatrixAddress{x: x1, y: y1};
            a1.scale_in_place(x2 as f64);
            prop_assert_eq!(a1, MatrixAddress{x: x1 * x2,y: y1 * x2});
            
        }
    }
}
