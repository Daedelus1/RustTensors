use crate::{adressable::Addressable, generic_tensor_address::GenericTensorAddress};
use std::ops::{Add, Sub};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MatrixAddress {
    pub x: usize,
    pub y: usize,
}

impl MatrixAddress {
    /// Scales the position of the matrix address by the floating point scalar.
    /// Epsilon is added to the results before truncation to avoid floating point precision issues
    /// # Arguments
    ///
    /// * `scalar`: The scalar to multiply the value with.
    ///
    /// Returns: MatrixAddress
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_tensors::matrix_address::MatrixAddress;
    /// let address = MatrixAddress {x: 5, y: 10};
    /// assert_eq!(address.scale(0.5), MatrixAddress {x: 2, y: 5});
    /// ```
    pub fn scale(self, scalar: f64) -> Self {
        let (mut x, mut y) = (self.x as f64 * scalar, self.y as f64 * scalar);
        if x > 0.0 {
            x += f64::EPSILON;
        }
        if y > 0.0 {
            y += f64::EPSILON;
        }
        MatrixAddress {
            x: x as usize,
            y: y as usize,
        }
    }
}

impl Addressable<usize, 2usize> for MatrixAddress {
    fn get_value_at_rank(&self, index: usize) -> usize {
        match index {
            0 => self.x,
            1 => self.y,
            _ => panic!("Invalid Rank"),
        }
    }
}

impl From<GenericTensorAddress<2, usize>> for MatrixAddress {
    fn from(value: GenericTensorAddress<2, usize>) -> Self {
        Self {
            x: value[0],
            y: value[1],
        }
    }
}

impl From<MatrixAddress> for GenericTensorAddress<2, usize> {
    fn from(val: MatrixAddress) -> Self {
        GenericTensorAddress::new([val.x, val.y])
    }
}

impl Add for MatrixAddress {
    type Output = MatrixAddress;

    fn add(self, rhs: Self) -> Self::Output {
        MatrixAddress {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for MatrixAddress {
    type Output = MatrixAddress;

    fn sub(self, rhs: Self) -> Self::Output {
        MatrixAddress {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::adressable::Addressable;
    use crate::matrix_address::MatrixAddress;
    use proptest::proptest;

    proptest! {
        #[test]
        fn arithmetic_test(x1 in 0usize..100000usize, x2 in 0usize..100000usize, y1 in 0usize..100000usize, y2 in 0usize..100000usize, s in 0usize..10000usize) {
            let a1 = MatrixAddress{x: x1, y: y1};
            let a2 = MatrixAddress{x: x2, y: y2};

            assert_eq!(a1.get_value_at_rank(0), x1);
            assert_eq!(a1.get_value_at_rank(1), y1);
            assert_eq!(a2.get_value_at_rank(0), x2);
            assert_eq!(a2.get_value_at_rank(1), y2);

            assert_eq!(a1 + a2 - a2, a1);
            assert_eq!(a1.scale(2.0), MatrixAddress{x: a1.x * 2, y: a1.y * 2});

            let a1 = MatrixAddress{x: x1, y: y1};
            assert_eq!(a1.scale(s as f64), MatrixAddress{x: a1.x * s, y: a1.y * s});
        }
    }
}
