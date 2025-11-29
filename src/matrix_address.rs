use crate::adressable::Addressable;
use std::ops::{Add, Neg, Sub};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MatrixAddress {
    pub x: i32,
    pub y: i32,
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
            x: x as i32,
            y: y as i32,
        }
    }
}

impl Addressable<i32, 2usize> for MatrixAddress {
    fn get_value_at_dimension_index(&self, index: usize) -> i32 {
        match index {
            0 => self.x,
            1 => self.y,
            _ => panic!("Invalid Dimension Index"),
        }
    }
}

impl From<[i32; 2]> for MatrixAddress {
    fn from(value: [i32; 2]) -> Self {
        Self {
            x: value[0],
            y: value[1],
        }
    }
}

impl Into<[i32; 2]> for MatrixAddress {
    fn into(self) -> [i32; 2] {
        [self.x, self.y]
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

impl Neg for MatrixAddress {
    type Output = Self;

    fn neg(self) -> Self::Output {
        MatrixAddress {
            x: -self.x,
            y: -self.y,
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
        fn arithmetic_test(x1 in -100000i32..100000i32, x2 in -100000i32..100000i32, y1 in -100000i32..100000i32, y2 in -100000i32..100000i32, s in -10000i32..10000i32) {
            let a1 = MatrixAddress{x: x1, y: y1};
            let a2 = MatrixAddress{x: x2, y: y2};

            assert_eq!(a1.get_value_at_dimension_index(0), x1);
            assert_eq!(a1.get_value_at_dimension_index(1), y1);
            assert_eq!(a2.get_value_at_dimension_index(0), x2);
            assert_eq!(a2.get_value_at_dimension_index(1), y2);

            assert_eq!(a1 - a2, a1 + (-a2));
            assert_eq!(a2 - a1, a2 + (-a1));
            assert_eq!(-(-a1), a1);
            assert_eq!(a1 + a2 - a2, a1);
            assert_eq!(-a1 + a2 + a1, a2);
            assert_eq!(a1.scale(2.0), MatrixAddress{x: a1.x * 2, y: a1.y * 2});

            let a1 = MatrixAddress{x: x1, y: y1};
            assert_eq!(a1.scale(s as f64), MatrixAddress{x: a1.x * s, y: a1.y * s});
        }
    }
}
