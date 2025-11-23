use crate::adressable::Addressable;
use std::ops::{Add, Neg, Sub};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MatrixAddress {
    pub x: i32,
    pub y: i32,
}

impl Addressable<i32, 2usize> for MatrixAddress {
    fn new(data: [i32; 2]) -> Self {
        MatrixAddress {
            x: data[0],
            y: data[1],
        }
    }

    fn get_value_at_dimension_index(&self, index: usize) -> i32 {
        match index {
            0 => self.x,
            1 => self.y,
            _ => panic!("Invalid Dimension Index"),
        }
    }
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
    ///
    /// ```
    pub fn scale(self, scalar: f64) -> Self {
        MatrixAddress {
            x: (self.x as f64 * scalar + f64::EPSILON) as i32,
            y: (self.y as f64 * scalar + f64::EPSILON) as i32,
        }
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

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;
    use crate::matrix_address::MatrixAddress;
    use proptest::proptest;

    proptest! {
        #[test]
        fn arithmetic_test(x1 in -1000i32..1000i32, x2 in -1000i32..1000i32, y1 in -1000i32..1000i32, y2 in -1000i32..1000i32) {
            let a1 = MatrixAddress{x: x1, y: y1};
            let a2 = MatrixAddress{x: x2, y: y2};

            assert_eq!(a1 - a2, a1 + (-a2));
            assert_eq!(a2 - a1, a2 + (-a1));
            assert_eq!(-(-a1), a1);
            assert_eq!(a1 + a2 - a2, a1);
            assert_eq!(-a1 + a2 + a1, a2);
            assert_eq!(a1.scale(2.0), MatrixAddress{x: a1.x * 2, y: a1.y * 2});

            let a1 = MatrixAddress{x: x1, y: y1};
            assert_eq!(a1.scale(2.0), MatrixAddress{x: a1.x * 2, y: a1.y * 2});
        }
    }
}
