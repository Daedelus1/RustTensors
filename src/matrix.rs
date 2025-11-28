use crate::matrix_address::MatrixAddress;
use crate::tensor::Tensor;
use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut};

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Matrix<T> {
    width: usize,
    height: usize,
    data: Vec<T>,
}

impl<T> Matrix<T> {
    /// Creates a new Matrix based on dimensions and a mapper function.
    ///
    /// # Arguments
    ///
    /// * `width`: The width, or number of columns in the matrix
    /// * `height`: The height, or number of rows in the matrix
    /// * `address_value_converter`: Converts a matrix address to a value.
    ///
    /// Returns: Matrix<T>
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_tensors::matrix::Matrix;
    /// use rust_tensors::tensor::Tensor;
    ///
    /// // Creates a 1000x1000 zero matrix
    /// let (width, height) = (1000, 1000);
    /// let mut matrix = Matrix::new(width, height, |_address| 0usize);
    /// matrix.address_iter()
    ///     .for_each(|address| assert_eq!(matrix[address], 0));
    ///
    /// // Creates a 50x10 matrix where the value is the index of the array
    /// let (width, height) = (1000, 1000);
    /// let mut matrix = Matrix::new(width, height, |address| address.y * width as i32 + address.x);
    /// matrix.address_iter()
    ///     .for_each(|address| assert_eq!(matrix[address], address.y * width as i32 + address.x));
    /// ```
    pub fn new<F>(width: usize, height: usize, address_value_converter: F) -> Self
    where
        F: Fn(MatrixAddress) -> T,
    {
        let mut matrix = Matrix {
            width,
            height,
            data: Vec::<T>::with_capacity(width * height),
        };
        matrix
            .address_iter()
            .for_each(|address| matrix.data.push(address_value_converter(address)));
        matrix
    }

    /// Makes a string fit for displaying the contents of the matrix
    ///
    /// # Arguments
    ///
    /// * `display_func`: Converts a value to a string
    /// * `row_delimiter`: Separates the rows in the matrix
    /// * `column_delimiter`: Separates the columns in the matrix
    ///
    /// Returns: the formatted string
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_tensors::matrix::Matrix;
    /// let mut matrix =
    /// Matrix::<i32>::parse_matrix("1 2 3|4 5 6|7 8 9", " ", "|", |s| s.parse().unwrap())
    ///     .unwrap();
    /// assert_eq!(
    ///     matrix.to_display_string(|i| i.to_string(), "-", "|"),
    ///     "1-2-3|4-5-6|7-8-9"
    /// );
    /// ```
    pub fn to_display_string<T1: Display, F: Fn(&T) -> T1>(
        &self,
        display_func: F,
        row_delimiter: &str,
        column_delimiter: &str,
    ) -> String {
        self.address_iter()
            .enumerate()
            .map(|(i, address)| {
                format!(
                    "{}{}",
                    display_func(&self[address]),
                    if (i + 1) % (self.width) == 0 {
                        if i != self.width * self.height - 1 {
                            column_delimiter
                        } else {
                            ""
                        }
                    } else {
                        row_delimiter
                    }
                )
            })
            .fold("".to_string(), |a: String, b: String| a + &b)
    }

    /// Parses a matrix from a string.
    /// Fallible, will return an Err if the matrix cannot be parsed,
    /// or if the matrix does not have uniform row length
    ///
    /// # Arguments
    ///
    /// * `data_str`: The string to be parsed
    /// * `column_delimiter`: The string which separates the items in the columns
    /// * `row_delimiter`: The string which separates the rows
    /// * `str_to_t_converter`: The function which converts the item strings to a value
    ///
    /// returns: Result<Matrix<T>, String>
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_tensors::matrix::Matrix;
    ///
    /// let mut matrix =
    ///     Matrix::<i32>::parse_matrix("0 1 2|3 4 5|6 7 8", " ", "|", |s| s.parse().unwrap())
    ///         .unwrap();
    ///
    /// assert_eq!(
    ///     matrix, Matrix::new(3, 3, |address| address.x + 3 * address.y)
    /// );
    /// ```
    pub fn parse_matrix<F>(
        data_str: &str,
        column_delimiter: &str,
        row_delimiter: &str,
        str_to_t_converter: F,
    ) -> Result<Matrix<T>, String>
    where
        F: Fn(&str) -> T,
    {
        let values: Vec<Vec<&str>> = data_str
            .split(row_delimiter)
            .map(|row| {
                row.split(column_delimiter)
                    .filter(|string| !string.is_empty())
                    .collect()
            })
            .filter(|row: &Vec<&str>| !row.is_empty())
            .collect();
        if values
            .iter()
            .skip(1)
            .any(|row| row.len() != values.first().unwrap().len())
        {
            return Err("Row Lengths are not constant".into());
        }
        let height = values.len();
        let width = values.first().unwrap().len();

        Ok(Matrix::new(width, height, |address| {
            str_to_t_converter(values[address.y as usize][address.x as usize])
        }))
    }
    fn index_address(&self, address: MatrixAddress) -> usize {
        address.y as usize * self.width + address.x as usize
    }
}

impl<T: Display> Display for Matrix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.to_display_string(|t| t.to_string(), " ", "\n")
        )
    }
}

impl<T> Index<MatrixAddress> for Matrix<T> {
    type Output = T;

    fn index(&self, index: MatrixAddress) -> &Self::Output {
        &self.data[self.index_address(index)]
    }
}

impl<T> Index<(i32, i32)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (i32, i32)) -> &Self::Output {
        &self[MatrixAddress {
            x: index.0,
            y: index.1,
        }]
    }
}

impl<T> IndexMut<MatrixAddress> for Matrix<T> {
    fn index_mut(&mut self, index: MatrixAddress) -> &mut Self::Output {
        let index = self.index_address(index);
        &mut self.data[index]
    }
}

impl<T> IndexMut<(i32, i32)> for Matrix<T> {
    fn index_mut(&mut self, index: (i32, i32)) -> &mut Self::Output {
        &mut self[MatrixAddress {
            x: index.0,
            y: index.1,
        }]
    }
}

impl<T> Tensor<T, i32, MatrixAddress, 2> for Matrix<T> {
    fn contains_address(&self, address: MatrixAddress) -> bool {
        address.x >= 0
            && address.x < self.width as i32
            && address.y >= 0
            && address.y < self.height as i32
    }

    fn smallest_contained_address(&self) -> MatrixAddress {
        MatrixAddress { x: 0, y: 0 }
    }

    fn largest_contained_address(&self) -> MatrixAddress {
        //TODO: Come up with better name so redundant math doesnt have to happen
        MatrixAddress {
            x: (self.width - 1) as i32,
            y: (self.height - 1) as i32,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::address_iterator;
    use crate::address_iterator::AddressIterator;
    use crate::matrix::Matrix;
    use crate::matrix_address::MatrixAddress;
    use crate::tensor::Tensor;
    use proptest::proptest;
    use std::str::FromStr;

    #[test]
    fn display_test() {
        let (width, height) = (11, 11);
        assert_eq!(
            "0 1 2 3 4 5 6 0 1 2 3\n4 5 6 0 1 2 3 4 5 6 0\n1 2 3 4 5 6 0 1 2 3 4\n5 6 0 1 2 3 4 5 6 0 1\n2 3 4 5 6 0 1 2 3 4 5\n6 0 1 2 3 4 5 6 0 1 2\n3 4 5 6 0 1 2 3 4 5 6\n0 1 2 3 4 5 6 0 1 2 3\n4 5 6 0 1 2 3 4 5 6 0\n1 2 3 4 5 6 0 1 2 3 4\n5 6 0 1 2 3 4 5 6 0 1",
            format!(
                "{}",
                Matrix::new(width, height, |address: MatrixAddress| {(address.x as usize + address.y as usize * width ) % 7})
            )
        )
    }
    #[test]
    fn set_test() {
        let (width, height) = (1000, 1000);
        let mut matrix = Matrix::new(width, height, |_address| 0usize);
        matrix.address_iter().for_each(|address| {
            assert_eq!(matrix[address], 0usize);
            matrix[address] = matrix.index_address(address);
            assert_eq!(matrix[address], matrix.index_address(address));
        });
        matrix
            .address_iter()
            .for_each(|address| assert_eq!(matrix.index_address(address), matrix[address]))
    }

    #[test]
    fn get_test() {
        let (width, height) = (1000, 1000);
        let matrix = Matrix::new(width, height, |address| {
            address.x as usize + address.y as usize * width
        });
        assert_eq!(matrix.index_address(MatrixAddress { x: 999, y: 0 }), 999);
        assert_eq!(matrix.index_address(MatrixAddress { x: 0, y: 1 }), 1000);
        assert_eq!(matrix.index_address(MatrixAddress { x: 1, y: 1 }), 1001);
        matrix
            .address_iter()
            .for_each(|address| assert_eq!(matrix.index_address(address), matrix[address]))
    }
    #[test]
    fn parse_test() {
        let data_str = "0,1,2,3,4,5,6,0,1,2,3|4,5,6,0,1,2,3,4,5,6,0|1,2,3,4,5,6,0,1,2,3,4|5,6,0,1,2,3,4,5,6,0,1|2,3,4,5,6,0,1,2,3,4,5|6,0,1,2,3,4,5,6,0,1,2|3,4,5,6,0,1,2,3,4,5,6|0,1,2,3,4,5,6,0,1,2,3|4,5,6,0,1,2,3,4,5,6,0|1,2,3,4,5,6,0,1,2,3,4|5,6,0,1,2,3,4,5,6,0,1";
        let (width, height) = (11, 11);
        assert_eq!(
            Matrix::new(width, height, |address: MatrixAddress| (address.y
                * width as i32
                + address.x)
                % 7),
            Matrix::parse_matrix(data_str, ",", "|", |string| i32::from_str(string)
                .expect(""))
            .expect("")
        );
    }

    #[test]
    fn equality_test() {
        let (width, height) = (100, 200);
        let mut m1 = Matrix::new(width, height, |address| {
            address.y * width as i32 + address.x
        });
        let m2 = Matrix::new(width, height, |address| {
            address.y * width as i32 + address.x
        });
        assert_eq!(m1, m2);
        for address in m1.address_iter() {
            assert_eq!(m1, m2);
            m1[address] += 1;
            assert_ne!(m1, m2);
            m1[address] -= 1;
        }
    }
    #[test]
    fn address_iterator_test() {
        let iter: address_iterator::AddressIterator<_, MatrixAddress, 2> =
            AddressIterator::new([0, 0], [3, 5]);

        for a in iter {
            println!("{a:?}");
        }
        let iter: address_iterator::AddressIterator<_, MatrixAddress, 2> =
            AddressIterator::new([0, 0], [3, 5]);
        let values = iter
            .map(|address| (address.x, address.y))
            .collect::<Vec<(i32, i32)>>();
        assert_eq!(
            values,
            vec![
                (0, 0),
                (1, 0),
                (2, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (0, 3),
                (1, 3),
                (2, 3),
                (0, 4),
                (1, 4),
                (2, 4),
            ]
        )
    }
    #[test]
    fn address_sugar_test_indv() {
        let (x, y) = (0, 198);
        let matrix = Matrix::new(100, 200, |address| address.y * 100 + address.x);
        let mut mut_matrix = matrix.clone();
        let pos_tuple = (x, y);
        let pos_address = MatrixAddress { x, y };
        assert_eq!(matrix[pos_tuple], matrix[pos_address]);
        let temp = matrix[pos_tuple];
        mut_matrix[pos_address] = -1;
        mut_matrix[pos_tuple] = temp;
        assert_eq!(matrix, mut_matrix);
    }

    proptest! {
        #[test]
        fn address_sugar_test(x in 0..100, y in 0..200) {
            let matrix = Matrix::new(100, 200, |address| address.y * 100 + address.x);
            let mut mut_matrix = matrix.clone();
            let pos_tuple = (x, y);
            let pos_address = MatrixAddress{x, y};
            assert_eq!(matrix[pos_tuple], matrix[pos_address]);
            let temp = matrix[pos_tuple];
            mut_matrix[pos_address] = -1;
            mut_matrix[pos_tuple] = temp;
            assert_eq!(matrix, mut_matrix);
        }
    }
}
