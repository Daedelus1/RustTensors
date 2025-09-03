use crate::address_bound::AddressBound;
use crate::matrix_address::MatrixAddress;
use crate::tensor::Tensor;
use std::fmt::{Display, Formatter};
use std::io::ErrorKind;
use std::io::ErrorKind::InvalidInput;

/// A tensor of two dimensions accessed using MatrixAddress.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T> {
    data: Vec<T>,
    bounds: AddressBound<MatrixAddress>,
}

impl<T> Tensor<T, MatrixAddress> for Matrix<T> {
    fn new<F>(bounds: AddressBound<MatrixAddress>, address_value_converter: F) -> Matrix<T>
    where
        F: Fn(MatrixAddress) -> T,
    {
        let data: Vec<T> = bounds.iter().map(address_value_converter).collect();
        Matrix { data, bounds }
    }

    fn get(&self, address: &MatrixAddress) -> Option<&T> {
        if !self.bounds.contains_address(address) {
            return None;
        }
        self.data.get(self.bounds.index_address(address).unwrap())
    }

    fn get_mut(&mut self, address: &MatrixAddress) -> Option<&mut T> {
        if !self.bounds.contains_address(address) {
            return None;
        }
        self.data
            .get_mut(self.bounds.index_address(address).unwrap())
    }

    fn set(&mut self, address: &MatrixAddress, value: T) -> Result<(), std::io::Error> {
        if !self.bounds.contains_address(address) {
            return Err(std::io::Error::new(
                InvalidInput,
                format!("The following address is out of bounds: {address}"),
            ));
        }
        self.data[self.bounds.index_address(address).unwrap()] = value;
        Ok(())
    }

    fn bounds(&self) -> &AddressBound<MatrixAddress> {
        &self.bounds
    }
}

impl<T> Matrix<T> {
    pub fn to_display_string<T1: Display, F: Fn(&T) -> T1>(
        &self,
        display_func: F,
        row_delimiter: &str,
        column_delimiter: &str,
    ) -> String {
        self.address_iterator()
            .enumerate()
            .map(|(i, address)| {
                format!(
                    "{}{}",
                    display_func(self.get(&address).unwrap()),
                    if (i as i64 + 1) % (self.bounds.largest_possible_position.x + 1) == 0 {
                        column_delimiter
                    } else {
                        row_delimiter
                    }
                )
            })
            .fold("".to_string(), |a: String, b: String| a + &b)
    }

    pub fn parse_matrix<F>(
        data_str: &str,
        column_delimiter: &str,
        row_delimiter: &str,
        str_to_t_converter: F,
    ) -> Result<Matrix<T>, std::io::Error>
    where
        F: Fn(&str) -> T,
    {
        let values: Vec<Vec<&str>> = data_str
            .split(row_delimiter)
            .map(|row| {
                row.split(column_delimiter)
                    .filter(|string| *string != "")
                    .collect()
            })
            .filter(|row: &Vec<&str>| row.len() != 0)
            .collect();
        if values
            .iter()
            .skip(1)
            .any(|row| row.len() != values.get(0).unwrap().len())
        {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "Row Lengths are not constant",
            ));
        }
        let height = values.len();
        let width = values.get(0).unwrap().len();
        let matrix_bounds = AddressBound {
            smallest_possible_position: MatrixAddress { x: 0, y: 0 },
            largest_possible_position: MatrixAddress {
                x: (width - 1) as i64,
                y: (height - 1) as i64,
            },
        };

        Ok(Matrix::new(matrix_bounds, |address| {
            str_to_t_converter(
                values
                    .get(address.y as usize)
                    .unwrap()
                    .get(address.x as usize)
                    .unwrap(),
            )
        }))
    }
}

impl<T: Display> Display for Matrix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "{}",
            self.to_display_string(|x: &T| x.to_string(), " ", "\n")
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::address_bound::AddressBound;
    use crate::matrix::Matrix;
    use crate::matrix_address::MatrixAddress;
    use crate::tensor::Tensor;
    use proptest::num::usize;
    use std::str::FromStr;

    #[test]
    fn display_test() {
        let bound = AddressBound::new(MatrixAddress { x: 0, y: 0 }, MatrixAddress { x: 10, y: 10 });
        assert_eq!(
            "0 1 2 3 4 5 6 0 1 2 3\n4 5 6 0 1 2 3 4 5 6 0\n1 2 3 4 5 6 0 1 2 3 4\n5 6 0 1 2 3 4 5 6 0 1\n2 3 4 5 6 0 1 2 3 4 5\n6 0 1 2 3 4 5 6 0 1 2\n3 4 5 6 0 1 2 3 4 5 6\n0 1 2 3 4 5 6 0 1 2 3\n4 5 6 0 1 2 3 4 5 6 0\n1 2 3 4 5 6 0 1 2 3 4\n5 6 0 1 2 3 4 5 6 0 1\n",
            format!(
                "{}",
                Matrix::new(bound.clone(), |address: MatrixAddress| bound
                    .index_address(&address)
                    .unwrap()
                    % 7)
            )
        )
    }

    #[test]
    fn parse_test() {
        let data_str = "0,1,2,3,4,5,6,0,1,2,3|4,5,6,0,1,2,3,4,5,6,0|1,2,3,4,5,6,0,1,2,3,4|5,6,0,1,2,3,4,5,6,0,1|2,3,4,5,6,0,1,2,3,4,5|6,0,1,2,3,4,5,6,0,1,2|3,4,5,6,0,1,2,3,4,5,6|0,1,2,3,4,5,6,0,1,2,3|4,5,6,0,1,2,3,4,5,6,0|1,2,3,4,5,6,0,1,2,3,4|5,6,0,1,2,3,4,5,6,0,1";
        let bound = AddressBound::new(MatrixAddress { x: 0, y: 0 }, MatrixAddress { x: 10, y: 10 });
        assert_eq!(
            Matrix::new(bound.clone(), |address: MatrixAddress| bound
                .index_address(&address)
                .unwrap()
                % 7),
            Matrix::parse_matrix(data_str, ",", "|", |string| usize::from_str(string)
                .expect(""))
            .expect("")
        );
    }
    #[test]
    fn get_test() {
        let bound = AddressBound {
            smallest_possible_position: MatrixAddress { x: 0, y: 0 },
            largest_possible_position: MatrixAddress { x: 1000, y: 1000 },
        };
        let matrix = Matrix::new(bound.clone(), |address| bound.index_address(&address));
        matrix.address_iterator().for_each(|address| {
            assert_eq!(
                bound.index_address(&address),
                *matrix.get(&address).unwrap()
            )
        })
    }

    #[test]
    fn set_test() {
        let bound = AddressBound {
            smallest_possible_position: MatrixAddress { x: 0, y: 0 },
            largest_possible_position: MatrixAddress { x: 1000, y: 1000 },
        };
        let mut matrix = Matrix::new(bound.clone(), |_address| 0usize);
        matrix.address_iterator().for_each(|address| {
            assert_eq!(matrix.get(&address).unwrap(), &0usize);
            matrix
                .set(&address, bound.index_address(&address).unwrap())
                .expect("Index out of bounds error");
            assert_eq!(
                matrix.get(&address).unwrap(),
                &(bound.index_address(&address).unwrap())
            );
        });
        matrix.address_iterator().for_each(|address| {
            assert_eq!(
                bound.index_address(&address).unwrap(),
                *(matrix.get(&address).unwrap())
            )
        })
    }
    #[test]
    fn transform_test() {
        let bound = AddressBound {
            smallest_possible_position: MatrixAddress { x: 0, y: 0 },
            largest_possible_position: MatrixAddress { x: 1000, y: 1000 },
        };
        let matrix = Matrix::new(bound.clone(), |address| {
            bound.index_address(&address).unwrap()
        });
        let transformed_matrix: Matrix<f64> = matrix.transform(|value| *value as f64);
        let transformed_by_address_matrix: Matrix<f64> =
            matrix.transform_by_address(|_address, value| *value as f64);
        matrix.address_iterator().for_each(|address| {
            assert_eq!(
                *matrix.get(&address).unwrap() as f64,
                *transformed_matrix.get(&address).unwrap(),
            );
            assert_eq!(
                *matrix.get(&address).unwrap() as f64,
                *transformed_by_address_matrix.get(&address).unwrap()
            );
        })
    }

    #[test]
    fn transform_in_place_test() {
        let bound = AddressBound {
            smallest_possible_position: MatrixAddress { x: 0, y: 0 },
            largest_possible_position: MatrixAddress { x: 1000, y: 1000 },
        };
        let original_matrix = Matrix::new(bound.clone(), |address| {
            bound.index_address(&address).unwrap() as i64
        });
        let mut working_matrix = original_matrix.clone();
        let mut working_by_address_matrix = original_matrix.clone();

        working_matrix.transform_in_place(|value| *value *= -2);
        working_by_address_matrix.transform_by_address_in_place(|_address, value| *value *= -2);
        assert_eq!(
            original_matrix.transform::<_, i64, Matrix<i64>>(|value| *value * -2),
            working_matrix
        );
        assert_eq!(working_matrix, working_by_address_matrix)
    }

    #[test]
    fn equality_test() {
        let bound = AddressBound {
            smallest_possible_position: MatrixAddress { x: 0, y: 0 },
            largest_possible_position: MatrixAddress { x: 1000, y: 1000 },
        };
        let m1 = Matrix::new(bound.clone(), |address| {
            bound.index_address(&address).unwrap() as i32
        });
        let m2 = Matrix::new(bound.clone(), |address| {
            bound.index_address(&address).unwrap() as i32
        });
        assert_eq!(m1, m2);
    }
}
