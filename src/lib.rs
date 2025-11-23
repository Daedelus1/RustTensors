pub mod adressable;
pub mod matrix;
pub mod matrix_address;
pub mod tensor;

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn general_usage_test() {
        let mut matrix =
            Matrix::<i32>::parse_matrix("1 2 3|4 5 6|7 8 9", " ", "|", |s| s.parse().unwrap())
                .unwrap();
        assert_eq!(
            matrix.to_display_string(|i| i.to_string(), "-", "|"),
            "1-2-3|4-5-6|7-8-9"
        );
        println!(
            "Matrix: [{}]",
            matrix.to_display_string(i32::to_string, ", ", " | ")
        );
        matrix[(1, 1)] = 2;

        println!(
            "Matrix: [{}]",
            matrix.to_display_string(i32::to_string, ", ", " | ")
        );
    }
}
