use std::hash::Hash;

pub trait Addressable: PartialEq + Ord + Sized + Copy + Hash {
    fn get_dimension_count() -> u32;
    fn new_from_value_vec(values: Vec<i64>) -> Self;
    fn get_item_at_dimension_index(&self, dimension_index: u32) -> &i64;
    fn get_mut_item_at_dimension_index(&mut self, dimension_index: u32) -> &mut i64;
    fn add_in_place(&mut self, other: &Self) {
        for dimension_index in 0..Self::get_dimension_count() {
            *self.get_mut_item_at_dimension_index(dimension_index) +=
                other.get_item_at_dimension_index(dimension_index);
        }
    }
    fn subtract_in_place(&mut self, other: &Self) {
        for dimension_index in 0..Self::get_dimension_count() {
            *self.get_mut_item_at_dimension_index(dimension_index) -=
                other.get_item_at_dimension_index(dimension_index);
        }
    }
    fn scale_in_place(&mut self, scalar: f64) {
        for dimension_index in 0..Self::get_dimension_count() {
            let value = self.get_mut_item_at_dimension_index(dimension_index);
            *value = (*value as f64 * scalar) as i64;
        }
    }
    fn abs_in_place(&mut self) {
        for dimension_index in 0..Self::get_dimension_count() {
            let value = self.get_mut_item_at_dimension_index(dimension_index);
            *value = value.abs()
        }
    }
    fn difference_in_place(&mut self, other: &Self) {
        self.subtract_in_place(other);
        self.abs_in_place();
    }
    fn add(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.add_in_place(other);
        out
    }
    fn subtract(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.subtract_in_place(other);
        out
    }
    fn difference(&self, other: &Self) -> Self {
        let mut out = self.clone();
        out.difference_in_place(other);
        out
    }
    fn scale(&self, scalar: f64) -> Self {
        let mut out = self.clone();
        out.scale_in_place(scalar);
        out
    }
    fn distance(&self, other: &Self) -> f64 {
        let mut sum: i64 = 0;
        for dimension_index in 0..Self::get_dimension_count() {
            sum += (self.get_item_at_dimension_index(dimension_index)
                - other.get_item_at_dimension_index(dimension_index))
            .pow(2);
        }
        (sum as f64).sqrt()
    }
}
