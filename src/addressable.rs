pub trait Addressable: PartialEq + Ord + Sized + Copy {
    fn get_dimension_count() -> u32;
    fn new_from_value_vec(values: Vec<i64>) -> Self;
    fn get_item_at_dimension_index(&self, dimension_index: u32) -> &i64;
    fn add(&self, other: &Self) -> Self {
        Self::new_from_value_vec(
            (0..Self::get_dimension_count())
                .into_iter()
                .map(|dimension_index: u32| {
                    self.get_item_at_dimension_index(dimension_index)
                        + other.get_item_at_dimension_index(dimension_index)
                })
                .collect(),
        )
    }
    fn subtract(&self, other: &Self) -> Self {
        Self::new_from_value_vec(
            (0..Self::get_dimension_count())
                .into_iter()
                .map(|dimension_index: u32| {
                    self.get_item_at_dimension_index(dimension_index)
                        - other.get_item_at_dimension_index(dimension_index)
                })
                .collect(),
        )
    }
    fn difference(&self, other: &Self) -> Self {
        Self::new_from_value_vec(
            (0..Self::get_dimension_count())
                .into_iter()
                .map(|dimension_index: u32| {
                    (self.get_item_at_dimension_index(dimension_index)
                        - other.get_item_at_dimension_index(dimension_index))
                    .abs()
                })
                .collect(),
        )
    }
    fn distance(&self, other: &Self) -> f64 {
        ((0..Self::get_dimension_count())
            .into_iter()
            .map(|dimension_index: u32| {
                (self.get_item_at_dimension_index(dimension_index)
                    - other.get_item_at_dimension_index(dimension_index))
                .pow(2)
            })
            .reduce(|acc, e| acc + e)
            .unwrap() as f64)
            .sqrt()
    }
    fn scale(&self, scalar: i64) -> Self {
        Self::new_from_value_vec(
            (0..Self::get_dimension_count())
                .into_iter()
                .map(|dimension_index: u32| {
                    self.get_item_at_dimension_index(dimension_index) * scalar
                })
                .collect(),
        )
    }
}
