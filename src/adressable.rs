use std::fmt::Debug;

pub trait Addressable<V, const DIMENSION: usize>: Copy + Clone + Debug {
    fn new(data: [V; DIMENSION]) -> Self;
    fn get_value_at_dimension_index(&self, index: usize) -> V;
}
