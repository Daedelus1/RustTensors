use std::fmt::Debug;

pub trait Addressable<V, const DIMENSION: usize>:
    Copy + Clone + Debug + From<[V; DIMENSION]> + Into<[V; DIMENSION]>
{
    fn get_value_at_dimension_index(&self, index: usize) -> V;
}
