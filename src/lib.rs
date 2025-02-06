pub mod address_bound;
pub mod addressable;
pub mod matrix;
pub mod matrix_address;
pub mod tensor;
pub mod prelude {
    pub use crate::address_bound::AddressBound;
    pub use crate::address_bound::AddressIterator;
    pub use crate::addressable::Addressable;
    pub use crate::matrix::Matrix;
    pub use crate::matrix_address::MatrixAddress;
    pub use crate::tensor::Tensor; 
}