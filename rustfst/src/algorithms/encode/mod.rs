pub use decode_static::decode;
pub use encode_static::{encode, encode_into};
pub use encode_type::EncodeType;
pub use table::EncodeTable;
use table::EncodeTableMut;

mod decode_static;
mod encode_static;
mod encode_type;
mod table;
