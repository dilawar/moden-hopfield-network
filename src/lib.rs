mod dam;
mod helper;
mod memory;

#[cfg(feature = "db")]
mod db;

pub mod data;
pub mod numeric;

pub use crate::dam::DenseAssociativeMemory;
pub use crate::helper::*;
pub use crate::memory::Memory;

#[cfg(feature = "db")]
pub fn db_default_name() -> &'static str {
    db::DEFAULT_DB_NAME
}
