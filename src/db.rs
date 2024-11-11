//! database layer.

use std::path::Path;

use crate::memory::Memory;
use byteorder::{ByteOrder, LittleEndian};
use num_traits::Float;
use rusqlite::Connection;

pub const DEFAULT_DB_NAME: &str = "dam.sqlite";

#[derive(Debug)]
pub struct Database {
    conn: Connection,
    table_prefix: String,
}

impl Default for Database {
    fn default() -> Self {
        let dbpath = if let Ok(path) = std::env::var("MEENA_DB_PATH") {
            path.into()
        } else if let Some(proj_dir) = directories::ProjectDirs::from("tech", "subcom", "meena") {
            proj_dir.data_dir().join(DEFAULT_DB_NAME)
        } else {
            DEFAULT_DB_NAME.into()
        };
        println!("Using database {}", dbpath.display());
        Self::new(dbpath, None)
    }
}

impl Database {
    /// Create a new [Database].
    pub fn new<P>(dbpath: P, table_prefix: Option<String>) -> Self
    where
        P: AsRef<Path>,
    {
        let mut d = Database {
            conn: Connection::open(dbpath).unwrap(),
            table_prefix: table_prefix.unwrap_or("default".to_string()),
        };
        d.init().unwrap();
        d
    }

    pub fn new_with_named_table<P>(dbpath: P, table_prefix: impl Into<String>) -> Self
    where
        P: AsRef<Path>,
    {
        let mut d = Database {
            conn: Connection::open(dbpath).unwrap(),
            table_prefix: table_prefix.into(),
        };
        d.init().unwrap();
        d
    }

    fn memory_table(&self) -> String {
        format!("{}_memories", &self.table_prefix)
    }

    fn pattern_table(&self) -> String {
        format!("{}_patterns", &self.table_prefix)
    }

    /// Set table prefix
    pub fn set_table_prefix(&mut self, table_prefix: &str) {
        if self.table_prefix != table_prefix {
            self.table_prefix = table_prefix.into();
            self.init().unwrap();
        }
    }

    /// Clear data from all tables.
    pub fn clear_tables(&mut self) {
        self.clear_table(&self.memory_table());
        self.clear_table(&self.pattern_table());
    }

    /// Clear a given table.
    fn clear_table(&mut self, tablename: &str) {
        tracing::info!("Clearning {tablename}");
        self.conn
            .execute(&format!("DELETE FROM {tablename}"), ())
            .unwrap();
    }

    /// initialize the database.
    fn init(&mut self) -> anyhow::Result<()> {
        // memory table.
        self.conn
            .execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS {} (
            id INTEGER PRIMARY KEY,
            pattern BLOB,
            class INT,
            subclass INT,
            UNIQUE(pattern)
            )",
                    self.memory_table()
                ),
                (),
            )
            .unwrap();
        // pattern table.
        self.conn
            .execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS {} (
            id INTEGER PRIMARY KEY,
            pattern BLOB,
            class INT,
            subclass INT,
            UNIQUE(pattern))",
                    self.pattern_table()
                ),
                (),
            )
            .unwrap();
        Ok(())
    }

    /// store a memory into dam.
    pub fn store_pattern<T: Float + std::fmt::Debug + std::iter::Sum + std::default::Default>(
        &mut self,
        memory: &Memory<T>,
    ) -> i64 {
        // covert vector of f32 to bytes.
        let data = memory
            .get_feature_vec()
            .iter()
            .map(|x| x.to_f32().unwrap())
            .collect::<Vec<_>>();
        let mut bytes = vec![0; data.len() * 4];
        LittleEndian::write_f32_into(&data, &mut bytes);

        assert!(memory.get_class().is_some(), "Expecting class of memory");

        self.conn
            .execute(
                &format!(
                    "INSERT INTO {} (pattern, class, subclass) VALUES (?1, ?2, ?3)",
                    self.memory_table()
                ),
                (bytes, memory.get_class().unwrap(), memory.get_subclass()),
            )
            .unwrap();
        self.conn.last_insert_rowid()
    }

    /// Convert buffer from database to vector of floats (f32).
    fn buf_to_f32_vec(buf: Vec<u8>) -> Vec<f32> {
        let mut v = vec![0f32; buf.len() / 4];
        LittleEndian::read_f32_into(&buf, &mut v);
        v
    }

    /// Delete stored memory for a given class.
    pub fn delete_stored_memories(&self, class: Option<usize>) -> Option<usize> {
        let mut q = format!("DELETE FROM {}", self.memory_table());
        if let Some(cls) = class {
            q.push_str(&format!(" WHERE class='{cls}'"));
        }
        match self.conn.execute(&q, []) {
            Ok(nrows) => Some(nrows),
            Err(e) => {
                tracing::error!("Failed to delete from table. {e}");
                None
            }
        }
    }

    /// Return all stored memories.
    pub fn fetch_stored_memories<
        T: Float + std::iter::Sum + std::fmt::Debug + std::default::Default,
    >(
        &self,
        class: Option<usize>,
        max_category: usize,
    ) -> Vec<Memory<T>> {
        let mut q = format!(
            "SELECT pattern, class, subclass FROM {}",
            self.memory_table()
        );
        if let Some(cls) = class {
            q.push_str(&format!(" WHERE class='{cls}'"));
        }
        let mut stmt = self.conn.prepare(&q).unwrap();
        let rows = stmt
            .query_map([], |row| {
                let sigma: Vec<T> = Self::buf_to_f32_vec(row.get(0).unwrap())
                    .into_iter()
                    .map(|x| T::from(x).unwrap())
                    .collect();
                Ok(Memory::new(
                    &sigma,
                    row.get(1).ok(),
                    row.get(2).ok(),
                    max_category as usize,
                ))
            })
            .unwrap();
        rows.flatten().collect()
    }

    /// Fetch a memory stored at given rowid
    pub fn fetch_pattern(&mut self, rowid: i64, max_category: usize) -> Option<Memory<f32>> {
        match self.conn.query_row(
            &format!(
                "SELECT pattern, class, subclass FROM {} WHERE id=?1",
                self.memory_table()
            ),
            [rowid],
            |row| {
                let sigma = Self::buf_to_f32_vec(row.get(0).unwrap());
                Ok(Memory::new(
                    &sigma,
                    row.get(1).ok(),
                    row.get(2).ok(),
                    max_category,
                ))
            },
        ) {
            Ok(pat) => Some(pat),
            Err(e) => {
                eprintln!("Failed to fetch row: {e}");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_sanity() {
        let mut db = Database::new("_test.sqlite", None);
        for _i in 0..100 {
            let pat: Memory<f32> = Memory::random(100, 10);
            let rowid = db.store_pattern(&pat);
            assert!(rowid >= 0);
            println!("... stored at {rowid}.");

            // fetch and compare.
            let pat2 = db.fetch_pattern(rowid, 10);
            assert!(pat2.is_some());
            assert_eq!(pat, pat2.unwrap());
        }
    }
}
