//! Various data sources.

use std::fs::File;
use std::io::{prelude::*, BufReader};
use std::path::PathBuf;

use crate::memory::Memory;
use crate::numeric::{mean, median};

#[derive(Debug)]
pub struct MNIST {
    /// Train file.
    trainfile_csv: PathBuf,
    /// Test file.
    testfile_csv: PathBuf,
}

impl Default for MNIST {
    fn default() -> Self {
        let topdir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        Self::new(
            topdir.join("./data/mnist_train.csv"),
            topdir.join("./data/mnist_test.csv"),
        )
    }
}

impl MNIST {
    pub fn new(trainfile_csv: PathBuf, testfile_csv: PathBuf) -> Self {
        Self {
            trainfile_csv,
            testfile_csv,
        }
    }

    /// Get the training patterns.
    pub fn train_patterns(&mut self, start: usize, total: usize) -> Vec<Memory<f32>> {
        self.read_patterns(&self.trainfile_csv, start, total)
    }

    /// Get the testing patterns.
    pub fn test_patterns(&mut self, start: usize, total: usize) -> Vec<Memory<f32>> {
        self.read_patterns(&self.testfile_csv, start, total)
    }

    /// Read patterns from a given file.
    fn read_patterns(&self, csvfile: &PathBuf, start: usize, n: usize) -> Vec<Memory<f32>> {
        let file = File::open(csvfile).unwrap();
        let reader = BufReader::new(file);
        let mut results = vec![];
        // Additionally skip the header.
        for line in reader.lines().skip(1 + start).take(n).flatten() {
            let fs: Vec<u16> = line.split(',').map(|x| x.parse::<u16>().unwrap()).collect();
            let p = Memory::new(
                &fs[1..]
                    .iter()
                    .map(|&x| if x < 1 { -1.0 } else { 1.0 })
                    .collect::<Vec<f32>>(),
                Some(fs[0] as usize),
                None,
                10,
            );
            results.push(p);
        }
        results
    }
}

/// Create a DAM using MNIST dataset.
pub fn create_mnist_dam(
    averaging_scmeme: &str,
    num_train: usize,
) -> crate::DenseAssociativeMemory<f32> {
    let mut avg_pattern = vec![];
    let mut mnist = MNIST::default();

    let patterns = mnist.train_patterns(0, num_train);
    tracing::info!("Selected {} patterns from MNIST", num_train);
    let mut groups = multimap::MultiMap::new();
    let _ = patterns
        .iter()
        .for_each(|x| groups.insert(x.get_category().unwrap(), x.get_feature_vec()));
    if averaging_scmeme == "none" {
        avg_pattern = patterns;
    } else if averaging_scmeme == "mean" {
        for (k, v) in groups.iter_all() {
            let avg = mean(v, true);
            tracing::debug!("mean={:?}", avg);
            avg_pattern.push(Memory::new(&avg, Some(*k), None, 10));
        }
    } else if averaging_scmeme == "median" {
        for (k, v) in groups.iter_all() {
            let avg = median(v);
            tracing::debug!("median={:?}", avg);
            avg_pattern.push(Memory::new(&avg, Some(*k), None, 10));
        }
    } else {
        tracing::warn!("Unsupported averaging scheme {averaging_scmeme}");
    }
    crate::DenseAssociativeMemory::from_patterns(
        avg_pattern,
        0.05,
        crate::db::DEFAULT_DB_NAME,
        None,
    )
}

/// Read testing patterns from MNIST.
pub fn mnist_train_patterns(start: usize, end: usize) -> Vec<Memory<f32>> {
    let mut mnist = MNIST::default();
    mnist.train_patterns(start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnist_load() {
        let mut mnist = MNIST::default();
        let pats = mnist.train_patterns(0, 10);
        assert_eq!(pats.len(), 10);
        println!("Total train patterns: {:#?}", pats);
    }
}
