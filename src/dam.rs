//! Dense Associative Memory.

use std::collections::HashMap;
use std::default::Default;
use std::path::Path;
use std::time::Instant;

use num_traits::Float;
use rand::{Rng, SeedableRng};

#[cfg(feature = "db")]
use crate::db::Database;

#[allow(unused_imports)]
use tracing::{debug, info, warn};

use crate::memory::{Memory, NUM_SUBCATEGORY};
use crate::numeric::*;

#[derive(Debug)]
pub struct DenseAssociativeMemory<T> {
    /// Stored pattern or memories.
    pub patterns: Vec<Memory<T>>,

    /// state vector.
    state: Vec<T>,

    /// scaling factor.
    beta: T,

    // max norm (across stored memroies).
    max_norm: T,

    /// runtime performance.
    runtime: RuntimePerf,

    #[allow(dead_code)]
    #[cfg(feature = "db")]
    pub db: Database,
}

impl<T: Float + std::iter::Sum + std::fmt::Debug + Default + for<'a> std::iter::Sum<&'a T>>
    DenseAssociativeMemory<T>
{
    pub fn new_from_db<P: AsRef<Path>>(beta: T, dbpath: P, table_prefix: Option<String>) -> Self {
        #[cfg(feature = "db")]
        let db =
            Database::new_with_named_table(dbpath, table_prefix.unwrap_or("default".to_string()));

        // TODO: Estimate beta.
        DenseAssociativeMemory {
            patterns: vec![],
            state: vec![],
            beta,
            max_norm: T::zero(),
            runtime: RuntimePerf::default(),
            #[cfg(feature = "db")]
            db,
        }
    }

    /// Create new DenseAssociativeMemory with given patterns.
    pub fn from_patterns<P: AsRef<Path>>(
        patterns: Vec<Memory<T>>,
        beta: T,
        dbpath: P,
        table_prefix: Option<String>,
    ) -> Self {
        let state_size = patterns.first().unwrap().get_data_size();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(10);

        let init_state: Vec<T> = (0..state_size)
            .map(|_| T::from(rng.gen::<f32>()).unwrap())
            .collect();

        assert!(
            init_state.len() == state_size,
            "{init_state:?}, state_size={state_size}",
        );

        let mut max_norm = T::zero();
        for pat in patterns.iter() {
            max_norm = max_norm.max(pat.norm_squared().sqrt());
        }

        #[cfg(feature = "db")]
        let db =
            Database::new_with_named_table(dbpath, table_prefix.unwrap_or("default".to_string()));

        // TODO: Estimate beta.
        DenseAssociativeMemory {
            patterns,
            state: init_state,
            beta,
            max_norm,
            runtime: RuntimePerf::default(),
            #[cfg(feature = "db")]
            db,
        }
    }

    /// Load memories from a database.
    pub fn load_from_db(&mut self, max_category: usize) {
        self.store_patterns(&mut self.db.fetch_stored_memories(None, max_category));
    }

    /// Clear data from internal tables.
    pub fn clear_tables(&mut self) {
        self.db.clear_tables()
    }

    /// Clear data from DAM.
    pub fn clear(&mut self) {
        self.patterns.clear();
        self.runtime.clear();
        self.db.clear_tables();
    }

    /// Set the internal table prefix.
    pub fn set_name(&mut self, name: &str) {
        self.db.set_table_prefix(name);
    }

    /// Add a [Memory] to [DenseAssociativeMemory].
    pub fn add_pattern(&mut self, memory: Memory<T>, save_to_database: bool) {
        if self.patterns.len() > 0 {
            assert_eq!(
                memory.get_data_size(),
                self.patterns[0].get_data_size(),
                "Stored patterns: {:?}",
                self.patterns
            );
        }
        let size = memory.get_data_size();
        if save_to_database {
            let rowid = self.db.store_pattern(&memory);
            tracing::info!("Stored at rowid={rowid}.");
        }
        self.patterns.push(memory);
        self.init_state(size);
    }

    fn reinit_state(&mut self, state_size: usize) {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(10);
        self.state = (0..state_size)
            .map(|_| T::from(rng.gen::<f32>()).unwrap())
            .collect();
    }

    fn init_state(&mut self, state_size: usize) {
        if self.state.len() != state_size {
            self.reinit_state(state_size);
        }
    }

    /// store given patterns into the dam.
    fn store_patterns(&mut self, patterns: &mut Vec<Memory<T>>) {
        if patterns.is_empty() {
            tracing::warn!("Trying to store 0 patterns. Won't do anything...");
            return;
        }

        // removes duplicate.
        // FIXME: Can this be taken care of by database layer?
        self.patterns.append(patterns);
        self.patterns
            .sort_unstable_by(|a, b| a.sigma().partial_cmp(b.sigma()).unwrap());

        self.patterns.dedup_by(|a, b| a.sigma() == b.sigma());
        let state_size = self.patterns.first().unwrap().get_data_size();

        tracing::info!("Loaded {} patterns.", patterns.len());

        self.init_state(state_size);

        assert!(
            self.state.len() == state_size,
            "{:?}, state_size={state_size}",
            self.state
        );
    }

    /// Return the number of categories stored in the DAM.
    fn num_categories(&self) -> usize {
        self.patterns[0].cardinality_of_class()
    }

    /// Compute max energy from max_norm
    #[inline(always)]
    pub fn max_energy(&self) -> T {
        (T::from(2.0).unwrap()) * self.max_norm.powi(2)
    }

    #[inline(always)]
    pub fn num_stored_patterns(&self) -> usize {
        self.patterns.len()
    }

    #[inline(always)]
    pub fn num_stored_categories(&self) -> HashMap<usize, usize> {
        let mut result = HashMap::new();
        for cat in self.patterns.iter().map(|pat| pat.get_category()).flatten() {
            result.insert(cat, result.get(&cat).unwrap_or(&0) + 1);
        }
        result
    }

    #[inline(always)]
    pub fn num_neurons(&self) -> usize {
        self.state.len()
    }

    #[inline(always)]
    pub fn get_pattern(&self, idx: usize) -> &Memory<T> {
        &self.patterns[idx]
    }

    /// Get the pattern vector (copy)
    #[inline(always)]
    pub fn get_pattern_vec(&self, idx: usize) -> Vec<T> {
        self.patterns[idx].sigma().to_vec()
    }

    /// size of patterns stored.
    #[inline(always)]
    fn get_pattern_feature_size(&self) -> usize {
        assert!(self.patterns.len() > 0, "There is no memory stored.");
        self.patterns[0].get_feature_size()
    }

    /// size of patterns stored.
    #[inline(always)]
    fn get_pattern_data_size(&self) -> usize {
        self.patterns[0].get_data_size()
    }

    #[inline(always)]
    fn lse(&self, pats: &[Memory<T>], sigma: &[T]) -> T {
        let res: T = pats
            .iter()
            .map(|pat| self.beta * inner_product(pat.sigma(), sigma).exp())
            .sum();
        res.ln() / self.beta
    }

    /// Compute the energy of DAM.
    #[inline(always)]
    pub fn energy(&self) -> T {
        let two = T::from(2.0).unwrap();
        let mut e: T = self.lse(&self.patterns, &self.state);
        e = e + inner_product(&self.state, &self.state) / two;
        e = e + T::from(self.num_stored_patterns()).unwrap() / self.beta;
        e = e + self.max_norm.powi(2) / two;

        assert!(e >= T::zero());
        assert!(e <= self.max_energy());
        e
    }

    fn apply_input(&mut self, input: &[T]) {
        assert!(self.state.len() == input.len());
        self.state = input.to_vec();

        // Reset the classification slice to 0.
        // FIXME: Try with random() if it improves the perf.
        for i in self.get_pattern_feature_size()..self.get_pattern_data_size() {
            // self.state[i] = -T::one();
            self.state[i] = T::zero();
        }
    }

    pub fn set_internal_state(&mut self, states: &[T]) {
        self.state = states.to_vec();
    }

    /// Train the dam.
    pub fn train_batch(&mut self, patterns: &[Memory<T>]) {
        tracing::info!("Training with {} samples", patterns.len());
        let mut unclassified = multimap::MultiMap::new();
        let mut n_unclassified = 0usize;
        for pattern in patterns {
            let cls = pattern.get_class();
            if cls.is_none() {
                tracing::error!("Memory class must be specified during training...");
                continue;
            }
            let result = self.apply_pattern(pattern);
            if result.is_none() {
                n_unclassified += 1;
                unclassified.insert(cls.unwrap(), pattern.get_feature_vec());
            }
        }

        tracing::info!(
            "A {:.1}% of total {} could not be classfied.",
            100f32 * (n_unclassified as f32) / (patterns.len() as f32),
            patterns.len()
        );

        for cls in unclassified.keys() {
            let batch = unclassified.get_vec(cls).unwrap();
            tracing::info!(" Size of batch for class {cls} is {}", batch.len());
            let avg_sigma = crate::numeric::mean(batch, true);
            tracing::info!("Average pattern: \n{}", crate::print_matrix(&avg_sigma));
            self.patterns.push(Memory::new(
                &avg_sigma,
                Some(*cls),
                None,
                self.num_categories(),
            ));
        }

        // Important. Reset the counts because we are traning.
        self.runtime.clear();
    }

    /// Apply a [Memory] and return its class.
    #[inline(always)]
    pub fn apply_pattern(&mut self, pattern: &Memory<T>) -> Option<usize> {
        self.apply_and_complete(pattern.sigma())
    }

    /// Apply a partial pattern and try to complete it. Returns it class.
    pub fn apply_and_complete(&mut self, pat: &[T]) -> Option<usize> {
        if self.state.len() == 0 {
            self.init_state(pat.len()); // just to be sure.
        }
        self.runtime.total += 1;
        let inst = Instant::now();

        assert_eq!(pat.len(), self.state.len(), "{pat:?} {:?}", self.state);
        self.apply_input(pat);

        // the fature subset must be the same for both pat and state.
        assert_eq!(
            self.state[0..self.get_pattern_feature_size()],
            pat[0..self.get_pattern_feature_size()]
        );

        let num_nrns = self.num_neurons();
        let num_pats = self.num_stored_patterns();
        // tracing::debug!(">> #neurons {num_nrns} and #pats {num_pats}");

        let mut p = vec![T::zero(); num_pats];

        #[allow(clippy::needless_range_loop)]
        for i in 0..num_pats {
            p[i] = self.beta * inner_product(self.patterns[i].sigma(), &self.state);
            assert!(!p[i].is_nan());
        }

        must_not_have_nan(&p);
        softmax(&mut p);
        must_not_have_nan(&p);

        for i in 0..num_nrns {
            let xi: Vec<T> = self.patterns.iter().map(|pat| pat.sigma()[i]).collect();
            self.state[i] = T::one().copysign(inner_product(&xi, &p));
        }

        self.runtime.time_spent_secs += inst.elapsed().as_secs_f32();

        tracing::debug!(">> state: {:?}", self.state);

        // If result is `None`, we could not classify it.
        let result = self.get_category();
        if result.is_none() {
            self.runtime.unclassified += 1;
        }
        result
    }

    /// Return which category.
    fn get_category(&self) -> Option<usize> {
        let pat_size = self.get_pattern_feature_size();
        // tracing::debug!(">> pat_size={pat_size} #state={}", self.state.len());
        self.state[pat_size..self.state.len() + 1 - NUM_SUBCATEGORY]
            .iter()
            .position(|&x| x == T::one())
    }

    /// get the state.
    pub fn get_state(&self) -> &[T] {
        &self.state
    }

    /// Get runtime report.
    pub fn runtime_report(&self) -> String {
        let rt = &self.runtime;
        format!(
            "stored memories={}, classified={}, correct={}%, wrong={}%, unclassified={}%, speed={:.1} patterns/sec",
            self.patterns.len(),
            rt.total,
            rt.accuracy(),
            100f32 * (rt.total - rt.correct - rt.unclassified) as f32 / (rt.total as f32),
            (100f32 * rt.unclassified as f32) / (rt.total as f32),
            rt.classification_rate_per_sec()
        )
    }

    /// Report the expected category. Not that we already count the un-classified.
    pub fn submit_expected_category(&mut self, expected: usize, verbose: bool) {
        let got = self.get_category();
        if Some(expected) == got {
            self.runtime.correct += 1;
            if verbose {
                print!("O");
            }
            return;
        }

        if got.is_none() && verbose {
            print!("?");
        } else if verbose {
            print!("X");
        }
    }

    /// Print the stored memories.
    pub fn print_memories(&self) {
        for pat in &self.patterns {
            println!("{}", pat.repr_polar_binary());
        }
    }
}

#[derive(Debug, Default)]
struct RuntimePerf {
    total: usize,
    correct: usize,
    unclassified: usize,
    time_spent_secs: f32,
}

impl RuntimePerf {
    /// How many classifications per second we have done.
    fn classification_rate_per_sec(&self) -> f32 {
        self.total as f32 / self.time_spent_secs
    }

    /// accuracy at runtime.
    fn accuracy(&self) -> f32 {
        100f32 * self.correct as f32 / self.total as f32
    }

    fn clear(&mut self) {
        self.total = 0;
        self.correct = 0;
        self.unclassified = 0;
        self.time_spent_secs = 0f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dam_simple() {
        let p1 = Memory::new(&[-1.0, -1.0, 1.0, 1.0], Some(0), None, 2);
        let p2 = Memory::new(&[-1.0, -1.0, -1.0, 1.0], Some(0), None, 2);
        let p3 = Memory::new(&[1.0, -1.0, -1.0, -1.0], Some(1), None, 2);
        let p4 = Memory::new(&[1.0, 1.0, -1.0, -1.0], Some(1), None, 2);
        let mut dam = DenseAssociativeMemory::from_patterns(
            vec![p1, p2, p3, p4],
            1.0,
            crate::db_default_name(),
            None,
        );

        dam.set_internal_state(&[0.1, -0.11, -0.91, 0.91, -1.0, -1.0]);
        assert_eq!(dam.num_stored_patterns(), 4);
        assert_eq!(
            dam.get_pattern(0),
            &Memory::new(&[-1.0, -1.0, 1.0, 1.0], Some(0), None, 2)
        );

        assert_eq!(
            dam.get_pattern(2),
            &Memory::new(&[1.0, -1.0, -1.0, -1.0], Some(1), None, 2)
        );
    }

    #[test]
    fn test_xor() {
        // Patterns without any class or category.
        let p1 = Memory::new(&[-1.0, -1.0], Some(0), None, 2);
        let p2 = Memory::new(&[-1.0, 1.0], Some(1), None, 2);
        let p3 = Memory::new(&[1.0, -1.0], Some(1), None, 2);
        let p4 = Memory::new(&[1.0, 1.0], Some(0), None, 2);

        // 4 patterns and beta=1 classes. This should not be able to learn XOR.
        let mut xor = DenseAssociativeMemory::from_patterns(
            vec![p1, p2, p3, p4],
            1.0,
            crate::db_default_name(),
            None,
        );
        for (pi, cl) in vec![(0, 0), (1, 1), (2, 1), (3, 0)] {
            let p1 = xor.get_pattern_vec(pi);
            let p2 = crate::numeric::add_noise(&p1, 0.5);
            let o = xor.apply_and_complete(&p2);
            println!("state={:?} input={p2:?} output={o:?}", xor.get_state());
            assert_eq!(o, Some(cl));
        }
    }

    #[test]
    fn test_recall() {
        // Patterns without any class or category.
        let p1 = Memory::new(&[-1.0, -1.0, -1.0, -1.0, -1.0], None, None, 0);
        let p2 = Memory::new(&[1.0, 1.0, 1.0, -1.0, -1.0], None, None, 0);
        // repeat a pattern. Recall should be stronger for this patern.
        let p3 = Memory::new(&[-1.0, -1.0, -1.0, 1.0, 1.0], None, None, 0);
        let p4 = Memory::new(&[-1.0, -1.0, -1.0, 1.0, 1.0], None, None, 0);

        // dam with
        let mut dam = DenseAssociativeMemory::from_patterns(
            vec![p1, p2, p3, p4],
            1.0,
            crate::db_default_name(),
            None,
        );
        println!("{dam:?}");
        assert!(dam.energy() <= dam.max_energy());

        // apply a pattern and recall it.
        for i in 0..dam.num_stored_patterns() {
            let pv = dam.get_pattern_vec(i);
            dam.apply_and_complete(&pv);
            assert_eq!(dam.get_state(), pv);
        }

        // apply a pattern + noise and recall it.
        for i in 0..dam.num_stored_patterns() {
            let pv = dam.get_pattern_vec(i);
            let p2 = add_noise(&pv, 0.25);
            dam.apply_and_complete(&p2);
            assert_eq!(dam.get_state(), pv);
        }
    }

    /// test MNIST database.
    #[test]
    fn test_learn_mnist() {
        let mut mnist = crate::data::MNIST::default();
        let n_patterns_training = 500;
        let n_patterns_testing = 200;

        let learn_mnist = mnist.train_patterns(0, n_patterns_training);
        let mut dam = DenseAssociativeMemory::from_patterns(
            learn_mnist,
            0.05,
            crate::db_default_name(),
            None,
        );
        assert_eq!(dam.num_stored_patterns(), n_patterns_training);

        let mut num_correct_classifications = 0usize;
        for pat in mnist.test_patterns(0, n_patterns_testing) {
            let vec = pat.sigma();
            let expected = pat.get_category();
            assert_eq!(vec.len(), 28 * 28 + 10 + crate::memory::NUM_SUBCATEGORY);
            println!("Applied\n{}", pat.repr_polar_image(Some(28)));
            let got = dam.apply_and_complete(&vec);
            if got == expected {
                num_correct_classifications += 1;
                println!("Successfully classified as digit {got:?}");
            } else {
                eprintln!("Wrong classification. Expected {expected:?} got {got:?}");
            }
        }
        println!(
            "Correct classifications {num_correct_classifications} out of {}",
            n_patterns_testing
        );
        assert!((num_correct_classifications as f32) / (n_patterns_testing as f32) > 0.7);
    }
}
