//! Some helper functions.

use num_traits::Float;

pub fn print_matrix<T: Float>(vec: &[T]) -> String {
    let n = (vec.len() as f32).sqrt() as usize;
    repr_polar_image(vec, n)
}

/// Polar binary format.
pub fn repr_polar_image<T: Float>(vec: &[T], num_cols: usize) -> String {
    let line = repr_polar_binary(vec);
    let mut result = vec![];
    for i in 0..(line.len() / num_cols) {
        result.push(line[i * num_cols..(i + 1) * num_cols].to_string());
    }
    result.join("\n")
}

/// Polar binary format.
pub fn repr_polar_binary<T: Float>(vec: &[T]) -> String {
    vec.iter()
        .map(|x| if x > &T::zero() { '1' } else { '0' })
        .collect::<String>()
}
