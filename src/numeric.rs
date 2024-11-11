//! Numerical helpers.

use num_traits::Float;

/// Inner product of two vectors.
pub fn inner_product<T: Float + std::fmt::Debug>(a: &[T], b: &[T]) -> T {
    assert!(a.len() == b.len(), "{} != {}", a.len(), b.len());
    let mut s = T::from(0.0).unwrap();
    for i in 0..a.len() {
        assert!(a[i].abs() <= T::one(), "{a:?}");
        assert!(b[i].abs() <= T::one(), "{b:?}");
        s = s + a[i] * b[i];
    }
    s
}

/// Softmax.
pub fn softmax<T: Float + std::fmt::Debug>(vals: &mut [T]) {
    let mut sum = T::zero();

    for elem in vals.iter_mut() {
        *elem = (*elem).exp();
        sum = sum + *elem;
        assert!(!sum.is_infinite());
    }

    for elem in vals.iter_mut() {
        *elem = (*elem) / sum;
        assert!(!(*elem).is_nan(), "sum={sum:?}");
    }

    // #[allow(clippy::needless_range_loop)]
    // for i in 0..vals.len() {
    //     vals[i] = vals[i] / sum;
    //     assert!(!vals[i].is_nan(), "sum={sum:?}");
    // }
}

/// Add some noise to a vector and return a new vector.
/// frac: fraction of elements to add some noise. The level of the noise is hard-wired in this
/// function.
/// FIXME: let user select noise level as well.
pub fn add_noise<T: Float + std::ops::AddAssign>(vec: &[T], frac: f32) -> Vec<T> {
    let mut res = vec![T::zero(); vec.len()];
    for i in 0..vec.len() {
        res[i] = vec[i];
        if rand::random::<f32>() < frac {
            res[i] += T::from(rand::random::<f32>() - 0.5f32).unwrap();
            res[i] = res[i] * T::from(0.5).unwrap();
        }
    }
    res
}

/// Same as [add_noise] but adds noise to a vector in-place.
pub fn add_noise_mut<T: Float>(vec: &mut [T], frac: f32) {
    for elem in vec.iter_mut() {
        if rand::random::<f32>() < frac {
            *elem = *elem + T::from(rand::random::<f32>() - 0.5f32).unwrap();
            *elem = *elem * T::from(0.5).unwrap();
        }
    }
}

#[inline(always)]
fn mean1<T: Float + for<'a> std::iter::Sum<&'a T>>(v: &[T], rescale: bool) -> T {
    let result = v.iter().sum::<T>() / T::from(v.len()).unwrap();
    if rescale {
        if result > T::zero() {
            T::one()
        } else {
            -T::one()
        }
    } else {
        result
    }
}

#[inline(always)]
fn median1<T: Float>(v: &[T]) -> T {
    let mut v1 = v.to_vec();
    v1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v1[v1.len() / 2]
}

/// mean of vector of vector.
pub fn mean<T: Float + for<'a> std::iter::Sum<&'a T>>(vecs: &[&[T]], rescale: bool) -> Vec<T> {
    let n = vecs[0].len();
    let mut result = vec![T::zero(); n];
    for i in 0..n {
        let x: Vec<T> = vecs.iter().map(|v| v[i]).collect();
        result[i] = mean1(&x, rescale);
    }
    result
}

/// median of vector of vector.
pub fn median<T: Float + for<'a> std::iter::Sum<&'a T>>(vecs: &[&[T]]) -> Vec<T> {
    let n = vecs[0].len();
    let mut result = vec![T::zero(); n];
    for i in 0..n {
        let x: Vec<T> = vecs.iter().map(|v| v[i]).collect();
        result[i] = median1(&x)
    }
    result
}

/// panics if there is nan in the vector.
pub fn must_not_have_nan<T: Float + std::fmt::Debug>(v: &[T]) {
    assert!(!v.iter().any(|e| e.is_nan()), "NaN in {v:?}");
}

/// really bad nameing of the function: alpha * x + (1-alpha) * y
pub fn scaled_mixing<T: Float>(x: &[T], y: &[T], alpha: T) -> Vec<T> {
    let mut res = x.to_vec();
    let one_minus_alpha = T::one() - alpha;
    for i in 0..x.len() {
        res[i] = one_minus_alpha * res[i] + alpha * y[i];
    }
    res
}

#[inline(always)]
pub fn perc(a: usize, b: usize) -> f32 {
    (100f32 * a as f32) / (b as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let a = vec![1.0, 1.0, 1.0];
        let b = vec![2.0, 2.0, 3.0];
        let data = vec![a.as_slice(), b.as_slice()];
        let mean = mean(&data, false);
        assert_eq!(mean, vec![1.5, 1.5, 2.0]);
    }
}
