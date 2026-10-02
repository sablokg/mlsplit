//! Convenience wrappers for `ndarray::Array2<f64>` feature matrices,
//! enabled with the `ndarray` cargo feature. Purely additive: the core
//! crate never requires `ndarray`, so these exist only for users who
//! already store `X` as an `Array2`.

use crate::error::SplitError;
use crate::{train_test_split_indices, train_test_split_indices_stratified};
use ndarray::{Array1, Array2, Axis};
use std::hash::Hash;

/*
Gaurav Sablok
gsablok@proton.me
 */

/// Row-wise train/test split for an `ndarray::Array2<f64>` feature matrix
/// paired with an `Array1<Y>` target vector.
///
/// Returns `(x_train, x_test, y_train, y_test)`.
pub fn train_test_split_array2<Y: Clone>(
    x: &Array2<f64>,
    y: &Array1<Y>,
    test_size: f64,
    shuffle: bool,
    seed: Option<u64>,
) -> Result<(Array2<f64>, Array2<f64>, Array1<Y>, Array1<Y>), SplitError> {
    if x.nrows() != y.len() {
        return Err(SplitError::LengthMismatch {
            x_len: x.nrows(),
            y_len: y.len(),
        });
    }

    let (train_idx, test_idx) = train_test_split_indices(x.nrows(), test_size, shuffle, seed)?;
    Ok(gather(x, y, &train_idx, &test_idx))
}

/// Row-wise stratified train/test split for an `ndarray::Array2<f64>`
/// feature matrix paired with an `Array1<Y>` target vector.
///
/// Returns `(x_train, x_test, y_train, y_test)`.
pub fn train_test_split_array2_stratified<Y: Clone + Eq + Hash>(
    x: &Array2<f64>,
    y: &Array1<Y>,
    test_size: f64,
    seed: Option<u64>,
) -> Result<(Array2<f64>, Array2<f64>, Array1<Y>, Array1<Y>), SplitError> {
    if x.nrows() != y.len() {
        return Err(SplitError::LengthMismatch {
            x_len: x.nrows(),
            y_len: y.len(),
        });
    }

    let y_slice = y.as_slice().expect("y must be contiguous");
    let (train_idx, test_idx) = train_test_split_indices_stratified(y_slice, test_size, seed)?;
    Ok(gather(x, y, &train_idx, &test_idx))
}

fn gather<Y: Clone>(
    x: &Array2<f64>,
    y: &Array1<Y>,
    train_idx: &[usize],
    test_idx: &[usize],
) -> (Array2<f64>, Array2<f64>, Array1<Y>, Array1<Y>) {
    let x_train = x.select(Axis(0), train_idx);
    let x_test = x.select(Axis(0), test_idx);
    let y_train = Array1::from_iter(train_idx.iter().map(|&i| y[i].clone()));
    let y_test = Array1::from_iter(test_idx.iter().map(|&i| y[i].clone()));
    (x_train, x_test, y_train, y_test)
}
