//! # ml-split
//!
//! Framework-agnostic train/test splitting for `X`/`y` machine-learning data.
//!
//! This crate does **not** depend on any particular tensor/matrix type —
//! it works on plain slices (`&[X]`, `&[Y]`) of anything that implements
//! `Clone`, so it drops into a workflow built on `Vec<Vec<f64>>`,
//! `ndarray`, `nalgebra`, `linfa`, `smartcore`, or your own hand-rolled
//! dataset struct.
//!
//! Two kinds of split are provided:
//!
//! * [`train_test_split`] — uniformly random split (like scikit-learn's
//!   `train_test_split` with default `stratify=None`).
//! * [`train_test_split_stratified`] — split that preserves each class's
//!   proportion of samples in both the train and test sets (like
//!   scikit-learn's `train_test_split(..., stratify=y)`). Use this for
//!   classification tasks, especially with imbalanced classes.
//!
//! Both are also available as pure index-based primitives
//! ([`train_test_split_indices`], [`train_test_split_indices_stratified`])
//! for when you'd rather index into your own data structure than have
//! this crate clone it for you.
//!
//! Enable the optional `ndarray` feature for `Array2<f64>` convenience
//! wrappers ([`ndarray_support`]).
//!
//! ## Example
//! ```
//! use ml_split::{train_test_split, train_test_split_stratified};
//!
//! let x = vec![vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 1.0], vec![4.0, 0.0]];
//! let y = vec![0, 1, 0, 1];
//!
//! // Plain random split, reproducible via a seed.
//! let split = train_test_split(&x, &y, 0.25, true, Some(0)).unwrap();
//! assert_eq!(split.x_train.len() + split.x_test.len(), 4);
//!
//! // Stratified split keeps the 0/1 class ratio similar in both halves.
//! let strat = train_test_split_stratified(&x, &y, 0.5, Some(0)).unwrap();
//! assert_eq!(strat.y_train.len(), 2);
//! assert_eq!(strat.y_test.len(), 2);
//! ```

mod error;
mod random;
mod stratified;

#[cfg(feature = "ndarray")]
pub mod ndarray_support;

pub use error::SplitError;
pub use random::{train_test_split, train_test_split_indices};
pub use stratified::{train_test_split_indices_stratified, train_test_split_stratified};

/// The result of splitting paired `x`/`y` data into train and test sets.
///
/// `x_train[i]` corresponds to `y_train[i]`, and likewise for the test
/// fields; positional correspondence between features and labels is
/// preserved by every split function in this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainTestSplit<X, Y> {
    pub x_train: Vec<X>,
    pub x_test: Vec<X>,
    pub y_train: Vec<Y>,
    pub y_test: Vec<Y>,
}

impl<X, Y> TrainTestSplit<X, Y> {
    /// Number of samples in the training set.
    pub fn n_train(&self) -> usize {
        self.x_train.len()
    }

    /// Number of samples in the test set.
    pub fn n_test(&self) -> usize {
        self.x_test.len()
    }

    /// Convenience accessor returning `(x_train, x_test, y_train, y_test)`
    /// as a plain tuple, e.g. for destructuring at the call site.
    pub fn into_tuple(self) -> (Vec<X>, Vec<X>, Vec<Y>, Vec<Y>) {
        (self.x_train, self.x_test, self.y_train, self.y_test)
    }
}
