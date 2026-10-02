use crate::error::SplitError;
use crate::random::make_rng;
use crate::TrainTestSplit;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::hash::Hash;

/*
Gaurav Sablok
gsablok@proton.me
 */

/// Compute a stratified train/test split as index lists.
///
/// Samples are grouped by their label in `y`, each group is split
/// independently at (approximately) `test_size`, and the resulting
/// index sets are combined and shuffled. This preserves each class's
/// proportion of the whole dataset in both the train and test sets,
/// which `train_test_split_indices` does not guarantee for imbalanced
/// classes.
///
/// A class with only a single member cannot be split (it would leave
/// either the train or the test set without any example of it); such a
/// sample is placed into the **train** set and skipped for the test set.
///
/// Returns `(train_indices, test_indices)`.
pub fn train_test_split_indices_stratified<Y: Eq + Hash + Clone>(
    y: &[Y],
    test_size: f64,
    seed: Option<u64>,
) -> Result<(Vec<usize>, Vec<usize>), SplitError> {
    if y.is_empty() {
        return Err(SplitError::EmptyInput);
    }
    if !(test_size > 0.0 && test_size < 1.0) {
        return Err(SplitError::InvalidTestSize(test_size));
    }

    // Group sample indices by their class label.
    let mut groups: HashMap<Y, Vec<usize>> = HashMap::new();
    for (i, label) in y.iter().enumerate() {
        groups.entry(label.clone()).or_default().push(i);
    }

    let mut rng = make_rng(seed);

    let mut train_idx: Vec<usize> = Vec::with_capacity(y.len());
    let mut test_idx: Vec<usize> = Vec::with_capacity(y.len());

    for (_label, mut idxs) in groups {
        idxs.shuffle(&mut rng);
        let class_size = idxs.len();

        let n_test = if class_size < 2 {
            // Can't give both sides a copy of a singleton class; keep it in train.
            0
        } else {
            let raw = (class_size as f64 * test_size).round() as usize;
            raw.clamp(1, class_size - 1)
        };

        let (test_part, train_part) = idxs.split_at(n_test);
        test_idx.extend_from_slice(test_part);
        train_idx.extend_from_slice(train_part);
    }

    if train_idx.is_empty() || test_idx.is_empty() {
        return Err(SplitError::TooFewSamples {
            n_samples: y.len(),
            test_size,
        });
    }

    // Undo the "grouped by class" ordering so rows aren't sorted by label.
    train_idx.shuffle(&mut rng);
    test_idx.shuffle(&mut rng);

    Ok((train_idx, test_idx))
}

/// Split paired `x` (features) and `y` (labels) slices into train and test
/// sets, preserving each class's proportion in `y` across both sets
/// ("stratified sampling"). Use this instead of [`crate::train_test_split`]
/// for classification tasks, especially with imbalanced classes.
///
/// `Y` must implement `Eq + Hash` so samples can be grouped by label.
///
/// # Example
/// ```
/// use ml_split::train_test_split_stratified;
///
/// // 8 samples: 6 of class 0, 2 of class 1.
/// let x: Vec<f64> = (0..8).map(|i| i as f64).collect();
/// let y = vec![0, 0, 0, 0, 0, 0, 1, 1];
///
/// let split = train_test_split_stratified(&x, &y, 0.25, Some(7)).unwrap();
///
/// // Both classes are represented in the test set, in proportion.
/// let test_class1 = split.y_test.iter().filter(|&&v| v == 1).count();
/// assert!(test_class1 >= 1);
/// ```
pub fn train_test_split_stratified<X: Clone, Y: Clone + Eq + Hash>(
    x: &[X],
    y: &[Y],
    test_size: f64,
    seed: Option<u64>,
) -> Result<TrainTestSplit<X, Y>, SplitError> {
    if x.len() != y.len() {
        return Err(SplitError::LengthMismatch {
            x_len: x.len(),
            y_len: y.len(),
        });
    }

    let (train_idx, test_idx) = train_test_split_indices_stratified(y, test_size, seed)?;

    Ok(TrainTestSplit {
        x_train: train_idx.iter().map(|&i| x[i].clone()).collect(),
        y_train: train_idx.iter().map(|&i| y[i].clone()).collect(),
        x_test: test_idx.iter().map(|&i| x[i].clone()).collect(),
        y_test: test_idx.iter().map(|&i| y[i].clone()).collect(),
    })
}
