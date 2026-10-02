use crate::error::SplitError;
use crate::TrainTestSplit;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/*
Gaurav Sablok
gsablok@proton.me
 */

/// Build an `StdRng`, seeded for reproducibility if `seed` is provided,
/// or seeded from entropy (i.e. non-deterministic) otherwise.
pub(crate) fn make_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

/// Compute the train/test split as index lists, without touching any data.
///
/// This is the primitive every other split function is built on. Because it
/// only returns `usize` indices, it works no matter what type your features
/// or labels are stored as (`Vec<Vec<f64>>`, `ndarray::Array2<f64>` rows,
/// a `polars` DataFrame's row indices, etc.) — just index into your own data
/// structure with the returned indices.
///
/// # Arguments
/// * `n_samples` - total number of samples available.
/// * `test_size` - fraction of samples to allocate to the test set, in `(0.0, 1.0)`.
/// * `shuffle` - whether to shuffle before splitting. If `false`, the first
///   `n_samples - n_test` indices become the train set and the rest become
///   the test set, in original order.
/// * `seed` - optional seed for a reproducible shuffle. Ignored if `shuffle` is `false`.
///
/// Returns `(train_indices, test_indices)`.
pub fn train_test_split_indices(
    n_samples: usize,
    test_size: f64,
    shuffle: bool,
    seed: Option<u64>,
) -> Result<(Vec<usize>, Vec<usize>), SplitError> {
    if n_samples == 0 {
        return Err(SplitError::EmptyInput);
    }
    if !(test_size > 0.0 && test_size < 1.0) {
        return Err(SplitError::InvalidTestSize(test_size));
    }

    let mut indices: Vec<usize> = (0..n_samples).collect();
    if shuffle {
        let mut rng = make_rng(seed);
        indices.shuffle(&mut rng);
    }

    let mut n_test = (n_samples as f64 * test_size).round() as usize;
    n_test = n_test.max(1);
    if n_test >= n_samples {
        return Err(SplitError::TooFewSamples {
            n_samples,
            test_size,
        });
    }

    let test_idx = indices[..n_test].to_vec();
    let train_idx = indices[n_test..].to_vec();
    Ok((train_idx, test_idx))
}

/// Split paired `x` (features) and `y` (labels/targets) slices into
/// train and test sets, uniformly at random.
///
/// `x` and `y` are matched by position: `x[i]` corresponds to `y[i]`.
/// Works with any element type that implements `Clone` — plain numbers,
/// `Vec<f64>` feature rows, structs, enums, strings, etc. — so it plugs
/// into whatever ML crate you're using without requiring a specific
/// matrix/tensor type.
///
/// # Example
/// ```
/// use ml_split::train_test_split;
///
/// let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
/// let y = vec![0, 1, 0, 1, 0];
///
/// let split = train_test_split(&x, &y, 0.2, true, Some(42)).unwrap();
/// assert_eq!(split.x_train.len(), 4);
/// assert_eq!(split.x_test.len(), 1);
/// assert_eq!(split.y_train.len(), 4);
/// assert_eq!(split.y_test.len(), 1);
/// ```
pub fn train_test_split<X: Clone, Y: Clone>(
    x: &[X],
    y: &[Y],
    test_size: f64,
    shuffle: bool,
    seed: Option<u64>,
) -> Result<TrainTestSplit<X, Y>, SplitError> {
    if x.len() != y.len() {
        return Err(SplitError::LengthMismatch {
            x_len: x.len(),
            y_len: y.len(),
        });
    }

    let (train_idx, test_idx) = train_test_split_indices(x.len(), test_size, shuffle, seed)?;

    Ok(TrainTestSplit {
        x_train: train_idx.iter().map(|&i| x[i].clone()).collect(),
        y_train: train_idx.iter().map(|&i| y[i].clone()).collect(),
        x_test: test_idx.iter().map(|&i| x[i].clone()).collect(),
        y_test: test_idx.iter().map(|&i| y[i].clone()).collect(),
    })
}
