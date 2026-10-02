use thiserror::Error;

/// Errors that can occur while splitting `X`/`y` data.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum SplitError {
    /// `test_size` must lie strictly between 0.0 and 1.0.
    #[error("test_size must be in the open interval (0.0, 1.0), got {0}")]
    InvalidTestSize(f64),

    /// `x` and `y` did not have the same number of samples.
    #[error("x and y must have the same length: x has {x_len}, y has {y_len}")]
    LengthMismatch { x_len: usize, y_len: usize },

    /// The input arrays contained no samples at all.
    #[error("input arrays must not be empty")]
    EmptyInput,

    /// There were too few samples to produce a non-empty train and test set.
    #[error(
        "with {n_samples} sample(s) and test_size={test_size}, at least one of the train/test sets would be empty"
    )]
    TooFewSamples { n_samples: usize, test_size: f64 },
}
