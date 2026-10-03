# ml-split

Random and stratified **train/test splitting** for `X`/`y` machine-learning
data in Rust — framework-agnostic, so it works with `Vec<Vec<f64>>`,
`ndarray`, `nalgebra`, `linfa`, `smartcore`, or any custom dataset type.

## Why

Every ML workflow eventually needs to split a dataset into train/test
(or train/validation) subsets. This crate provides that as a small,
dependency-light, well-tested building block instead of every project
reinventing it:

- `train_test_split` — uniform random split (like scikit-learn's
  `train_test_split(..., stratify=None)`).
- `train_test_split_stratified` — split that preserves each class's
  proportion of samples in *both* the train and test sets (like
  scikit-learn's `train_test_split(..., stratify=y)`). Use this for
  classification, especially with imbalanced classes.

Both come in two flavors:

- A **value-based** API that clones your data into new `Vec`s
  (`x_train`, `x_test`, `y_train`, `y_test`).
- An **index-based** primitive (`train_test_split_indices`,
  `train_test_split_indices_stratified`) that only returns `usize`
  indices, so you can index into whatever structure you're already
  using (an `ndarray::Array2`, a `polars::DataFrame`, a memory-mapped
  file, ...) without an intermediate copy.

## Install

```toml
[dependencies]
ml-split = "0.1"
```

Add the optional `ndarray` feature if you want `Array2<f64>` convenience
wrappers:

```toml
[dependencies]
ml-split = { version = "0.1", features = ["ndarray"] }
```

## Usage

```rust
use ml_split::{train_test_split, train_test_split_stratified};

let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, (i * 2) as f64]).collect();
let y: Vec<i32> = vec![0, 0, 0, 0, 0, 0, 0, 1, 1, 1]; // imbalanced classes

// Plain random split. `seed` makes it reproducible; pass `None` for
// a non-deterministic shuffle.
let split = train_test_split(&x, &y, 0.3, true, Some(42)).unwrap();
println!("{} train, {} test", split.n_train(), split.n_test());

// Stratified split: keeps the 7:3 class ratio in both the train and
// test sets instead of leaving it to chance.
let strat = train_test_split_stratified(&x, &y, 0.3, Some(42)).unwrap();
let (x_train, x_test, y_train, y_test) = strat.into_tuple();
```

### Index-only API

Useful when you don't want the crate to clone your data for you, e.g.
because you're indexing rows of a matrix you already own:

```rust
use ml_split::train_test_split_indices;

let (train_idx, test_idx) = train_test_split_indices(1000, 0.2, true, Some(0)).unwrap();
// train_idx.len() == 800, test_idx.len() == 200
```

### With `ndarray` (requires the `ndarray` feature)

```rust,ignore
use ml_split::ndarray_support::train_test_split_array2_stratified;
use ndarray::{Array1, Array2};

let x: Array2<f64> = /* n_samples x n_features */;
let y: Array1<i32> = /* n_samples */;

let (x_train, x_test, y_train, y_test) =
    train_test_split_array2_stratified(&x, &y, 0.2, Some(0)).unwrap();
```

## Running the examples / tests

```bash
cargo test                                   # unit + integration + doctests
cargo run --example basic
cargo run --example ndarray_usage --features ndarray
```

Gaurav Sablok \
gsablok@proton.me
