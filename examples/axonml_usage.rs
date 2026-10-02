// Illustrates using ml-split with AxonML (https://github.com/AutomataNexus/AxonML).
//
// NOTE: this example is written against AxonML's documented API (README /
// crates.io as of v0.6.x) but was NOT compiled against the real `axonml`
// crate in this environment -- AxonML requires Rust 1.85+ (edition2024)
// and only rustc 1.75 was available here. Double-check exact method names
// (`slice_dim0`, `cat`, etc.) against the `axonml` version you pin.
//
// Run with: cargo run --example axonml_usage --features axonml_example
// (this example is not wired into Cargo.toml by default -- see comment below)

use axonml::prelude::*; // Tensor, from_vec, etc.
use ml_split::train_test_split_indices_stratified;

fn main() {
    // Your full dataset as one AxonML Tensor: [n_samples, n_features].
    let n_samples = 6;
    let n_features = 2;
    let x_data: Vec<f32> = vec![
        0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
    ];
    let x = Tensor::<f32>::from_vec(x_data, &[n_samples, n_features]).unwrap();

    // Labels can live in a plain Vec alongside the tensor -- ml-split
    // doesn't need them to be a Tensor at all.
    let y: Vec<i64> = vec![0, 0, 0, 1, 1, 1];

    // ml-split only needs to know the label for each row; it hands back
    // index lists, not a copy of your tensor.
    let (train_idx, test_idx) =
        train_test_split_indices_stratified(&y, 0.34, Some(42)).unwrap();

    // Gather rows by index. AxonML's `slice_dim0(i, i+1)` returns a
    // [1, n_features] view of row i; `Tensor::cat` stitches the selected
    // rows back into one [n_train, n_features] / [n_test, n_features]
    // tensor. If your AxonML version exposes a batch `index_select` /
    // `gather` helper, prefer that over this loop -- it'll be faster.
    let gather_rows = |idx: &[usize]| -> Tensor<f32> {
        let rows: Vec<Tensor<f32>> = idx.iter().map(|&i| x.slice_dim0(i, i + 1).unwrap()).collect();
        let row_refs: Vec<&Tensor<f32>> = rows.iter().collect();
        Tensor::cat(&row_refs, 0).unwrap()
    };

    let x_train = gather_rows(&train_idx);
    let x_test = gather_rows(&test_idx);
    let y_train: Vec<i64> = train_idx.iter().map(|&i| y[i]).collect();
    let y_test: Vec<i64> = test_idx.iter().map(|&i| y[i]).collect();

    println!("x_train shape: {:?}", x_train.shape());
    println!("x_test shape:  {:?}", x_test.shape());
    println!("y_train: {:?}", y_train);
    println!("y_test:  {:?}", y_test);

    // From here, wrap (x_train, y_train) / (x_test, y_test) in whatever
    // implements axonml_data::Dataset and hand it to axonml_data::DataLoader
    // for your training loop, same as any other AxonML dataset.
}
