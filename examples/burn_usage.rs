// Illustrates using ml-split with Burn (https://burn.dev / https://docs.rs/burn).
//
// NOTE: this example is written against Burn's documented `Dataset` /
// `InMemDataset` API (burn::data::dataset) but was NOT compiled against the
// real `burn` crate in this environment -- Burn's dependency tree requires
// Rust 1.85+ (edition2024) and only rustc 1.75 was available here.
// Double-check import paths/types against the Burn version you pin.
//
// Run with: cargo run --example burn_usage --features burn_example

use burn::data::dataset::{Dataset, InMemDataset};
use ml_split::train_test_split_stratified;

/// One row of your dataset. Burn's `Dataset<I>` trait works with any `I`,
/// so this is exactly the same struct you'd hand to a `Batcher` -- ml-split
/// doesn't need it to be anything special, just `Clone`.
#[derive(Debug, Clone)]
struct FeatureItem {
    features: Vec<f32>,
    label: i64,
}

fn main() {
    // Your raw dataset, e.g. loaded from CSV/JSON/SQLite via Burn's dataset
    // helpers, or built by hand as here.
    let items: Vec<FeatureItem> = vec![
        FeatureItem { features: vec![0.0, 0.1], label: 0 },
        FeatureItem { features: vec![1.0, 1.1], label: 0 },
        FeatureItem { features: vec![2.0, 2.1], label: 0 },
        FeatureItem { features: vec![3.0, 3.1], label: 0 },
        FeatureItem { features: vec![4.0, 4.1], label: 1 },
        FeatureItem { features: vec![5.0, 5.1], label: 1 },
        FeatureItem { features: vec![6.0, 6.1], label: 1 },
        FeatureItem { features: vec![7.0, 7.1], label: 1 },
    ];
    let labels: Vec<i64> = items.iter().map(|it| it.label).collect();

    // Because Burn's items already bundle features + label together, we can
    // split the *items themselves* as `X`, using `labels` purely to decide
    // the stratification -- no separate feature/label split step needed.
    let split = train_test_split_stratified(&items, &labels, 0.25, Some(0)).unwrap();

    // Each half drops straight into an InMemDataset, which already
    // implements Burn's `Dataset<FeatureItem>` trait.
    let train_dataset: InMemDataset<FeatureItem> = InMemDataset::new(split.x_train);
    let test_dataset: InMemDataset<FeatureItem> = InMemDataset::new(split.x_test);

    println!("train: {} items, test: {} items", train_dataset.len(), test_dataset.len());
    if let Some(first) = train_dataset.get(0) {
        println!("first train item: {:?}", first);
    }

    // From here, feed train_dataset / test_dataset into a `Batcher` and
    // `DataLoaderBuilder` as usual, e.g.:
    //
    // let dataloader_train = DataLoaderBuilder::new(batcher.clone())
    //     .batch_size(32)
    //     .shuffle(seed)
    //     .build(train_dataset);
}
