use ml_split::{
    train_test_split, train_test_split_indices, train_test_split_indices_stratified,
    train_test_split_stratified, SplitError,
};
use std::collections::HashMap;
use std::hash::Hash;

/*
Gaurav Sablok
gsablok@proton.me
 */

fn make_xy(n: usize) -> (Vec<f64>, Vec<i32>) {
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<i32> = (0..n).map(|i| (i % 2) as i32).collect();
    (x, y)
}

#[test]
fn random_split_sizes_are_correct() {
    let (x, y) = make_xy(10);
    let split = train_test_split(&x, &y, 0.3, true, Some(1)).unwrap();
    assert_eq!(split.n_train(), 7);
    assert_eq!(split.n_test(), 3);
    assert_eq!(split.n_train() + split.n_test(), 10);
}

#[test]
fn random_split_is_reproducible_with_seed() {
    let (x, y) = make_xy(20);
    let a = train_test_split(&x, &y, 0.25, true, Some(42)).unwrap();
    let b = train_test_split(&x, &y, 0.25, true, Some(42)).unwrap();
    assert_eq!(a.x_train, b.x_train);
    assert_eq!(a.x_test, b.x_test);
    assert_eq!(a.y_train, b.y_train);
    assert_eq!(a.y_test, b.y_test);
}

#[test]
fn random_split_no_shuffle_preserves_order() {
    let (x, y) = make_xy(6);
    let split = train_test_split(&x, &y, 1.0 / 3.0, false, None).unwrap();
    // n_test = round(6 * 1/3) = 2, taken from the front since shuffle=false.
    assert_eq!(split.x_test, vec![0.0, 1.0]);
    assert_eq!(split.x_train, vec![2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn x_and_y_stay_paired_after_shuffle() {
    let (x, y) = make_xy(50);
    let split = train_test_split(&x, &y, 0.4, true, Some(9)).unwrap();
    for (xi, yi) in split.x_train.iter().zip(split.y_train.iter()) {
        assert_eq!((*xi as i32) % 2, *yi);
    }
    for (xi, yi) in split.x_test.iter().zip(split.y_test.iter()) {
        assert_eq!((*xi as i32) % 2, *yi);
    }
}

#[test]
fn rejects_invalid_test_size() {
    let (x, y) = make_xy(10);
    assert_eq!(
        train_test_split(&x, &y, 0.0, true, None).unwrap_err(),
        SplitError::InvalidTestSize(0.0)
    );
    assert_eq!(
        train_test_split(&x, &y, 1.0, true, None).unwrap_err(),
        SplitError::InvalidTestSize(1.0)
    );
    assert_eq!(
        train_test_split(&x, &y, -0.1, true, None).unwrap_err(),
        SplitError::InvalidTestSize(-0.1)
    );
}

#[test]
fn rejects_mismatched_lengths() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![0, 1];
    assert_eq!(
        train_test_split(&x, &y, 0.5, true, None).unwrap_err(),
        SplitError::LengthMismatch { x_len: 3, y_len: 2 }
    );
}

#[test]
fn rejects_empty_input() {
    let x: Vec<f64> = vec![];
    let y: Vec<i32> = vec![];
    assert_eq!(
        train_test_split(&x, &y, 0.5, true, None).unwrap_err(),
        SplitError::EmptyInput
    );
}

#[test]
fn indices_are_a_valid_partition() {
    let (train_idx, test_idx) = train_test_split_indices(37, 0.2, true, Some(3)).unwrap();
    let mut all: Vec<usize> = train_idx.iter().chain(test_idx.iter()).copied().collect();
    all.sort_unstable();
    assert_eq!(all, (0..37).collect::<Vec<_>>());
}

// ---- Stratified split tests ----

fn class_counts<Y: Eq + Hash + Clone>(labels: &[Y]) -> HashMap<Y, usize> {
    let mut counts = HashMap::new();
    for l in labels {
        *counts.entry(l.clone()).or_insert(0) += 1;
    }
    counts
}

#[test]
fn stratified_split_preserves_class_ratio_roughly() {
    // 100 samples: 80 class "a", 20 class "b".
    let mut y = vec!["a"; 80];
    y.extend(vec!["b"; 20]);
    let x: Vec<usize> = (0..y.len()).collect();

    let split = train_test_split_stratified(&x, &y, 0.2, Some(5)).unwrap();

    let test_counts = class_counts(&split.y_test);
    let train_counts = class_counts(&split.y_train);

    // ~20% test overall => class a: ~16 test/64 train, class b: ~4 test/16 train.
    assert_eq!(*test_counts.get("a").unwrap(), 16);
    assert_eq!(*test_counts.get("b").unwrap(), 4);
    assert_eq!(*train_counts.get("a").unwrap(), 64);
    assert_eq!(*train_counts.get("b").unwrap(), 16);
}

#[test]
fn stratified_split_keeps_every_class_in_both_sets_when_possible() {
    let mut y = vec![0; 6];
    y.extend(vec![1; 6]);
    y.extend(vec![2; 6]);
    let x: Vec<usize> = (0..y.len()).collect();

    let split = train_test_split_stratified(&x, &y, 0.5, Some(11)).unwrap();

    for class in [0, 1, 2] {
        assert!(split.y_train.iter().filter(|&&v| v == class).count() > 0);
        assert!(split.y_test.iter().filter(|&&v| v == class).count() > 0);
    }
}

#[test]
fn stratified_split_singleton_class_goes_to_train() {
    // class 2 has only one member and can't appear in both sets.
    let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 2];
    let x: Vec<usize> = (0..y.len()).collect();

    let split = train_test_split_stratified(&x, &y, 0.25, Some(2)).unwrap();

    assert_eq!(split.y_train.iter().filter(|&&v| v == 2).count(), 1);
    assert_eq!(split.y_test.iter().filter(|&&v| v == 2).count(), 0);
}

#[test]
fn stratified_split_x_y_stay_paired() {
    let mut y = vec![0; 30];
    y.extend(vec![1; 70]);
    let x: Vec<i32> = y.iter().map(|&v| v * 1000).collect(); // encode label into x for checking

    let split = train_test_split_stratified(&x, &y, 0.3, Some(123)).unwrap();
    for (xi, yi) in split.x_train.iter().zip(split.y_train.iter()) {
        assert_eq!(*xi, yi * 1000);
    }
    for (xi, yi) in split.x_test.iter().zip(split.y_test.iter()) {
        assert_eq!(*xi, yi * 1000);
    }
}

#[test]
fn stratified_indices_partition_all_samples() {
    let mut y = vec![0; 13];
    y.extend(vec![1; 29]);
    y.extend(vec![2; 8]);

    let (train_idx, test_idx) = train_test_split_indices_stratified(&y, 0.35, Some(77)).unwrap();
    let mut all: Vec<usize> = train_idx.iter().chain(test_idx.iter()).copied().collect();
    all.sort_unstable();
    assert_eq!(all, (0..y.len()).collect::<Vec<_>>());
}

#[test]
fn stratified_rejects_mismatched_lengths() {
    let x = vec![1, 2, 3];
    let y = vec![0, 1];
    assert_eq!(
        train_test_split_stratified(&x, &y, 0.5, None).unwrap_err(),
        SplitError::LengthMismatch { x_len: 3, y_len: 2 }
    );
}
