use ml_split::{train_test_split, train_test_split_stratified};

fn main() {
    // Feature rows can be any type you like -- here, Vec<f64> rows.
    let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, (i * 2) as f64]).collect();
    let y: Vec<i32> = vec![0, 0, 0, 0, 0, 0, 0, 1, 1, 1]; // imbalanced classes

    println!("== plain random split ==");
    let split = train_test_split(&x, &y, 0.3, true, Some(42)).unwrap();
    println!("train: {} samples, test: {} samples", split.n_train(), split.n_test());
    println!("y_train = {:?}", split.y_train);
    println!("y_test  = {:?}", split.y_test);

    println!("\n== stratified split (keeps the 7:3 class ratio in both sets) ==");
    let strat = train_test_split_stratified(&x, &y, 0.3, Some(42)).unwrap();
    println!("train: {} samples, test: {} samples", strat.n_train(), strat.n_test());
    println!("y_train = {:?}", strat.y_train);
    println!("y_test  = {:?}", strat.y_test);
}
