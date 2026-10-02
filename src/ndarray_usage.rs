// Run with: cargo run --example ndarray_usage --features ndarray
use ml_split::ndarray_support::{train_test_split_array2, train_test_split_array2_stratified};
use ndarray::{Array1, Array2};

/*
Gaurav Sablok
gsablok@proton.me
 */

fn main() {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
    )
    .unwrap();
    let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

    let (x_train, x_test, y_train, y_test) =
        train_test_split_array2(&x, &y, 0.34, true, Some(1)).unwrap();
    println!(
        "random split -> x_train shape {:?}, x_test shape {:?}",
        x_train.dim(),
        x_test.dim()
    );
    println!("y_train = {:?}, y_test = {:?}", y_train, y_test);

    let (x_train, x_test, y_train, y_test) =
        train_test_split_array2_stratified(&x, &y, 0.34, Some(1)).unwrap();
    println!(
        "stratified split -> x_train shape {:?}, x_test shape {:?}",
        x_train.dim(),
        x_test.dim()
    );
    println!("y_train = {:?}, y_test = {:?}", y_train, y_test);
}
