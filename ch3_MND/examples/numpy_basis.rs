use nalgebra::{Matrix2x3, Matrix2, Vector3};

fn main() {
    let x = Vector3::new(1,2,3).transpose();
    println!("{}", x);
    // print the column size of x
    println!("{}", x.shape().1);
    // print the row size of x
    println!("{}", x.shape().0);

    let w = Matrix2x3::new(1,2,3,4,5,6);
    println!("{}", w);
    println!("{}", w.shape().1);
    println!("{}", w.shape().0);

    let x = Matrix2x3::new(0, 1,2,3,4,5);
    println!("{}", w + x);
    // element-wise multiplication
    println!("{}", w.component_mul(&x));

    let a = Vector3::new(1, 2, 3).transpose();
    let b = Vector3::new(4, 5, 6).transpose();
    let y = a.dot(&b);
    println!("{}", y);

    let a  = Matrix2::new(1, 2, 3, 4);
    let b = Matrix2::new(5, 6, 7, 8);
    let y = a * b;
    println!("{}", y);
}