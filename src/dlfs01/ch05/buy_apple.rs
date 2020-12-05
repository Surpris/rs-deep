//! buy_apple
//!
//! buy apple

use super::layer_naive::*;

pub fn main() {
    println!("< buy_apple sub module >");
    let apple = 100f32;
    let num = 2f32;
    let tax = 1.1f32;

    let mut mul_apple_layer = MulLayer::new();
    let mut mul_tax_layer = MulLayer::new();

    // forward
    let apple_price = mul_apple_layer.forward(apple, num);
    let price = mul_tax_layer.forward(apple_price, tax);

    // backward
    let dprice = 1f32;
    let (dapple_price, dtax) = mul_tax_layer.backward(dprice);
    let (dapple, dnum) = mul_apple_layer.backward(dapple_price);

    println!("price: {}", price as u32);
    println!("dApple: {}", dapple);
    println!("dApple_num: {}", dnum as u32);
    println!("dTax: {}", dtax);
}
