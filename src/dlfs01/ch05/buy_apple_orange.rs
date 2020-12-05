//! buy_apple_orange
//!
//! buy apple

use super::layer_naive::*;

pub fn main() {
    println!("< buy_apple_orange sub module >");
    let apple = 100f32;
    let apple_num = 2f32;
    let orange = 100f32;
    let orange_num = 2f32;
    let tax = 1.1f32;

    let mut mul_apple_layer = MulLayer::new();
    let mut mul_orange_layer = MulLayer::new();
    let mut add_apple_orange_layer = AddLayer::new();
    let mut mul_tax_layer = MulLayer::new();

    // forward
    let apple_price = mul_apple_layer.forward(apple, apple_num);
    let orange_price = mul_orange_layer.forward(orange, orange_num);
    let all_price = add_apple_orange_layer.forward(apple_price, orange_price);
    let price = mul_tax_layer.forward(all_price, tax);

    // backward
    let dprice = 1f32;
    let (dall_price, dtax) = mul_tax_layer.backward(dprice);
    let (dapple_price, dorange_price) = add_apple_orange_layer.backward(dall_price);
    let (dorange, dorange_num) = mul_orange_layer.backward(dorange_price);
    let (dapple, dapple_num) = mul_apple_layer.backward(dapple_price);

    println!("price: {}", price as u32);
    println!("dApple: {}", dapple);
    println!("dApple_num: {}", dapple_num as u32);
    println!("dOrange: {}", dorange);
    println!("dOrange_num: {}", dorange_num as u32);
    println!("dTax: {}", dtax);
}
