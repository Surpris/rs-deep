//! layer_naive
//!
//! naive layers

pub trait Layer {
    fn new() -> Self;
    fn forward(&mut self, x: f32, y: f32) -> f32;
    fn backward(&mut self, dout: f32) -> (f32, f32);
}

pub struct MulLayer {
    x: f32,
    y: f32,
}

impl Layer for MulLayer {
    fn new() -> Self {
        MulLayer { x: 0.0, y: 0.0 }
    }
    fn forward(&mut self, x: f32, y: f32) -> f32 {
        self.x = x;
        self.y = y;
        x * y
    }
    fn backward(&mut self, dout: f32) -> (f32, f32) {
        (self.y * dout, self.x * dout)
    }
}

pub struct AddLayer {
    x: f32,
    y: f32,
}

impl Layer for AddLayer {
    fn new() -> Self {
        AddLayer { x: 0.0, y: 0.0 }
    }
    fn forward(&mut self, x: f32, y: f32) -> f32 {
        self.x = x;
        self.y = y;
        x + y
    }
    fn backward(&mut self, dout: f32) -> (f32, f32) {
        (dout, dout)
    }
}
