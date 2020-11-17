//! # man
//!
//!

struct Man {
    name: String,
}

impl Man {
    fn hello(&self) {
        println!("Hello {}", self.name);
    }

    fn goodbye(&self) {
        println!("Good-bye {}", self.name);
    }
}

pub fn main() {
    println!("< man sub module >");
    let m = Man {
        name: String::from("David"),
    };
    m.hello();
    m.goodbye();
}
