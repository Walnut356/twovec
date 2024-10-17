# TwoVec

A data structure that can store 2 different types of objects in arbitrary orders in a packed, heap allocated array. Analogous to `Vec<Enum>`, but typically with less storage overhead.

## Example:

```rust
let x = 8u8;
let y = 20.0f32;
// type inference from .push_a and .push_b
let mut list = TwoVec::new();

// dedicated functions
list.push_a(x);
list.push_b(y);
// or through type inference
list.push(18.0);
list.push(17);
list.push(12);
list.push(1);

dbg!(&list);
// in bytes
dbg!(list.capacity());

// returns Some(val) if the value at that index is the correct type
let mut v: Option<u8> = list.get(0);
dbg!(v);
// returns None when the value is not the correct type
let mut w: Option<f32> = list.get(0);
dbg!(w);
w = list.get(1);
dbg!(w);
```

```txt
[Output]
&list = TwoVec[8, 20.0, 18.0, 17, 12, 1]
list.capacity() = 18
v = Some(8,)
w = None
w = Some(20.0,)
```