# Modern Hopfield Network

This crate implements Modern Hopfield Netowrk aka Dense Associative Memory [1].

## TODO

- [ ] Docs
- [ ] More tests
- [ ] (Stupid) Python bindings


## Usage

`cargo run --example mnist --release` runs the net on MNIST dataset.

On my machine, I get the following results. When I use 1000 images for training,
my accuracy is roughly 76%. With a small training set of 10, accuracy is about
70%! The speed of inference decreases linearly with number of stored patterns.


```
Results
none  : stored memories=1000, classified=10000, correct=76.62%, wrong=4.82%, unclassified=18.56%, speed=374.2 patterns/sec
mean  : stored memories=10, classified=10000, correct=69.14%, wrong=14.35%, unclassified=16.51%, speed=29304.4 patterns/sec
median: stored memories=10, classified=10000, correct=69.14%, wrong=14.35%, unclassified=16.51%, speed=29477.0 patterns/sec
auto  : stored memories=40, classified=10000, correct=69.6%, wrong=6.44%, unclassified=23.96%, speed=10354.1 patterns/sec
```


## References

1. Krotov, D., & Hopfield, J. J. (2016). Dense associative memory for pattern
   recognition. Advances in neural information processing systems, 29.
   https://proceedings.neurips.cc/paper_files/paper/2016/file/eaae339c4d89fc102edd9dbdb6a28915-Paper.pdf

# Demo

A simple demo of pattern recognition: https://meena.subcom.link/
