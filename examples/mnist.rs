//! MNIST data.

#[allow(unused_imports)]
use tracing::{debug, info, warn};

use modern_hopfield_network::data::{create_mnist_dam, mnist_train_patterns};

fn main() {
    tracing_subscriber::fmt::init();

    let mut dam_none = create_mnist_dam("none", 1000);
    let mut dam_mean = create_mnist_dam("mean", 30000);
    let mut dam_median = create_mnist_dam("median", 30000);

    // Create auto-training regime with seed of 1000 patterns.
    let mut dam_auto = create_mnist_dam("mean", 10000);
    dam_auto.train_batch(&mnist_train_patterns(0, 10000));
    dam_auto.train_batch(&mnist_train_patterns(10000, 20000));
    dam_auto.train_batch(&mnist_train_patterns(20000, 40000));

    let num_test_patterns = 10000;
    for pat in mnist_train_patterns(0, num_test_patterns) {
        let _c0 = dam_none.apply_and_complete(&modern_hopfield_network::numeric::add_noise(
            pat.sigma(),
            0.25,
        ));
        dam_none.submit_expected_category(pat.get_category().unwrap(), false);

        let _c1 = dam_mean.apply_and_complete(pat.sigma());
        dam_mean.submit_expected_category(pat.get_category().unwrap(), false);

        let _c2 = dam_median.apply_and_complete(pat.sigma());
        dam_median.submit_expected_category(pat.get_category().unwrap(), false);

        let _c3 = dam_auto.apply_and_complete(pat.sigma());
        dam_auto.submit_expected_category(pat.get_category().unwrap(), true);
    }

    println!("\nResults");
    println!("none  : {}", dam_none.runtime_report());
    println!("mean  : {}", dam_mean.runtime_report());
    println!("median: {}", dam_median.runtime_report());
    println!("auto  : {}", dam_auto.runtime_report());
}
