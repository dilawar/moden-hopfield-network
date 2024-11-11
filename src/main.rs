// src/main.rs

use clap::{Parser, Subcommand};
use modern_hopfield_network::{DenseAssociativeMemory, Memory};

#[derive(Debug, Parser)]
#[command(author, version, about, long_about)]
struct Cli {
    // Name of the DAM to operate on.
    #[arg(short, long)]
    name: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[inline(always)]
fn bool_str_to_polar_binary(data: &str) -> Vec<f32> {
    data.split(",")
        .map(|x| 2f32 * (x.parse::<f32>().unwrap() - 0.5f32))
        .collect()
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Train the named net.
    Train {
        /// csv of booleans (0 or 1).
        data: String,
        /// class of the pattern. Currently only integer values are supported.
        class: usize,
        /// maximum possible classes.
        total_classes: usize,
        /// Optional, subclass of the pattern.
        #[arg(short, long)]
        subclass: Option<usize>,
    },

    /// Show stored memories.
    Show {
        // Belonging to this class else show all.
        #[arg(short, long)]
        class: Option<usize>,
    },

    /// Delete stored memories.
    Delete {
        // Belonging to this class else show all.
        #[arg(short, long)]
        class: Option<usize>,
    },

    /// Clear everything.
    Clear {},

    /// Classify given pattern.
    Classify { data: String },
}

fn train(
    dam: &mut DenseAssociativeMemory<f32>,
    data: &str,
    class: usize,
    subclass: Option<usize>,
    max_categories: usize,
) {
    tracing::info!("Adding memory with class {class}/{subclass:?} ");
    let vec = bool_str_to_polar_binary(&data);
    let pat = Memory::new(&vec, Some(class), subclass, max_categories);
    assert!(pat.get_class().is_some());
    let rowid = dam.db.store_pattern(&pat);
    println!("Stored pattern with row id {rowid}");
}

fn classify(
    dam: &mut DenseAssociativeMemory<f32>,
    data: &str,
    max_category: usize,
) -> Option<usize> {
    dam.load_from_db(max_category);
    // dam.print_memories();
    let mut vec = bool_str_to_polar_binary(&data);
    vec.resize_with(dam.num_neurons(), || -1f32);
    tracing::debug!("Classifying pattern {vec:?}...");
    dam.apply_and_complete(&vec)
}

fn show(dam: &mut DenseAssociativeMemory<f32>, class: Option<usize>, max_category: usize) {
    tracing::info!("Showing stored memories with class {class:?}.");
    for pat in dam.db.fetch_stored_memories(class, max_category) {
        println!("{}", pat);
    }
}

fn delete(dam: &mut DenseAssociativeMemory<f32>, class: Option<usize>) {
    tracing::info!("Deleting stored memories with class {class:?}.");
    let nr = dam.db.delete_stored_memories(class);
    tracing::info!("{nr:?} rows were deleted.");
}

fn main() {
    tracing_subscriber::fmt::init();
    const MAX_CAT: usize = 20usize;
    let cli = Cli::parse();
    let mut dam =
        DenseAssociativeMemory::<f32>::new_from_db(0.05, modern_hopfield_network::db_default_name(), cli.name);
    match &cli.command {
        Some(Commands::Train {
            data,
            class,
            total_classes,
            subclass,
        }) => train(&mut dam, &data, *class, *subclass, *total_classes),
        Some(Commands::Classify { data }) => {
            let res = classify(&mut dam, &data, MAX_CAT);
            println!("{}", res.unwrap_or(usize::MAX));
        }
        Some(Commands::Show { class }) => show(&mut dam, *class, MAX_CAT),
        Some(Commands::Delete { class }) => delete(&mut dam, *class),
        Some(Commands::Clear {}) => dam.clear(),
        None => {}
    }
    // let pat = cli.train.data.split(",").map(|x| x.parse::<f32>()).collect::<Vec<f32>>();
    // tracing::info!("Got pattern {:?}", pat);
    tracing::debug!("All done");
}
