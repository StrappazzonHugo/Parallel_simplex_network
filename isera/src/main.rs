use isera::*;
use isera::pivotrules::*;
use std::time::SystemTime;
use std::marker::PhantomData;

use crate::dimacs_parser::parsed_graph;
mod dimacs_parser;

fn main() {
    //let args: Vec<String> = std::env::args().collect();
    //let mut rng = rand::thread_rng();

    let (graph, sources, sinks) = parsed_graph::<i64>();

    let start = SystemTime::now();
    println!(
        "node nb = {:?}, edge nb = {:?}", // sources = {:?}, sinks = {:?},
        graph.node_count(),
        graph.edge_count(),
        //sources,
        //sinks
    );

    let pr: ParallelBestEligible<i64> = ParallelBestEligible{  phantom: PhantomData };
    let _min_cost_flow = min_cost(graph, sources, sinks, pr, 8);
    match start.elapsed() {
        Ok(elapsed) => {
            println!("time = {:?}", (elapsed.as_millis() as f64 / 1000f64) as f64);
        }
        Err(e) => {
            println!("Error: {e:?}");
        }
    }
}
