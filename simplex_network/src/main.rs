use main::min_cost;
use std::time::SystemTime;

use crate::parser::parsed_graph;
mod parser;

fn main() {
    //let args: Vec<String> = std::env::args().collect();
    //let mut rng = rand::thread_rng();

    let graph = parsed_graph();

    let start = SystemTime::now();
    let demand: i64 = 16;
    println!(
        "node nb = {:?}, edge nb = {:?}, demand = {:?}",
        graph.node_count(),
        graph.edge_count(),
        demand
    );
    let _min_cost_flow = min_cost(graph, demand);
    match start.elapsed() {
        Ok(elapsed) => {
            println!("time = {:?}", (elapsed.as_millis() as f64 / 1000f64) as f64);
        }
        Err(e) => {
            println!("Error: {e:?}");
        }
    }
}
