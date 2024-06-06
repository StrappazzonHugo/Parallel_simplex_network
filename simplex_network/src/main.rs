use main::min_cost;
use std::time::SystemTime;

use crate::parser::parsed_graph;
mod parser;

fn main() {
    //let args: Vec<String> = std::env::args().collect();
    //let mut rng = rand::thread_rng();

    let (graph, sources, sinks) = parsed_graph();

    let start = SystemTime::now();
    println!(
        "node nb = {:?}, edge nb = {:?}",// sources = {:?}, sinks = {:?}",
        graph.node_count(),
        graph.edge_count(),
        //sources,
        //sinks
    );
    let _min_cost_flow = min_cost(graph, sources, sinks);
    match start.elapsed() {
        Ok(elapsed) => {
            println!("time = {:?}", (elapsed.as_millis() as f64 / 1000f64) as f64);
        }
        Err(e) => {
            println!("Error: {e:?}");
        }
    }
}
