use main::min_cost;
//use petgraph::dot::Dot;
use std::time::SystemTime;
mod parser_2;

fn main() {
    let graph = parser_2::parsed_graph(); 
    let start = SystemTime::now();
    let demand: f64 = 50000f64;
    println!(
        "node nb = {:?}, edge nb = {:?}, demand = {:?}",
        graph.node_count(),
        graph.edge_count(),
        demand
    );
    let _min_cost_flow = min_cost(graph, demand);
    match start.elapsed() {
        Ok(elapsed) => {
            println!("time = {:?}", (elapsed.as_millis()as f64/1000f64) as f64);
        }
        Err(e) => {
            println!("Error: {e:?}");
        }
    }
}
