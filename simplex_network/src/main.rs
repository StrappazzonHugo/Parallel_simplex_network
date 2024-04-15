use main::min_cost;
use std::time::SystemTime;
mod parser;

fn main() {
    let graph = parser::parsed_graph();
    let start = SystemTime::now();
    let _min_cost_flow = min_cost(graph, 10);
    match start.elapsed() {
        Ok(elapsed) => {
            println!("time = {}", elapsed.as_micros());
        }
        Err(e) => {
            println!("Error: {e:?}");
        }

    }
}
