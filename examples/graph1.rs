use main::min_cost;
use main::CustomEdgeIndices;
use petgraph::data::FromElements;
use petgraph::dot::Dot;
use petgraph::graph::*;
use rand::Rng;
use std::time::SystemTime;

const NODE_NUMBER: u32 = 10;

fn main() {
    let mut graph = DiGraph::<u32, CustomEdgeIndices<f32>>::new();
    let n0 = graph.add_node(0);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);
    let n4 = graph.add_node(4);
    let n5 = graph.add_node(5);
    graph.add_edge(n0, n1, CustomEdgeIndices {cost: (1.), capacity: (6.5), flow: (0.), state: (0.), },);
    graph.add_edge(n0, n3, CustomEdgeIndices {cost: (1.), capacity: (5.5), flow: (0.), state: (0.), },);
    graph.add_edge(n1, n2, CustomEdgeIndices {cost: (1.), capacity: (3.5), flow: (0.), state: (0.), },);
    graph.add_edge(n1, n4, CustomEdgeIndices {cost: (1.), capacity: (4.5), flow: (0.), state: (0.), },);
    graph.add_edge(n3, n2, CustomEdgeIndices {cost: (1.), capacity: (1.5), flow: (0.), state: (0.), },);
    graph.add_edge(n3, n4, CustomEdgeIndices {cost: (1.), capacity: (5.5), flow: (0.), state: (0.), },);
    graph.add_edge(n2, n5, CustomEdgeIndices {cost: (1.), capacity: (8.5), flow: (0.), state: (0.), },);
    graph.add_edge(n4, n5, CustomEdgeIndices {cost: (1.), capacity: (7.5), flow: (0.), state: (0.), },);
    //println!("{:?}", Dot::new(&graph));
    let start = SystemTime::now();
    let min_cost_flow = min_cost(graph, 8.0);
    match start.elapsed() {
       Ok(elapsed) => {
           println!("time = {}", elapsed.as_micros());
       }
       Err(e) => {
           println!("Error: {e:?}");
       }
   }
   println!("{:?}", Dot::new(&min_cost_flow));

}
