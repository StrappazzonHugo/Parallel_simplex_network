use main::min_cost;
use main::CustomEdgeIndices;
use petgraph::dot::Dot;
use petgraph::graph::*;
use std::time::{Duration, SystemTime};

fn main() {
    let mut graph = DiGraph::<u32, CustomEdgeIndices<f64>>::new();

    let n0 = graph.add_node(0);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);
    let n4 = graph.add_node(4);
    let n5 = graph.add_node(5);
    let n6 = graph.add_node(6);
    let n7 = graph.add_node(7);
    let n8 = graph.add_node(8);
    let n9 = graph.add_node(9);
    let n10 = graph.add_node(10);
    let n11 = graph.add_node(11);
    let n12 = graph.add_node(12);
    let n13 = graph.add_node(13);
    let n14 = graph.add_node(14);
    let n15 = graph.add_node(15);
    let n16 = graph.add_node(99);
    
    graph.add_edge(n0, n1, CustomEdgeIndices { cost: (1.), capacity: (2.), flow: (0.), }, );
    graph.add_edge(n0, n2, CustomEdgeIndices { cost: (2.), capacity: (6.), flow: (0.), }, );
    graph.add_edge(n0, n3, CustomEdgeIndices { cost: (2.), capacity: (4.), flow: (0.), }, );
    graph.add_edge(n0, n4, CustomEdgeIndices { cost: (1.), capacity: (2.), flow: (0.), }, );
    graph.add_edge(n0, n5, CustomEdgeIndices { cost: (1.), capacity: (3.), flow: (0.), }, );

    graph.add_edge(n1, n6, CustomEdgeIndices { cost: (1.), capacity: (3.), flow: (0.), }, );
    graph.add_edge(n2, n6, CustomEdgeIndices { cost: (2.), capacity: (8.), flow: (0.), }, );
    graph.add_edge(n2, n3, CustomEdgeIndices { cost: (1.), capacity: (2.), flow: (0.), }, );
    graph.add_edge(n3, n7, CustomEdgeIndices { cost: (1.), capacity: (10.), flow: (0.), }, );
    graph.add_edge(n3, n4, CustomEdgeIndices { cost: (2.), capacity: (2.), flow: (0.), }, );
    graph.add_edge(n4, n9, CustomEdgeIndices { cost: (1.), capacity: (3.), flow: (0.), }, );
    graph.add_edge(n5, n8, CustomEdgeIndices { cost: (2.), capacity: (7.), flow: (0.), }, );

    graph.add_edge(n6, n10, CustomEdgeIndices { cost: (1.), capacity: (11.), flow: (0.), }, );
    graph.add_edge(n6, n7, CustomEdgeIndices { cost: (2.), capacity: (7.), flow: (0.), }, );
    graph.add_edge(n7, n8, CustomEdgeIndices { cost: (2.), capacity: (4.), flow: (0.), }, );
    graph.add_edge(n8, n10, CustomEdgeIndices { cost: (1.), capacity: (7.), flow: (0.), }, );
    graph.add_edge(n8, n11, CustomEdgeIndices { cost: (2.), capacity: (8.), flow: (0.), }, );
    graph.add_edge(n9, n11, CustomEdgeIndices { cost: (1.), capacity: (5.), flow: (0.), }, );

    graph.add_edge(n10, n12, CustomEdgeIndices { cost: (1.), capacity: (3.), flow: (0.), }, );
    graph.add_edge(n10, n14, CustomEdgeIndices { cost: (2.), capacity: (6.), flow: (0.), }, );
    graph.add_edge(n11, n15, CustomEdgeIndices { cost: (1.), capacity: (10.), flow: (0.), }, );
    graph.add_edge(n11, n13, CustomEdgeIndices { cost: (3.), capacity: (3.), flow: (0.), }, );
    graph.add_edge(n13, n15, CustomEdgeIndices { cost: (2.), capacity: (8.), flow: (0.), }, );

    graph.add_edge(n14, n15, CustomEdgeIndices { cost: (2.), capacity: (3.), flow: (0.), }, );
    graph.add_edge(n14, n16, CustomEdgeIndices { cost: (1.), capacity: (10.), flow: (0.), }, );
    graph.add_edge(n15, n16, CustomEdgeIndices { cost: (2.), capacity: (12.), flow: (0.), }, );
    println!("{:?}", Dot::new(&graph));
    let start = SystemTime::now();
    let min_cost_flow = min_cost(graph, 10.0);
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
