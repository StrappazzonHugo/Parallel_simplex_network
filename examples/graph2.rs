use main::min_cost;
use main::CustomEdgeIndices;
use petgraph::data::FromElements;
use petgraph::dot::Dot;
use petgraph::graph::*;
use rand::Rng;
use std::time::SystemTime;

const NODE_NUMBER: u32 = 10;

fn main() {
    let mut graph = DiGraph::<u32, CustomEdgeIndices<i32>>::new();
    let s = graph.add_node(0);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);
    let n4 = graph.add_node(4);
    let t = graph.add_node(5);
    let N = graph.node_count();
    let u = Vec::from([
        0i32, 9, 0, 8, 0, 0, 0, 0, 5, 0, 2, 0, 0, 0, 0, 0, 0, 7, 0, 0, 3, 0, 6, 0, 0, 0, 0, 0, 0,
        6, 8, 0, 0, 0, 0, 0,
    ]);

    graph.add_edge(
        s,
        n1,
        CustomEdgeIndices {
            cost: (2),
            capacity: (u[s.index() * N + n1.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        s,
        n3,
        CustomEdgeIndices {
            cost: (1),
            capacity: (u[s.index() * N + n3.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n1,
        n2,
        CustomEdgeIndices {
            cost: (2),
            capacity: (u[n1.index() * N + n2.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n1,
        n4,
        CustomEdgeIndices {
            cost: (1),
            capacity: (u[n1.index() * N + n4.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n3,
        n2,
        CustomEdgeIndices {
            cost: (3),
            capacity: (u[n3.index() * N + n2.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n3,
        n4,
        CustomEdgeIndices {
            cost: (1),
            capacity: (u[n3.index() * N + n4.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n2,
        t,
        CustomEdgeIndices {
            cost: (3),
            capacity: (u[n2.index() * N + t.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n4,
        t,
        CustomEdgeIndices {
            cost: (4),
            capacity: (u[n4.index() * N + t.index()]),
            flow: (0),
        },
    );
    //println!("{:?}", Dot::new(&graph));
    let start = SystemTime::now();
    let min_cost_flow = min_cost(graph, 4);
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
