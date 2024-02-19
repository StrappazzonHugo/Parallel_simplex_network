use main::min_cost;
use main::CustomEdgeIndices;
use petgraph::dot::Dot;
use petgraph::graph::*;

fn main() {
    let mut graph = DiGraph::<u32, CustomEdgeIndices<i32>>::new();
    let s = graph.add_node(0);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);
    let n4 = graph.add_node(4);
    let t = graph.add_node(5);

    graph.add_edge(
        n1,
        n2,
        CustomEdgeIndices {
            cost: (1),
            capacity: (3),
            flow: (0),
        },
    );
    graph.add_edge(
        n1,
        n4,
        CustomEdgeIndices {
            cost: (1),
            capacity: (2),
            flow: (0),
        },
    );
    graph.add_edge(
        n3,
        n2,
        CustomEdgeIndices {
            cost: (1),
            capacity: (6),
            flow: (0),
        },
    );
    graph.add_edge(
        n3,
        n4,
        CustomEdgeIndices {
            cost: (1),
            capacity: (2),
            flow: (0),
        },
    );
    graph.add_edge(
        n2,
        t,
        CustomEdgeIndices {
            cost: (1),
            capacity: (4),
            flow: (0),
        },
    );
    graph.add_edge(
        n4,
        t,
        CustomEdgeIndices {
            cost: (1),
            capacity: (8),
            flow: (0),
        },
    );
    graph.add_edge(
        s,
        n1,
        CustomEdgeIndices {
            cost: (1),
            capacity: (0),
            flow: (0),
        },
    );
    graph.add_edge(
        s,
        n2,
        CustomEdgeIndices {
            cost: (1),
            capacity: (7),
            flow: (0),
        },
    );
    graph.add_edge(
        s,
        n3,
        CustomEdgeIndices {
            cost: (1),
            capacity: (6),
            flow: (0),
        },
    );
    println!("{:?}", Dot::new(&graph));
    let min_cost_flow = min_cost(graph);
    println!("{:?}", Dot::new(&min_cost_flow));
}
