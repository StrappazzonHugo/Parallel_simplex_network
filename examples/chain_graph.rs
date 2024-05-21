use isera::min_cost;
use isera::CustomEdgeIndices;
use petgraph::dot::Dot;
use petgraph::graph::*;
use rand::Rng;
use std::time::SystemTime;

const NODE_NUMBER: u32 = 5000;

fn main() {
    let mut rng = rand::thread_rng();
    let mut graph = DiGraph::<u32, CustomEdgeIndices<i64>>::new();

    let source = graph.add_node(0);

    //One part
    for i in 1..NODE_NUMBER + 1 {
        let n = graph.add_node(i);
        if i == 1 {
            graph.add_edge(
                source,
                n,
                CustomEdgeIndices {
                    cost: (1),
                    capacity: (1000),
                    flow: (0),
                    state: (1),
                },
            );
        } else {
            graph.add_edge(
                NodeIndex::new((i - 1) as usize),
                n,
                CustomEdgeIndices {
                    cost: (1),
                    capacity: (1000),
                    flow: (0),
                    state: (1),
                },
            );
        };
    }
    //the other part
    for i in NODE_NUMBER + 1..NODE_NUMBER * 2 + 1 {
        let n = graph.add_node(i);
        if i == NODE_NUMBER + 1 {
            graph.add_edge(
                source,
                n,
                CustomEdgeIndices {
                    cost: (1),
                    capacity: (1000),
                    flow: (0),
                    state: (1),
                },
            );
        } else {
            graph.add_edge(
                NodeIndex::new((i - 1) as usize),
                n,
                CustomEdgeIndices {
                    cost: (1),
                    capacity: (1000),
                    flow: (0),
                    state: (1),
                },
            );
            /*let v = (rng.gen::<f32>() * NODE_NUMBER as f32) as usize;
            let c: i64 = (rng.gen::<f32>() * 100f32) as i64;
            if c <= 1 {
                graph.add_edge(
                    n,
                    NodeIndex::new(v),
                    CustomEdgeIndices {
                        cost: (1),
                        capacity: (1000),
                        flow: (0),
                        state: (1),
                    },
                );
            }
            if c >= 99 {
                graph.add_edge(
                    NodeIndex::new(v),
                    n,
                    CustomEdgeIndices {
                        cost: (1),
                        capacity: (1000),
                        flow: (0),
                        state: (1),
                    },
                );
            }*/
        };
    }

    let sink = graph.add_node(NODE_NUMBER * 2 + 1);
    graph.add_edge(
        NodeIndex::new(NODE_NUMBER as usize),
        sink,
        CustomEdgeIndices {
            cost: (5),
            capacity: (1000),
            flow: (0),
            state: (1),
        },
    );
    graph.add_edge(
        NodeIndex::new(NODE_NUMBER as usize * 2),
        sink,
        CustomEdgeIndices {
            cost: (7),
            capacity: (1000),
            flow: (0),
            state: (1),
        },
    );

    let start = SystemTime::now();
    let demand: i64 = 900;
    println!("node nb = {:?}, demand = {:?}", graph.node_count(), demand,);
    //println!("{:?}", Dot::new(&graph));
    let _min_cost_flow = min_cost(graph, demand);
    match start.elapsed() {
        Ok(elapsed) => {
            println!("time = {}", elapsed.as_secs());
        }
        Err(e) => {
            println!("Error: {e:?}");
        }
    }
}
