use main::min_cost;
use main::CustomEdgeIndices;
//use petgraph::dot::Dot;
use petgraph::graph::*;
use rand::Rng;
use std::time::SystemTime;

const GRID_SIZE: i32 = 500;

fn main() {
    //let args: Vec<String> = std::env::args().collect();
    let mut rng = rand::thread_rng();

    let mut graph = DiGraph::<u32, CustomEdgeIndices<i32>>::new();

    let source = graph.add_node(0); //source

    // first row of the grid
    let mut vertical_node: Vec<NodeIndex> = Vec::new();
    for i in 1..GRID_SIZE + 1 {
        let n = graph.add_node(i as u32);
        graph.add_edge(
            source,
            n,
            CustomEdgeIndices {
                cost: (0),
                capacity: (100000000),
                flow: (0),
            },
        );
        vertical_node.push(n);
    }

    for n in 0..vertical_node.len() {
        graph.add_edge(
            vertical_node[n],
            vertical_node[(n + 1) % vertical_node.len()],
            CustomEdgeIndices {
                cost: ((rng.gen::<f32>() * GRID_SIZE as f32) as i32),
                capacity: (10),
                flow: (0),
            },
        );
    }

    
    for i in 1..GRID_SIZE {
        vertical_node.clear();
        for j in 1..GRID_SIZE + 1 {
            let n = graph.add_node((i * GRID_SIZE + j) as u32);
            vertical_node.push(n);
            graph.add_edge(
                NodeIndex::new(n.index() - GRID_SIZE as usize),
                n,
                CustomEdgeIndices {
                    cost: ((rng.gen::<f32>() * GRID_SIZE as f32) as i32),
                    capacity: (10),
                    flow: (0),
                },
            );
        }
        for n in 0..vertical_node.len() {
            graph.add_edge(
                vertical_node[n],
                vertical_node[(n + 1) % vertical_node.len()],
                CustomEdgeIndices {
                    cost: ((rng.gen::<f32>() * GRID_SIZE as f32) as i32),
                    capacity: (10),
                    flow: (0),
                },
            );
        }
    }

    let sink = graph.add_node((GRID_SIZE * GRID_SIZE + 1) as u32);
    for i in (GRID_SIZE * (GRID_SIZE - 1) + 1)..(GRID_SIZE * GRID_SIZE)+1 {
        graph.add_edge(
            NodeIndex::new(i as usize),
            sink,
            CustomEdgeIndices {
                cost: (0),
                capacity: (100000000),
                flow: (0),
            },
        );
    }

    let start = SystemTime::now();
    let demand: i32 = 5;
    println!(
        "node nb = {:?}, edge nb = {:?}, demand = {:?}",
        graph.node_count(),
        graph.edge_count(),
        demand * (GRID_SIZE as i32)
    );
    let _min_cost_flow = min_cost(graph, demand * (GRID_SIZE as i32));
    match start.elapsed() {
        Ok(elapsed) => {
            println!("time = {:?}", (elapsed.as_millis() as f64 / 1000f64) as f64);
        }
        Err(e) => {
            println!("Error: {e:?}");
        }
    }
}
