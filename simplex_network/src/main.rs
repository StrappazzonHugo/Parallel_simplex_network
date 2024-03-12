use main::min_cost;
use main::CustomEdgeIndices;
use main::State;
use petgraph::data::FromElements;
use petgraph::dot::Dot;
use petgraph::graph::*;
use petgraph::matrix_graph::DiMatrix;
use rand::Rng;
use std::time::SystemTime;

const NODE_NUMBER: u32 = 300;

fn main() {
    let mut rng = rand::thread_rng();
    let node_number = NODE_NUMBER * 2;

    let mut coord: Vec<(i64, i64)> = vec![(0, 0); (node_number + 1) as usize];

    let mut graph = DiMatrix::<u32, CustomEdgeIndices<i64>>::new();

    let source = graph.add_node(0); //source

    //One part
    for i in 1..NODE_NUMBER + 1 {
        let n = graph.add_node(i);
        let x: i64 = (rng.gen::<f32>() * 10000f32) as i64;
        let y: i64 = (rng.gen::<f32>() * 10000f32) as i64;
        coord[i as usize] = (x, y);
        graph.add_edge(
            source,
            n,
            CustomEdgeIndices {
                cost: (0),
                capacity: (100000000),
                flow: (0),
                state: State::LowerRestricted,
            },
        );
    }
    //the other part
    for i in NODE_NUMBER + 1..node_number + 1 {
        graph.add_node(i);
        let x: i64 = (rng.gen::<f32>() * 10000f32) as i64;
        let y: i64 = (rng.gen::<f32>() * 10000f32) as i64;
        coord[i as usize] = (x, y);
    }

    let sink = graph.add_node(node_number + 1);
    for i in NODE_NUMBER + 1..node_number + 1 {
        graph.add_edge(
            NodeIndex::new(i as usize),
            sink,
            CustomEdgeIndices {
                cost: (0),
                capacity: (100000000),
                flow: (0),
                state: State::LowerRestricted,
            },
        );
    }

    for i in 1..NODE_NUMBER + 1 {
        for j in NODE_NUMBER + 1..node_number + 1 {
            let cost = ((coord[i as usize].0 - coord[j as usize].0) as f32
                * (coord[i as usize].0 - coord[j as usize].0) as f32
                + (coord[i as usize].1 - coord[j as usize].1) as f32
                    * (coord[i as usize].1 - coord[j as usize].1) as f32)
                .sqrt() as i64;
            graph.add_edge(
                NodeIndex::new(i as usize),
                NodeIndex::new(j as usize),
                CustomEdgeIndices {
                    cost: (cost),
                    capacity: (10),
                    flow: (0),
                    state: State::LowerRestricted,
                },
            );
        }
    }

    let start = SystemTime::now();
    let demand: i64 = 10 ;
    println!("node nb = {:?}, demand = {:?}", graph.node_count(), demand * (NODE_NUMBER as i64));
    let _min_cost_flow = min_cost(graph, demand * (NODE_NUMBER as i64));
    match start.elapsed() {
        Ok(elapsed) => {
            println!("time = {}", elapsed.as_secs());
        }
        Err(e) => {
            println!("Error: {e:?}");
        }
    }
    //println!("{:?}", Dot::new(&_min_cost_flow));
}
