use isera::*;
use petgraph::graph::*;
use std::env;
use std::fs;



//TODO ARBITRARY INIT SOURCES AND SINKS 
pub fn parsed_graph<NUM:CloneableNum>() -> (
    DiGraph<u32, CustomEdgeIndices<NUM>>,
    Vec<(usize, NUM)>, // Vec<(usize, NUM)>
    Vec<(usize, NUM)>, // Vec<(usize, NUM)>
) where <NUM as std::str::FromStr>::Err: std::fmt::Debug{
    println!("starting parser...");
    let args: Vec<String> = env::args().collect();
    let file_path = &args[1];
    let mut sources: Vec<(usize, NUM)> = vec![]; 
    let mut sinks: Vec<(usize, NUM)> = vec![]; 

    let mut graph = DiGraph::<u32, CustomEdgeIndices<NUM>>::new();
    let contents = fs::read_to_string(file_path).expect("Should have been able to read the file");
    let as_vec: Vec<&str> = contents.lines().collect();
    let node_nb_str = as_vec[1].split('\t').collect::<Vec<&str>>()[0];
    let node_nb = node_nb_str
        .split(' ')
        .collect::<Vec<&str>>()
        .last()
        .expect("found")
        .parse::<u32>()
        .expect("found");
    for i in 0..node_nb+1 {
        graph.add_node(i);
    }
    println!("node nb = {:?}", node_nb);
    //let mut demand = 0;
    for x in contents.lines().skip(9) {
        let line = x.split('\t').collect::<Vec<&str>>()[1..].to_vec();
        if line.len() == 0 {
            continue;
        } 
        let source: usize = line[0].parse::<usize>().unwrap();
        let target: usize = line[1].parse::<usize>().unwrap();
        let capacity: NUM = line[2].parse::<NUM>().unwrap();
        let cost: NUM = line[3].parse::<NUM>().unwrap();
        graph.update_edge(
            NodeIndex::new(source),
            NodeIndex::new(target),
            CustomEdgeIndices {
                cost: (cost),
                capacity: (capacity),
                flow: (num_traits::zero()),
            },
        );
    }
    println!("end of parsing");

    (graph, sources, sinks)
}
