use main::*;
use petgraph::graph::*;
use std::env;
use std::fs;

pub fn parsed_graph() -> (
    DiGraph<u32, CustomEdgeIndices<i64>>,
    Vec<(usize, i64)>, // Vec<(usize, NUM)>
    Vec<(usize, i64)>, // Vec<(usize, NUM)>
) {
    println!("starting parser...");
    let args: Vec<String> = env::args().collect();
    let file_path = &args[1];

    let contents = fs::read_to_string(file_path).expect("Should have been able to read the file");

    let mut graph = DiGraph::<u32, CustomEdgeIndices<i64>>::new();
    let mut sources: Vec<(usize, i64)> = vec![]; //TODO change to general number NUM
    let mut sinks: Vec<(usize, i64)> = vec![]; //TODO change to general number NUM

    contents.lines().for_each(|x| {
        if x.chars().nth(0) == Some('p') {
            let line = x.split(' ').collect::<Vec<&str>>()[1..].to_vec();
            for i in 0..line[1].parse::<u32>().unwrap() + 1 {
                graph.add_node(i);
            }
        };
        if x.chars().nth(0) == Some('n') {
            let line = x.split(' ').collect::<Vec<&str>>()[1..].to_vec();
            if line[1].parse::<i64>().unwrap().is_negative() {
                sinks.push((line[0].parse::<usize>().unwrap(), line[1].parse::<i64>().unwrap())); 
            } else {
                sources.push((line[0].parse::<usize>().unwrap(), line[1].parse::<i64>().unwrap())); 
            }
        };
        if x.chars().nth(0) == Some('a') {
            let line = x.split(' ').collect::<Vec<&str>>()[1..].to_vec();
            let source: usize = line[0].parse::<usize>().unwrap();
            let target: usize = line[1].parse::<usize>().unwrap();
            let capacity: i64 = line[3].parse::<i64>().unwrap();
            let cost: i64 = line[4].parse::<i64>().unwrap();
            graph.update_edge(
                NodeIndex::new(source),
                NodeIndex::new(target),
                CustomEdgeIndices {
                    cost: (cost),
                    capacity: (capacity),
                    flow: (0),
                },
            );
        };
    });
    (graph, sources, sinks)
}
