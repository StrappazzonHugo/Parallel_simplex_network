use isera::*;
use petgraph::graph::*;
use std::env;
use std::fs;

pub fn parsed_graph() -> DiGraph::<u32, CustomEdgeIndices<i32>> {
    println!("starting parser...");
    let args: Vec<String> = env::args().collect();
    let file_path = &args[1];

    let contents = fs::read_to_string(file_path).expect("Should have been able to read the file");

    let mut graph = DiGraph::<u32, CustomEdgeIndices<i32>>::new();
    //let mut demand = 0;

    contents.lines().for_each(|x| {
        if x.chars().nth(0) == Some('p') {
            let line = x.split(' ').collect::<Vec<&str>>()[1..].to_vec();
            //println!("{:?}", line);
            for i in 0..line[1].parse::<u32>().unwrap() {
                graph.add_node(i);
            }
        };
        if x.chars().nth(0) == Some('a') {
            let line = x.split(' ').collect::<Vec<&str>>()[1..].to_vec();
            //println!("{:?}", line);
            let source: usize = line[0].parse::<usize>().unwrap();
            let target: usize = line[1].parse::<usize>().unwrap();
            let capacity: i32 = line[3].parse::<i32>().unwrap();
            let cost: i32 = line[4].parse::<i32>().unwrap();
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
    graph
}
