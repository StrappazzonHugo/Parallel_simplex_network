use isera::*;
use isera::pivotrules::*;
use std::time::SystemTime;
use std::marker::PhantomData;
use std::env;

use crate::dimacs_parser::parsed_graph;
mod dimacs_parser;

fn main() {
    //let args: Vec<String> = std::env::args().collect();
    //let mut rng = rand::thread_rng();

    let args: Vec<String> = env::args().collect();
    let _file_path = &args[1];
    let nbproc:usize = args[2].parse::<usize>().unwrap();
    let kfactor:usize = args[3].parse::<usize>().unwrap();

    let (graph, sources, sinks) = parsed_graph::<i64>();

    let start = SystemTime::now();

    let pr: ParallelBlockSearch<i64> = ParallelBlockSearch{  phantom: PhantomData };
    let _min_cost_flow = min_cost(graph, sources, sinks, pr, nbproc, kfactor);
    match start.elapsed() {
        Ok(elapsed) => {
            print!(", time = {:?}, k = {:?}, nbproc = {:?}\n", (elapsed.as_millis() as f64 / 1000f64) as f64, kfactor, nbproc);
        }
        Err(e) => {
            println!("Error: {e:?}");
        }
    }
}
