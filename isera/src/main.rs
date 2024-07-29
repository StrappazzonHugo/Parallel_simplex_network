use clap::Parser;
use isera::pivotrules::*;
use isera::*;
//use rand::prelude::*;
use std::marker::PhantomData;

use crate::dimacs_parser::parsed_graph;
mod dimacs_parser;

fn main() {
    //let args: Vec<String> = std::env::args().collect();
    //let mut rng = rand::thread_rng();

    #[derive(Parser, Debug)]
    #[command(version, about, long_about = None)]
    struct Args {
        /// path to file
        #[arg(short, long)]
        filename: String,
        /// nb of processors
        #[arg(short, long, default_value_t = 1)]
        nbproc: usize,
        /// kfactor for block size, only for block search pivot rule
        #[arg(short, long, default_value_t = 1)]
        kfactor: usize,
    }

    let args = Args::parse();

    let _file_path: String = args.filename.clone();
    let nbproc = args.nbproc;
    let kfactor = args.kfactor;

    let (graph, sources, sinks) = parsed_graph::<i64>(_file_path);

    let sink_copy = sinks.clone();

    let _best: BestEligible<i64> = BestEligible {
        phantom: PhantomData,
    };

    let _seq_bs: BlockSearch<i64> = BlockSearch {
        phantom: PhantomData,
    };
    let _par_bs: ParallelBlockSearch<i64> = ParallelBlockSearch {
        phantom: PhantomData,
    };

    let (mut _state, _flow, _potential, _new_graph);
    if nbproc == 1 {
        (_state, _flow, _potential, _new_graph) =
            min_cost(graph, sources, sink_copy, _seq_bs, nbproc, kfactor);
        //_min_cost_flow = min_cost(graph, sources, sinks, _best, nbproc, kfactor);
    } else {
        (_state, _flow, _potential, _new_graph) =
            min_cost(graph, sources, sink_copy, _par_bs, nbproc, kfactor);
    }

    /* WARM START EXAMPLE
    println!("\nWarmStart");

    //COST MODIFICATIONS
    let mut rng = rand::thread_rng();
    state.edges_state.cost.iter_mut().for_each(|x| {
        let mut y: u64 = rng.gen();
        let mut z: u64 = rng.gen();
        y = y % 500;
        z = z % 100;
        if z < 50 {
            *x = *x + y as i64;
        } else {
            if *x > y as i64 {
                *x = *x - y as i64;
            } else {
                *x = *x + y as i64;
            }
        }
    });

    let new_costs = state.edges_state.cost.clone();

    let (_state2, _flow2, _potential2, _graph2) =
        min_cost_from_state(new_graph, state, sinks.clone(), new_costs, _seq_bs, 1, 1);

    println!("{:?}", potential.len());
    */
    //println!("{:?}", potential);
}
