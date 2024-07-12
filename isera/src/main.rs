use clap::Parser;
use isera::pivotrules::*;
use isera::*;
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
    let file = args.filename.clone();
    let _file_path: String = args.filename;
    let nbproc = args.nbproc;
    let kfactor = args.kfactor;

    let (graph, sources, sinks) = parsed_graph::<i64>(_file_path);

    let _best: BestEligible<i64> = BestEligible {
        phantom: PhantomData,
    };

    let _seq_bs: BlockSearch<i64> = BlockSearch {
        phantom: PhantomData,
    };
    let _par_bs: ParallelBlockSearch<i64> = ParallelBlockSearch {
        phantom: PhantomData,
    };
    print!("{:?}, ", file);
    let _min_cost_flow;
    if nbproc == 1 {
        _min_cost_flow = min_cost(graph, sources, sinks, _seq_bs, nbproc, kfactor);
        //_min_cost_flow = min_cost(graph, sources, sinks, _best, nbproc, kfactor);
    } else {
        _min_cost_flow = min_cost(graph, sources, sinks, _par_bs, nbproc, kfactor);
    }
    print!("k = {:?}, nbproc = {:?}\n", kfactor, nbproc);
}
