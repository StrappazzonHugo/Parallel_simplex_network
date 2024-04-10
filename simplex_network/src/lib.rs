use itertools::Itertools;
use num_traits::identities::one;
use num_traits::identities::zero;
use num_traits::Num;
use petgraph::algo::bellman_ford;
use petgraph::graph::EdgeIndex;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::stable_graph::IndexType;
//use rayon::prelude::*;
//use rayon::ThreadPoolBuilder;

#[derive(Debug, Clone)]
struct SPTree {
    out_base: Vec<EdgeIndex>,
    pred: Vec<Option<NodeIndex>>,
    thread: Vec<NodeIndex>,
    rev_thread: Vec<NodeIndex>,
    depth: Vec<usize>,
}

#[derive(Clone, Debug, Copy, PartialEq, PartialOrd)]
pub struct CustomEdgeIndices<NUM: CloneableNum> {
    pub cost: NUM,
    pub capacity: NUM,
    pub flow: NUM,
    pub state: NUM,
}

pub trait CloneableNum:
    Num
    + PartialOrd
    + Clone
    + Copy
    + PartialEq
    + std::fmt::Debug
    + num_traits::bounds::Bounded
    + std::ops::AddAssign
    + std::ops::SubAssign
    + Sync
    + Send
    + Sized
{
}

impl CloneableNum for i8 {}
impl CloneableNum for i16 {}
impl CloneableNum for i32 {}
impl CloneableNum for i64 {}
impl CloneableNum for i128 {}
impl CloneableNum for isize {}
impl CloneableNum for f32 {}
impl CloneableNum for f64 {}

//Initialization with artificial root
fn initialization<'a, NUM: CloneableNum>(
    graph: &'a mut DiGraph<u32, CustomEdgeIndices<NUM>>,
    demand: NUM,
) -> SPTree {
    let initial_number_of_node: u32 = (graph.node_count() - 1) as u32;
    let sink_id = graph
        .node_indices()
        .find(|&x| (graph.node_weight(x).unwrap() == &initial_number_of_node))
        .unwrap();

    let mut non_tree_arcs: Vec<EdgeIndex> = Vec::new();

    let mut big_value: NUM = zero();
    for _ in 0..100 {
        big_value += demand;
    }
    graph.clone().edge_references().for_each(|x| {
        graph[x.id()].state = num_traits::one();
        non_tree_arcs.push(x.id());
    });

    let artificial_root = graph.add_node(graph.node_count() as u32);
    for node in graph.node_indices() {
        if node == artificial_root {
            continue;
        }
        if node == sink_id {
            graph.add_edge(
                artificial_root,
                node,
                CustomEdgeIndices {
                    cost: big_value,
                    capacity: big_value,
                    flow: demand,
                    state: zero(),
                },
            );
        } else if node == NodeIndex::new(0) {
            graph.add_edge(
                node,
                artificial_root,
                CustomEdgeIndices {
                    cost: big_value,
                    capacity: big_value,
                    flow: demand,
                    state: zero(),
                },
            );
        } else {
            graph.add_edge(
                node,
                artificial_root,
                CustomEdgeIndices {
                    cost: big_value,
                    capacity: big_value,
                    flow: zero(),
                    state: zero(),
                },
            );
        }
    }

    let mut predecessors: Vec<Option<NodeIndex>> =
        vec![Some(artificial_root); graph.node_count() - 1];
    predecessors.push(None);

    let mut depths: Vec<usize> = vec![1; graph.node_count()];
    depths[graph.node_count() - 1] = 0;

    let mut thread_id: Vec<NodeIndex> = vec![NodeIndex::new(0); graph.node_count()];
    for i in 0..thread_id.len() - 1 {
        thread_id[i] = NodeIndex::new(i + 1);
    }
    let mut rev_thread_id: Vec<NodeIndex> =
        vec![NodeIndex::new(graph.node_count() - 1); graph.node_count()];
    for i in 1..rev_thread_id.len() {
        rev_thread_id[i] = NodeIndex::new(i - 1);
    }

    SPTree {
        out_base: non_tree_arcs,
        pred: predecessors,
        thread: thread_id,
        rev_thread: rev_thread_id,
        depth: depths,
    }
}

//New version of compute_node_potentials using tree form of sptree.t to compute them in order
//they are sorted by distance to root/depth in the tree and starting from the root we compute each
//potential from the one of its predecessor starting with pi[0] = 0 we have :
//
//  pi[id] = if arc(id, pred(id))
//              cost(id, pred(id)) + pi[pred(id)]
//           else if  arc(pred(id), id)
//              pi[pred(id)] - cost(pred(id), id)
//
fn compute_node_potentials<'a, NUM: CloneableNum>(
    graph: &mut DiGraph<u32, CustomEdgeIndices<NUM>>,
) -> Vec<NUM> {
    let mut pi: Vec<NUM> = vec![zero(); graph.node_count()];
    let mut edges: Vec<(u32, u32, f32)> = graph
        .edges_directed(NodeIndex::new(graph.node_count() - 1), Incoming)
        .map(|x| (x.source().index() as u32, x.target().index() as u32, 1f32))
        .collect();
    let rest: Vec<(u32, u32, f32)> = graph
        .edges_directed(NodeIndex::new(graph.node_count() - 1), Outgoing)
        .map(|x| (x.source().index() as u32, x.target().index() as u32, 1f32))
        .collect();
    edges.push(rest[0]);

    let temp_graph = Graph::<(), f32, Undirected>::from_edges(edges);

    let path = bellman_ford(&temp_graph, NodeIndex::new(graph.node_count() - 1)).unwrap();
    let distances: Vec<i32> = path.distances.iter().map(|x| x.round() as i32).collect();

    //decompose Path type from petgraph into vector
    let dist_pred: Vec<(&i32, &Option<NodeIndex>)> =
        distances.iter().zip(path.predecessors.iter()).collect();

    //vector of the form : Vec<(NodeIndex, DistanceToRoot, predecessorIndex)>
    let mut id_dist_pred: Vec<(usize, &i32, &Option<NodeIndex>)> = dist_pred
        .iter()
        .enumerate()
        .map(|x| (x.0, x.1 .0, x.1 .1))
        .collect();
    id_dist_pred = id_dist_pred.into_iter().sorted_by_key(|&x| x.1).collect();
    id_dist_pred.iter().skip(1).for_each(|&(id, _, pred)| {
        let edge = graph.find_edge(pred.unwrap(), NodeIndex::new(id));
        pi[id] = if edge != None {
            pi[pred.unwrap().index()] - graph[edge.unwrap()].cost
        } else {
            graph
                .edge_weight(graph.find_edge(NodeIndex::new(id), pred.unwrap()).unwrap())
                .unwrap()
                .cost
                + pi[pred.unwrap().index()]
        }
    });
    pi
}

fn update_node_potentials<'a, NUM: CloneableNum>(
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    mut potential: Vec<NUM>,
    sptree: &mut SPTree,
    entering_arc: EdgeIndex,
    leaving_arc: EdgeIndex,
) -> Vec<NUM> {
    if entering_arc == leaving_arc {
        return potential;
    }
    let (k, l) = (
        graph.raw_edges()[entering_arc.index()].source(),
        graph.raw_edges()[entering_arc.index()].target(),
    );
    let mut change: NUM = zero();
    let start: NodeIndex;
    if sptree.pred[k.index()] == Some(l) {
        change += get_reduced_cost_edgeindex(graph, entering_arc, &potential);
        start = k;
    } else {
        change -= get_reduced_cost_edgeindex(graph, entering_arc, &potential);
        start = l;
    }
    let mut current_node = sptree.thread[start.index()];
    potential[start.index()] += change;
    while sptree.depth[current_node.index()] > sptree.depth[start.index()] {
        potential[current_node.index()] += change;
        current_node = sptree.thread[current_node.index()];
    }
    potential
}

fn find_cycle_with_arc<'a, NUM: CloneableNum>(
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    sptree: &mut SPTree,
    entering_arc: EdgeIndex,
) -> Vec<NodeIndex> {
    let (i, j) = graph.edge_endpoints(entering_arc).unwrap();
    let mut path_from_i: Vec<NodeIndex> = vec![i];
    let mut path_from_j: Vec<NodeIndex> = vec![j];

    let mut current_node_i: NodeIndex = i;
    let mut current_node_j: NodeIndex = j;

    while current_node_j != current_node_i {
        if sptree.depth[current_node_i.index()] < sptree.depth[current_node_j.index()] {
            current_node_j = sptree.pred[current_node_j.index()].unwrap();
            path_from_j.push(current_node_j);
        } else {
            current_node_i = sptree.pred[current_node_i.index()].unwrap();
            path_from_i.push(current_node_i);
        }
    }

    path_from_i
        .iter()
        .rev()
        .skip(1)
        .for_each(|&x| path_from_j.push(x));
    path_from_j
}

//computing delta the amount of unit of flow we can augment through the cycle
//returning (the leaving edge
fn compute_flowchange<'a, NUM: CloneableNum>(
    graph: &'a mut DiGraph<u32, CustomEdgeIndices<NUM>>,
    cycle: &mut Vec<NodeIndex>,
    entering_arc: EdgeIndex,
) -> EdgeIndex
//Vec<(EdgeReference<'a, CustomEdgeIndices<NUM>>, NUM)>,
{
    let mut edge_in_cycle: Vec<EdgeIndex> = vec![entering_arc; cycle.len()];
    let mut v: Vec<NUM> = vec![one(); cycle.len()];
    let mut delta: Vec<NUM> = vec![zero(); cycle.len()];

    cycle.push(cycle[0]);

    if graph[entering_arc].flow != zero() {
        cycle.reverse();
    };

    cycle
        .iter()
        .tuple_windows::<(&NodeIndex, &NodeIndex)>()
        .enumerate()
        .for_each(|(index, (i, j))| {
            let edge = graph.find_edge(*i, *j);
            if edge.is_none() {
                v[index] = zero::<NUM>() - one();
                edge_in_cycle[index] = graph.find_edge(*j, *i).unwrap();
            } else {
                edge_in_cycle[index] = edge.unwrap();
            }
        });

    edge_in_cycle.iter().enumerate().for_each(|(index, &x)| {
        if v[index] == one() {
            delta[index] = graph[x].capacity - graph[x].flow;
        } else {
            delta[index] = graph[x].flow;
        }
    });
    let flowchange = delta
        .iter()
        .enumerate()
        .min_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .unwrap();
    let farthest_blocking_edge: EdgeIndex = edge_in_cycle[flowchange.0];
    if *flowchange.1 != zero::<NUM>() {
        edge_in_cycle.iter().enumerate().for_each(|(index, &x)| {
            graph[x].flow += v[index] * *flowchange.1;
            if graph[x].flow == zero() {
                graph[x].state = one();
            }
            if graph[x].flow == graph[x].capacity {
                graph[x].state = zero::<NUM>() - one();
            }
        });
    }
    farthest_blocking_edge
}

/* Update sptree structure according to entering arc and leaving arc,
* reorder predecessors to keep tree coherent tree structure from one basis
* to another.
*/
fn update_sptree<NUM: CloneableNum>(
    graph: &mut DiGraph<u32, CustomEdgeIndices<NUM>>,
    sptree: &mut SPTree,
    entering_arc: EdgeIndex,
    leaving_arc: EdgeIndex,
    position: Option<usize>,
) {
    //if entering_arc != leaving_arc {
    if entering_arc == leaving_arc {
        return;
    }
    let node_nb = graph.node_count();
    let (i, j) = graph.edge_endpoints(entering_arc).unwrap();
    let (k, l) = graph.edge_endpoints(leaving_arc).unwrap();
    let mut path_to_change: &Vec<NodeIndex>;
    let mut path_to_root: &Vec<NodeIndex>;
    if sptree.pred[k.index()] == Some(l) {
        sptree.pred[k.index()] = None;
    } else {
        sptree.pred[l.index()] = None;
    }
    let mut path_from_i: Vec<NodeIndex> = Vec::new();
    let mut path_from_j: Vec<NodeIndex> = Vec::new();

    let mut current_node: Option<NodeIndex> = Some(i);
    while !current_node.is_none() {
        path_from_i.push(current_node.unwrap());
        current_node = sptree.pred[current_node.unwrap().index()];
    }
    current_node = Some(j);
    while !current_node.is_none() {
        path_from_j.push(current_node.unwrap());
        current_node = sptree.pred[current_node.unwrap().index()];
    }
    if path_from_i[path_from_i.len() - 1] != NodeIndex::new(node_nb - 1) {
        path_to_change = &path_from_i;
        path_to_root = &path_from_j;
    } else {
        path_to_change = &path_from_j;
        path_to_root = &path_from_i;
    }

    // update thread_id
    let mut current_node = sptree.thread[path_to_change.last().unwrap().index()];
    let mut block_parcour = vec![*path_to_change.last().unwrap()];
    while sptree.depth[current_node.index()] > sptree.depth[path_to_change.last().unwrap().index()]
    {
        block_parcour.push(current_node);
        current_node = sptree.thread[current_node.index()];
    }

    let mut dirty_rev_thread: Vec<NodeIndex> = vec![];
    let nodeid_to_block = sptree.rev_thread[block_parcour[0].index()];
    sptree.thread[nodeid_to_block.index()] = sptree.thread[block_parcour.last().unwrap().index()];
    dirty_rev_thread.push(nodeid_to_block);

    path_to_change
        .iter()
        .take(path_to_change.len() - 1)
        .for_each(|&x| {
            let mut current = sptree.thread[x.index()];
            let mut old_last = x;
            while sptree.depth[current.index()] > sptree.depth[x.index()] {
                old_last = current;
                current = sptree.thread[current.index()];
            }
            sptree.thread[sptree.rev_thread[x.index()].index()] = current;
            dirty_rev_thread.push(sptree.rev_thread[x.index()]);
            sptree.thread[old_last.index()] = sptree.pred[x.index()].unwrap();
            dirty_rev_thread.push(old_last);
        });

    let mut current = sptree.thread[path_to_change.last().unwrap().index()];
    let mut old = *path_to_change.last().unwrap();
    while sptree.depth[current.index()] > sptree.depth[path_to_change.last().unwrap().index()] {
        old = current;
        current = sptree.thread[current.index()];
    }
    sptree.thread[old.index()] = sptree.thread[path_to_root[0].index()];
    dirty_rev_thread.push(old);
    sptree.thread[path_to_root[0].index()] = path_to_change[0];
    dirty_rev_thread.push(path_to_root[0]);

    dirty_rev_thread
        .into_iter()
        .for_each(|new_rev| sptree.rev_thread[sptree.thread[new_rev.index()].index()] = new_rev);

    //Predecessors update

    if path_from_i[path_from_i.len() - 1] != NodeIndex::new(node_nb - 1) {
        sptree.pred[i.index()] = Some(j);
        path_from_i
            .iter()
            .enumerate()
            .skip(1)
            .for_each(|(index, &x)| sptree.pred[x.index()] = Some(path_from_i[index - 1]));
        path_to_change = &path_from_i;
        path_to_root = &path_from_j;
    } else {
        sptree.pred[j.index()] = Some(i);
        path_from_j
            .iter()
            .enumerate()
            .skip(1)
            .for_each(|(index, &x)| sptree.pred[x.index()] = Some(path_from_j[index - 1]));
        path_to_root = &path_from_i;
        path_to_change = &path_from_j;
    }

    //update depth
    sptree.depth[path_to_change[0].index()] = sptree.depth[path_to_root[0].index()] + 1;
    path_to_change.iter().skip(1).for_each(|x| {
        sptree.depth[x.index()] = sptree.depth[sptree.pred[x.index()].unwrap().index()] + 1
    });
    block_parcour.iter().for_each(|&x| {
        sptree.depth[x.index()] = sptree.depth[sptree.pred[x.index()].unwrap().index()] + 1
    });

    //can probably be faster with pointer manipulation
    if !position.is_none() {
        sptree.out_base[position.unwrap()] = leaving_arc;
    } else {
        let _index = sptree.out_base.iter().position(|&x| x == entering_arc);
        sptree.out_base[_index.unwrap()] = leaving_arc;
    }
}

///////////////////////
///////////////////////

fn get_reduced_cost_edgeindex<NUM: CloneableNum>(
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    edgeindex: EdgeIndex,
    potential: &Vec<NUM>,
) -> NUM {
    graph[edgeindex].cost - potential[graph.raw_edges()[edgeindex.index()].source().index()]
        + potential[graph.raw_edges()[edgeindex.index()].target().index()]
}

///////////////////////
///// Pivot rules /////
///////////////////////

//Best Eligible arc
fn _find_best_arc<NUM: CloneableNum>(
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    //    sptree: &SPTree,
    out_base: &Vec<EdgeIndex>,
    potential: &Vec<NUM>,
) -> (Option<usize>, Option<EdgeIndex>) {
    let mut min = zero();
    let mut entering_arc = None;
    let mut index = None;
    for i in 0..out_base.len() {
        let arc = out_base[i];
        let rc = graph[arc].state * get_reduced_cost_edgeindex(graph, arc, potential);
        if rc < min {
            min = rc;
            entering_arc = Some(arc);
            index = Some(i);
        }
    }
    (index, entering_arc)
}

//First eligible
fn _find_first_arc<NUM: CloneableNum>(
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    out_base: &Vec<EdgeIndex>,
    potential: &Vec<NUM>,
    mut index: Option<usize>,
) -> (Option<usize>, Option<EdgeIndex>) {
    if index.is_none() {
        index = Some(0);
    }
    for i in index.unwrap()..out_base.len() {
        let arc = out_base[i];
        let rc = graph[arc].state * get_reduced_cost_edgeindex(graph, arc, potential);
        if rc < zero() {
            return (Some(i), Some(arc));
        }
    }
    for i in 0..index.unwrap() {
        let arc = out_base[i];
        let rc = graph[arc].state * get_reduced_cost_edgeindex(graph, arc, potential);
        if rc < zero() {
            return (Some(i), Some(arc));
        }
    }
    (None, None)
}

//Block search
fn _find_block_search<NUM: CloneableNum>(
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    out_base: &Vec<EdgeIndex>,
    potential: &Vec<NUM>,
    mut index: Option<usize>,
    block_size: usize,
) -> (Option<usize>, Option<EdgeIndex>) {
    let mut block_number = 0;
    let mut arc_index = None;
    if index.is_none() {
        index = Some(0);
    }
    while block_size * block_number <= out_base.len() {
        let (index_, rc_entering_arc) = out_base
            .iter()
            .enumerate()
            .cycle()
            .skip(index.unwrap())
            .take(block_size)
            .map(|(index, arc)| {
                (
                    index,
                    graph[*arc].state * get_reduced_cost_edgeindex(graph, *arc, potential),
                )
            })
            .min_by(|&x, &y| x.1.partial_cmp(&y.1).unwrap())
            .unwrap();
        if rc_entering_arc >= zero::<NUM>() {
            index = Some(index.unwrap() + block_size);
            block_number += 1;
        } else {
            arc_index = Some(index_);
            break;
        }
    }

    if arc_index.is_none() {
        (None, None)
    } else {
        let arc = if arc_index.unwrap() < out_base.len() {
            out_base[arc_index.unwrap()]
        } else {
            out_base[arc_index.unwrap() - out_base.len()]
        };
        (arc_index, Some(arc))
    }
}

fn _par_find_block_search<NUM: CloneableNum>(
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    out_base: &Vec<EdgeIndex>,
    potential: &Vec<NUM>,
    mut index: Option<usize>,
    block_size: usize,
) -> (Option<usize>, Option<EdgeIndex>) {
    let mut block_number = 0;
    let mut arc_index = None;
    if index.is_none() {
        index = Some(0);
    }
    while block_size * block_number <= out_base.len() {
        let (index_, rc_entering_arc) = out_base
            .iter()
            .enumerate()
            .skip(index.unwrap())
            .chain(out_base.iter().enumerate().take(block_size))
            .take(block_size)
            .map(|(index, arc)| {
                (
                    index,
                    graph[*arc].state * get_reduced_cost_edgeindex(graph, *arc, potential),
                )
            })
            .min_by(|&x, &y| x.1.partial_cmp(&y.1).unwrap())
            .unwrap();
        if rc_entering_arc >= zero::<NUM>() {
            index = Some(index.unwrap() + block_size);
            block_number += 1;
            if index >= Some(out_base.len()) {
                index = Some(0);
            }
        } else {
            arc_index = Some(index_);
            break;
        }
    }
    if arc_index.is_none() {
        (None, None)
    } else {
        (arc_index, Some(out_base[arc_index.unwrap()]))
    }
}

//main algorithm function
pub fn min_cost<NUM: CloneableNum>(
    mut graph: DiGraph<u32, CustomEdgeIndices<NUM>>,
    demand: NUM,
    //pivotrules: &str,
) -> DiGraph<u32, CustomEdgeIndices<NUM>> {
    let mut tlu_solution = initialization::<NUM>(&mut graph, demand);
    let mut potentials = compute_node_potentials(&mut graph);

    let block_size = (graph.edge_count() / 15) as usize;
    let mut _index: Option<usize> = Some(0);
    let mut entering_arc: Option<EdgeIndex>;

    //ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();
    (_index, entering_arc) =
        _find_first_arc(&mut graph, &tlu_solution.out_base, &mut potentials, _index);

    //entering_arc = _find_best_arc(&mut graph, &tlu_solution.out_base, &mut potentials);
    let mut _iteration = 0;
    while !entering_arc.is_none() {
        //println!("\nITERATION {:?}", _iteration);
        let mut cycle = find_cycle_with_arc(&graph, &mut tlu_solution, entering_arc.unwrap());

        let leaving_arc = compute_flowchange(&mut graph, &mut cycle, entering_arc.unwrap());
        /*println!("enter {:?}", graph.edge_endpoints(entering_arc.unwrap()));
        println!("leave {:?}", graph.edge_endpoints(leaving_arc));*/
        update_sptree(
            &mut graph,
            &mut tlu_solution,
            entering_arc.unwrap(),
            leaving_arc,
            _index,
        );

        potentials = update_node_potentials(
            &graph,
            potentials,
            &mut tlu_solution,
            entering_arc.unwrap(),
            leaving_arc,
        );

        /*(_index, entering_arc) = _find_block_search(
            &mut graph,
            &tlu_solution.out_base,
            &mut potentials,
            _index,
            block_size,
        );*/

        (_index, entering_arc) =
            _find_best_arc(&mut graph, &tlu_solution.out_base, &mut potentials);

        /*(_index, entering_arc) =
            _find_first_arc(&mut graph, &tlu_solution.out_base, &mut potentials, _index);
        */
        _iteration += 1;
    }
    println!("iterations : {:?}", _iteration);
    let mut cost: NUM = zero();
    graph
        .edge_references()
        .for_each(|x| cost += x.weight().flow * x.weight().cost);
    println!("total cost = {:?}", cost);
    graph
}
