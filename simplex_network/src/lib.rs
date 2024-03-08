use itertools::Itertools;
use num_traits::identities::zero;
use num_traits::Float;
use num_traits::Num;
use petgraph::algo::bellman_ford;
use petgraph::data::DataMap;
use petgraph::dot::Dot;
use petgraph::graph::EdgeIndex;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::visit::EdgeCount;
use rayon::prelude::*;

#[derive(Debug, Clone)]
struct SPTree {
    t: Vec<EdgeIndex>,
    pred: Vec<Option<NodeIndex>>,
    l: Vec<EdgeIndex>,
    u: Vec<EdgeIndex>,
}

#[derive(Clone, Debug, Copy, PartialEq, PartialOrd)]
pub struct CustomEdgeIndices<NUM: CloneableNum> {
    pub cost: NUM,
    pub capacity: NUM,
    pub flow: NUM,
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

    let mut tree_arcs: Vec<EdgeIndex> = Vec::new();
    let mut lower_arcs: Vec<EdgeIndex> = Vec::new();
    let upper_arcs: Vec<EdgeIndex> = Vec::new();

    graph.edge_references().for_each(|x| {
        lower_arcs.push(x.id());
    });

    let mut big_value: NUM = zero();
    for _ in 0..100 {
        big_value = demand;
    }

    let artificial_root = graph.add_node(graph.node_count() as u32);
    let mut edge: EdgeIndex;
    for node in graph.node_indices() {
        if node == artificial_root {
            continue;
        }
        if node == sink_id {
            edge = graph.add_edge(
                artificial_root,
                node,
                CustomEdgeIndices {
                    cost: big_value,
                    capacity: big_value,
                    flow: demand,
                },
            );
        } else if node == NodeIndex::new(0) {
            edge = graph.add_edge(
                node,
                artificial_root,
                CustomEdgeIndices {
                    cost: big_value,
                    capacity: big_value,
                    flow: demand,
                },
            );
        } else {
            edge = graph.add_edge(
                node,
                artificial_root,
                CustomEdgeIndices {
                    cost: big_value,
                    capacity: big_value,
                    flow: zero(),
                },
            );
        }
        tree_arcs.push(edge);
    }

    let mut predecessors: Vec<Option<NodeIndex>> =
        vec![Some(artificial_root); graph.node_count() - 1];
    predecessors.push(None);

    SPTree {
        t: tree_arcs,
        pred: predecessors,
        l: lower_arcs,
        u: upper_arcs,
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
    sptree: &mut SPTree,
) -> Vec<NUM> {
    let mut pi: Vec<NUM> = vec![zero(); graph.node_count()];
    let edges: Vec<(u32, u32, f32)> = sptree
        .t
        .iter()
        .map(|&x| {
            let (u, v) = graph.edge_endpoints(x).unwrap();
            (u.index() as u32, v.index() as u32, 1f32)
        })
        .collect();
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
            pi[pred.unwrap().index()] - graph.edge_weight(edge.unwrap()).unwrap().cost
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

//Updating potential inspired by : NETWORK FLOWS Theory, Algorithms, and Applications (p.19)
//modified such that the difference from T1 and T2 is done with a BellmanFord algorithm
//detecting the single edge cost put to 1 that divide T1 and T2 in T
fn update_node_potentials<'a, NUM: CloneableNum>(
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    potential: Vec<NUM>,
    sptree: &mut SPTree,
    entering_arc: EdgeIndex,
) -> Vec<NUM> {
    let mut edges: Vec<(u32, u32, f32)> = sptree
        .t
        .iter()
        .filter(|&&x| x != entering_arc)
        .map(|&x| {
            let (i, j) = graph.edge_endpoints(x).unwrap();
            (i.index() as u32, j.index() as u32, 0f32)
        })
        .collect();
    let (u, v) = graph.edge_endpoints(entering_arc).unwrap();
    edges.push((u.index() as u32, v.index() as u32, 1f32)); // <-- edge separating T1 and T2
    let g = Graph::<(), f32, Undirected>::from_edges(edges);
    let path_cost = bellman_ford(&g, NodeIndex::new(graph.node_count() - 1))
        .unwrap()
        .distances;
    let potentials_to_update: Vec<usize> = path_cost
        .iter()
        .enumerate()
        .filter(|&(_, cost)| cost > &0.)
        .map(|x| x.0)
        .collect();

    let mut change: NUM = zero();
    let rc = get_reduced_cost_edgeindex(graph, entering_arc, &potential);
    if potentials_to_update.contains(&v.index()) {
        change -= rc;
    } else {
        change += rc;
    }
    potential
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if potentials_to_update.contains(&index) {
                value + change
            } else {
                value
            }
        })
        .collect()
}
/*
fn update_node_potentials<'a, NUM: CloneableNum>(
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    potential: Vec<NUM>,
    sptree: &mut SPTree,
    entering_arc: EdgeIndex,
) -> Vec<NUM> {
    let (k, l) = graph.edge_endpoints(entering_arc).unwrap();
    let mut change: NUM = zero();
    let rc = get_reduced_cost_edgeindex(graph, entering_arc, &potential);
    let start: NodeIndex;
    if sptree.pred[k.index()] == Some(l) {
        change += rc;
        start = k;
    } else {
        change -= rc;
        start = l;
    }
    let mut potentials_to_update: Vec<usize> = Vec::new();
    let mut new_vec: Vec<usize> = vec![start.index()];
    let mut prev_vec: Vec<usize>;
    while new_vec != [] {
        prev_vec = new_vec.clone();
        potentials_to_update.append(&mut new_vec);
        new_vec.clear();
        prev_vec.iter().for_each(|&x| {
            sptree
                .pred
                .iter()
                .enumerate()
                .filter(|(_, &y)| y == Some(NodeIndex::new(x)))
                .for_each(|(index, _)| new_vec.push(index));
        });
    }
    potential
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if potentials_to_update.contains(&index) {
                value + change
            } else {
                value
            }
        })
        .collect()
}*/

fn find_cycle_with_arc<'a, NUM: CloneableNum>(
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    sptree: &mut SPTree,
    entering_arc: EdgeIndex,
) -> Vec<NodeIndex> {
    let (i, j) = graph.edge_endpoints(entering_arc).unwrap();
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

    for (i, i_node) in path_from_i.iter().enumerate() {
        if path_from_j.contains(i_node) {
            let j = path_from_j
                .iter()
                .find_position(|&x| x == i_node)
                .unwrap()
                .0;
            path_from_j.truncate(j + 1);
            path_from_i.truncate(i);
            path_from_i.iter().rev().for_each(|&x| path_from_j.push(x));
            break;
        }
    }
    path_from_j.rotate_right(1);
    path_from_j
}

//checking if the specified edge is in forward direction according to a cycle
//needed in function compute_flowchange(...)
fn is_forward(edgeref: (NodeIndex, NodeIndex), cycle: &mut Vec<NodeIndex>) -> bool {
    let forward_arcs = distances_in_cycle(cycle);
    forward_arcs.contains(&edgeref)
}

//decompose cycle in tuple_altered_cycle variable ordered in distance to the entering arc
fn distances_in_cycle(cycle: &mut Vec<NodeIndex>) -> Vec<(NodeIndex, NodeIndex)> {
    let mut altered_cycle = cycle.clone();
    altered_cycle.push(*cycle.first().unwrap());
    let tuple_altered_cycle: Vec<(NodeIndex, NodeIndex)> = altered_cycle
        .into_iter()
        .tuple_windows::<(NodeIndex, NodeIndex)>()
        .collect();
    tuple_altered_cycle
}

//computing delta the amount of unit of flow we can augment through the cycle
//returning (the leaving edge
fn compute_flowchange<'a, NUM: CloneableNum>(
    graph: &'a mut DiGraph<u32, CustomEdgeIndices<NUM>>,
    mut cycle: &mut Vec<NodeIndex>,
) -> EdgeIndex
//Vec<(EdgeReference<'a, CustomEdgeIndices<NUM>>, NUM)>,
{
    let mut edge_in_cycle: Vec<EdgeIndex> = Vec::new();

    distances_in_cycle(cycle).iter().for_each(|(u, v)| {
        let edge = graph.find_edge(*u, *v);
        if edge != None {
            edge_in_cycle.push(edge.unwrap());
        } else {
            edge_in_cycle.push(graph.find_edge(*v, *u).unwrap());
        }
    });
    if graph.edge_weight(edge_in_cycle[0]).unwrap().flow != zero() {
        cycle.reverse();
        cycle.rotate_right(2);
    }
    let delta: Vec<NUM> = edge_in_cycle
        .iter()
        .map(|&x| {
            if is_forward(graph.edge_endpoints(x).unwrap(), &mut cycle) {
                graph.edge_weight(x).unwrap().capacity - graph.edge_weight(x).unwrap().flow
            } else {
                graph.edge_weight(x).unwrap().flow
            }
        })
        .collect();
    let flowchange = delta
        .iter()
        .enumerate()
        .rev()
        .min_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .unwrap();
    let farthest_blocking_edge: EdgeIndex = edge_in_cycle[flowchange.0];
    if *flowchange.1 != zero::<NUM>() {
        edge_in_cycle.iter().for_each(|&x| {
            let endpoints = graph.edge_endpoints(x).unwrap();
            graph.edge_weight_mut(x).unwrap().flow += if is_forward(endpoints, &mut cycle) {
                *flowchange.1
            } else {
                zero::<NUM>() - *flowchange.1
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
) {
    if entering_arc != leaving_arc {
        let (i, j) = graph.edge_endpoints(entering_arc).unwrap();
        let (k, l) = graph.edge_endpoints(leaving_arc).unwrap();
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
        if path_from_i[path_from_i.len() - 1] != NodeIndex::new(graph.node_count() - 1) {
            sptree.pred[i.index()] = Some(j);
            path_from_i
                .iter()
                .enumerate()
                .skip(1)
                .for_each(|(index, &x)| sptree.pred[x.index()] = Some(path_from_i[index - 1]));
        } else {
            sptree.pred[j.index()] = Some(i);
            path_from_j
                .iter()
                .enumerate()
                .skip(1)
                .for_each(|(index, &x)| sptree.pred[x.index()] = Some(path_from_j[index - 1]));
        }
    }
    sptree.t.push(entering_arc);
    sptree.l.retain(|&x| x != entering_arc);
    sptree.u.retain(|&x| x != entering_arc);

    sptree.t.retain(|&x| x != leaving_arc);
    let at_min_capacity: bool = graph.edge_weight(leaving_arc).unwrap().flow == zero();
    if at_min_capacity {
        sptree.l.push(leaving_arc);
    } else {
        sptree.u.push(leaving_arc)
    }
}

///////////////////////
///////////////////////

fn get_reduced_cost_edgeindex<NUM: CloneableNum>(
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    edgeindex: EdgeIndex,
    potential: &Vec<NUM>,
) -> NUM {
    let (u, v) = graph.edge_endpoints(edgeindex).unwrap();
    graph.edge_weight(edgeindex).unwrap().cost - potential[u.index()] + potential[v.index()]
}

fn get_block(
    edge_vec: &mut Vec<(usize, EdgeIndex)>,
    block_size: usize,
    index: usize,
) -> Vec<(usize, EdgeIndex)> {
    let block: Vec<(usize, EdgeIndex)> = edge_vec
        .iter()
        .cycle()
        .skip(index)
        .take(block_size)
        .map(|(index, x)| (*index, *x))
        .collect();
    block
}

fn find_best_eligible_arc<NUM: CloneableNum>(
    graph: &mut DiGraph<u32, CustomEdgeIndices<NUM>>,
    potential: &Vec<NUM>,
    block: &mut Vec<(usize, EdgeIndex)>,
) -> Option<usize> {
    let (index, rc_entering_arc) = block
        .iter()
        .map(|(index, arc)|
             (index, if graph.edge_weight(*arc).unwrap().flow == zero() {
                get_reduced_cost_edgeindex(graph, *arc, potential)
            } else {
                zero::<NUM>() - get_reduced_cost_edgeindex(graph, *arc, potential)
            }))
        .min_by(|&x, &y| 
            x.1.partial_cmp(&y.1)       
            .unwrap()
        )
        .unwrap();
    if rc_entering_arc >= zero::<NUM>() {
        None
    } else {
        Some(*index)
    }
}

///////////////////////
///// Pivot rules /////
///////////////////////

//Current strategy : Best Eligible arc
fn find_entering_arc<NUM: CloneableNum>(
    graph: &mut DiGraph<u32, CustomEdgeIndices<NUM>>,
    sptree: &mut SPTree,
    potential: &mut Vec<NUM>,
) -> Option<EdgeIndex> {
    let mut min_l: Option<EdgeIndex> = sptree.l.clone().into_par_iter().min_by(|&x, &y| {
        (get_reduced_cost_edgeindex(graph, x, potential))
            .partial_cmp(&(get_reduced_cost_edgeindex(graph, y, potential)))
            .unwrap()
    });
    let mut max_u: Option<EdgeIndex> = sptree.u.clone().into_par_iter().max_by(|&x, &y| {
        (get_reduced_cost_edgeindex(graph, x, potential))
            .partial_cmp(&(get_reduced_cost_edgeindex(graph, y, potential)))
            .unwrap()
    });
    let mut rcmin = None;
    let mut rcmax = None;
    if !min_l.is_none() {
        rcmin = Some(get_reduced_cost_edgeindex(graph, min_l.unwrap(), potential));
    };
    if !max_u.is_none() {
        rcmax = Some(get_reduced_cost_edgeindex(graph, max_u.unwrap(), potential));
    };
    if !min_l.is_none() && rcmin.unwrap() >= zero() {
        min_l = None;
    }
    if !max_u.is_none() && rcmax.unwrap() <= zero() {
        max_u = None;
    }

    match (min_l, max_u) {
        (None, None) => None,
        (_, None) => min_l,
        (None, _) => max_u,
        (_, _) => {
            if (zero::<NUM>() - rcmin.unwrap()) >= rcmax.unwrap() {
                min_l
            } else {
                max_u
            }
        }
    }
}

//Block search
fn find_block_search<NUM: CloneableNum>(
    graph: &mut DiGraph<u32, CustomEdgeIndices<NUM>>,
    sptree: &mut SPTree,
    potential: &mut Vec<NUM>,
    mut index: usize,
    block_size: usize,
) -> (Option<usize>, Option<EdgeIndex>) {
    let mut block_number = 0;
    let mut arc_index = None;
    let mut edge_vec: Vec<(usize, EdgeIndex)> = sptree
        .l
        .clone()
        .into_iter()
        .chain(sptree.u.clone().into_iter())
        .enumerate()
        .collect();
    while block_number * block_size <= graph.edge_count() {
        arc_index = find_best_eligible_arc(
            graph,
            &potential,
            &mut get_block(&mut edge_vec, block_size, index),
        );
        if arc_index.is_none() {
            index += block_size;
            block_number += 1;
        } else {
            break;
        }
    }
    (arc_index, Some(edge_vec[arc_index.unwrap()].1))
}

//main algorithm function
pub fn min_cost<NUM: CloneableNum>(
    mut graph: DiGraph<u32, CustomEdgeIndices<NUM>>,
    demand: NUM,
) -> DiGraph<u32, CustomEdgeIndices<NUM>> {
    let mut tlu_solution = initialization::<NUM>(&mut graph, demand);
    let mut potentials = compute_node_potentials(&mut graph, &mut tlu_solution);

    let block_size = (graph.edge_count() as f64).sqrt() as usize;
    /*let (mut index, mut entering_arc) = find_block_search(
        &mut graph,
        &mut tlu_solution,
        &mut potentials,
        0,
        block_size,
    );*/

    let mut entering_arc = find_entering_arc(&mut graph, &mut tlu_solution, &mut potentials);
    let mut _iteration = 0;
    while entering_arc != None {
        println!("ITERATION {:?}", _iteration);
        println!("enter {:?}", entering_arc);
        let mut cycle = find_cycle_with_arc(&graph, &mut tlu_solution, entering_arc.unwrap());
        let leaving_arc = compute_flowchange(&mut graph, &mut cycle);
        update_sptree(
            &mut graph,
            &mut tlu_solution,
            entering_arc.unwrap(),
            leaving_arc,
        );
        println!("leave {:?}", leaving_arc);
        potentials =
            update_node_potentials(&graph, potentials, &mut tlu_solution, entering_arc.unwrap());

       /* (index, entering_arc) = find_block_search(
            &mut graph,
            &mut tlu_solution,
            &mut potentials,
            index.unwrap(),
            block_size,
        );*/
        entering_arc = find_entering_arc(&mut graph, &mut tlu_solution, &mut potentials);
        _iteration += 1;
    }
    let mut cost: NUM = zero();
    graph
        .edge_references()
        .for_each(|x| cost += x.weight().flow * x.weight().cost);
    println!("total cost = {:?}", cost);
    graph
}
