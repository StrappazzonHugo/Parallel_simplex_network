use debug_print::debug_print;
use debug_print::debug_println;
use itertools::Itertools;
use num_traits::identities::zero;
use num_traits::Num;
use petgraph::algo::bellman_ford;
use petgraph::dot::Dot;
use petgraph::graph::EdgeIndex;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;

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
fn __initialization<'a, NUM: CloneableNum>(
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

    let mut big_value: NUM = zero();
    for _ in 0..100 {
        big_value += demand;
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

    graph.edge_references().for_each(|x| {
        if !tree_arcs.contains(&x.id()) {
            lower_arcs.push(x.id());
        }
    });

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
            (
                graph.edge_endpoints(x).unwrap().0.index() as u32,
                graph.edge_endpoints(x).unwrap().1.index() as u32,
                1f32,
            )
        }) //TODO
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
    reduced_cost: &mut Vec<NUM>,
) -> Vec<NUM> {
    let mut edges: Vec<(u32, u32, f32)> = sptree
        .t
        .iter()
        .filter(|&&x| x != entering_arc)
        .map(|&x| {
            (
                graph.edge_endpoints(x).unwrap().0.index() as u32,
                graph.edge_endpoints(x).unwrap().1.index() as u32,
                0f32,
            )
        })
        .collect();
    edges.push((
        graph.edge_endpoints(entering_arc).unwrap().0.index() as u32,
        graph.edge_endpoints(entering_arc).unwrap().1.index() as u32,
        1f32,
    )); // <-- edge separating T1 and T2
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
    if potentials_to_update.contains(&graph.edge_endpoints(entering_arc).unwrap().1.index()) {
        change -= reduced_cost[entering_arc.index()];
    } else {
        change += reduced_cost[entering_arc.index()];
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

fn _update_node_potentials<'a, NUM: CloneableNum>(
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    potential: Vec<NUM>,
    sptree: &mut SPTree,
    entering_arc: EdgeIndex,
    reduced_cost: &mut Vec<NUM>,
) -> Vec<NUM> {
    let (k, l) = graph.edge_endpoints(entering_arc).unwrap();
    let mut change: NUM = zero();
    let start: NodeIndex;
    if sptree.pred[k.index()] == Some(l) {
        change += reduced_cost[entering_arc.index()];
        start = k;
    } else {
        change -= reduced_cost[entering_arc.index()];
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
}

//Compute reduced cost such that : for any edge (i, j) in T : C^pi_ij = C_ij - pi(i) + pi(j) = 0
//                                 for any edge (i, j) notin T : C^pi_ij = C_ij - pi(i) + pi(j)
fn compute_reduced_cost<'a, NUM: CloneableNum>(
    pi: &mut Vec<NUM>,
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    sptree: &mut SPTree,
) -> Vec<NUM> {
    let mut reduced_cost: Vec<NUM> = vec![zero(); graph.edge_count()];
    sptree.l.iter().for_each(|&x| {
        reduced_cost[x.index()] = graph.edge_weight(x).unwrap().cost
            - pi[graph.edge_endpoints(x).unwrap().0.index()]
            + pi[graph.edge_endpoints(x).unwrap().1.index()];
    });
    sptree.u.iter().for_each(|&x| {
        reduced_cost[x.index()] = graph.edge_weight(x).unwrap().cost
            - pi[graph.edge_endpoints(x).unwrap().0.index()]
            + pi[graph.edge_endpoints(x).unwrap().1.index()];
    });
    reduced_cost
}

fn find_entering_arc<'a, NUM: CloneableNum>(
    sptree: &mut SPTree,
    reduced_cost: &mut Vec<NUM>,
) -> Option<EdgeIndex> {
    let mut min_l: Option<EdgeIndex> = sptree.l.clone().into_iter().min_by(|x, y| {
        reduced_cost[x.index()]
            .partial_cmp(&reduced_cost[y.index()])
            .unwrap()
    });
    let mut max_u: Option<EdgeIndex> = sptree.u.clone().into_iter().max_by(|x, y| {
        reduced_cost[x.index()]
            .partial_cmp(&reduced_cost[y.index()])
            .unwrap()
    });

    if !min_l.is_none() && reduced_cost[min_l.unwrap().index()] >= zero() {
        min_l = None;
    }
    if !max_u.is_none() && reduced_cost[max_u.unwrap().index()] <= zero() {
        max_u = None;
    }

    match (min_l, max_u) {
        (None, None) => None,
        (_, None) => min_l,
        (None, _) => max_u,
        (_, _) => {
            if (zero::<NUM>() - reduced_cost[min_l.unwrap().index()])
                >= reduced_cost[max_u.unwrap().index()]
            {
                min_l
            } else {
                max_u
            }
        }
    }
}

//Using Bellmanford algorithm to find negative weight cycle
//we build a graph with a cycle and forcing every arc weight to 0 except for one arc that we know
//in the cycle. BF then find negative weight cycle
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
    path_from_j
}

//checking if the specified edge is in forward direction according to a cycle
//needed in function compute_flowchange(...)
fn is_forward(edgeref: (NodeIndex, NodeIndex), cycle: &mut Vec<NodeIndex>) -> bool {
    let mut altered_cycle = cycle.clone();
    altered_cycle.push(*cycle.first().unwrap());
    let test: Vec<(NodeIndex, NodeIndex)> = altered_cycle
        .into_iter()
        .tuple_windows::<(NodeIndex, NodeIndex)>()
        .collect();
    test.contains(&edgeref)
}

//decompose cycle in tuple_altered_cycle variable ordered in distance to the entering arc
fn distances_in_cycle(cycle: &mut Vec<NodeIndex>) -> Vec<(NodeIndex, NodeIndex)> {
    let mut altered_cycle = cycle.clone();
    altered_cycle.push(*cycle.first().unwrap());
    let tuple_altered_cycle: Vec<(NodeIndex, NodeIndex)> = altered_cycle
        .into_iter()
        .tuple_windows::<(NodeIndex, NodeIndex)>()
        .collect();
    return tuple_altered_cycle;
}

//computing delta the amount of unit of flow we can augment through the cycle
//returning (the leaving edge:(u:u32, v:u32) , flow_change vector : Vec<(edge uv), delta(uv)>)
fn compute_flowchange<'a, NUM: CloneableNum>(
    graph: &'a mut DiGraph<u32, CustomEdgeIndices<NUM>>,
    mut cycle: &mut Vec<NodeIndex>,
) -> EdgeIndex
//Vec<(EdgeReference<'a, CustomEdgeIndices<NUM>>, NUM)>,
{
    let mut altered_cycle = cycle.clone();
    altered_cycle.push(*cycle.first().unwrap());
    let tuple_altered_cycle: Vec<(NodeIndex, NodeIndex)> = altered_cycle
        .into_iter()
        .tuple_windows::<(NodeIndex, NodeIndex)>()
        .collect();
    let distances = distances_in_cycle(&mut cycle);

    let mut edge_in_cycle: Vec<EdgeIndex> = Vec::new();

    tuple_altered_cycle.iter().for_each(|(u, v)| {
        let edge = graph.find_edge(*u, *v);
        if edge != None {
            edge_in_cycle.push(edge.unwrap());
        } else {
            edge_in_cycle.push(graph.find_edge(*v, *u).unwrap());
        }
    });

    edge_in_cycle = edge_in_cycle
        .into_iter()
        .sorted_by_key(|e| {
            distances
                .iter()
                .enumerate()
                .find(|x| {
                    (x.1 .0, x.1 .1) == graph.edge_endpoints(*e).unwrap()
                        || (x.1 .1, x.1 .0) == graph.edge_endpoints(*e).unwrap()
                })
                .unwrap()
                .0
        })
        .rev()
        .collect();

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
fn _update_sptree<NUM: CloneableNum>(
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

fn debug_print_tlu<NUM: CloneableNum>(graph: DiGraph<u32, CustomEdgeIndices<NUM>>, tlu: SPTree) {
    debug_print!("\nT = ");
    tlu.t.iter().for_each(|x| {
        let endspoints = graph.edge_endpoints(*x).unwrap();
        debug_print!("{:?}, ", (endspoints.0.index(), endspoints.1.index()))
    });
    debug_print!("\nL = ");
    tlu.l.iter().for_each(|x| {
        let endspoints = graph.edge_endpoints(*x).unwrap();
        debug_print!("{:?}, ", (endspoints.0.index(), endspoints.1.index()))
    });
    debug_print!("\nU = ");
    tlu.u.iter().for_each(|x| {
        let endspoints = graph.edge_endpoints(*x).unwrap();
        debug_print!("{:?}, ", (endspoints.0.index(), endspoints.1.index()))
    });
    debug_print!("\npred = {:?}", tlu.pred);
    debug_print!("\n");
}

fn debug_print_reduced_cost<NUM: CloneableNum>(
    graph: DiGraph<u32, CustomEdgeIndices<NUM>>,
    reduced_cost: Vec<NUM>,
) {
    debug_println!("reduced_ cost :");
    reduced_cost.iter().enumerate().for_each(|(edge, rc)| {
        let endspoints = graph.edge_endpoints(EdgeIndex::new(edge)).unwrap();
        debug_println!(
            "{:?} : {:?}",
            (endspoints.0.index(), endspoints.1.index()),
            rc
        )
    });
}

//main algorithm function
pub fn min_cost<NUM: CloneableNum>(
    mut graph: DiGraph<u32, CustomEdgeIndices<NUM>>,
    demand: NUM,
) -> DiGraph<u32, CustomEdgeIndices<NUM>> {
    let mut tlu_solution = __initialization::<NUM>(&mut graph, demand);

    debug_print_tlu(graph.clone(), tlu_solution.clone());

    let mut potentials = compute_node_potentials(&mut graph, &mut tlu_solution);
    debug_println!("potentials = {:?}", potentials);

    let mut reduced_cost = compute_reduced_cost(&mut potentials, &graph, &mut tlu_solution);
    debug_print_reduced_cost(graph.clone(), reduced_cost.clone());

    let mut entering_arc = find_entering_arc(&mut tlu_solution, &mut reduced_cost);
    debug_println!("entering arc : {:?}", {
        if entering_arc.is_none() {
            (0, 0)
        } else {
            let endpoints = graph.edge_endpoints(entering_arc.unwrap()).unwrap();
            (endpoints.0.index(), endpoints.1.index())
        }
    });

    let mut iteration = 0;

    while entering_arc != None {
        debug_println!("ITERATION {:?}", iteration);
        let mut cycle = find_cycle_with_arc(&graph, &mut tlu_solution, entering_arc.unwrap());
        let leaving_arc = compute_flowchange(&mut graph, &mut cycle);
        debug_println!("leaving arc : {:?}", {
            {
                let endpoints = graph.edge_endpoints(leaving_arc).unwrap();
                (endpoints.0.index(), endpoints.1.index())
            }
        });
        _update_sptree(
            &mut graph,
            &mut tlu_solution,
            entering_arc.unwrap(),
            leaving_arc,
        );
        debug_print_tlu(graph.clone(), tlu_solution.clone());

        potentials = update_node_potentials(
            &graph,
            potentials,
            &mut tlu_solution,
            entering_arc.unwrap(),
            &mut reduced_cost,
        );
        debug_println!("potentials = {:?}", potentials);

        reduced_cost = compute_reduced_cost(&mut potentials, &graph, &mut tlu_solution);

        debug_print_reduced_cost(graph.clone(), reduced_cost.clone());

        entering_arc = find_entering_arc(&mut tlu_solution, &mut reduced_cost);

        debug_println!("entering arc : {:?}", {
            if entering_arc.is_none() {
                (0, 0)
            } else {
                let endpoints = graph.edge_endpoints(entering_arc.unwrap()).unwrap();
                (endpoints.0.index(), endpoints.1.index())
            }
        });

        iteration += 1;
    }
    graph.remove_node(NodeIndex::new(graph.node_count() - 1));
    debug_println!("_test = {:?}", Dot::new(&graph));
    let mut cost: NUM = zero();
    graph
        .edge_references()
        .for_each(|x| cost += x.weight().flow * x.weight().cost);
    println!("total cost = {:?}", cost);
    graph
}
