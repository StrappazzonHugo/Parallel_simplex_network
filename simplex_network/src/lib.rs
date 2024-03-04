use debug_print::debug_println;
use itertools::Itertools;
use num_traits::identities::zero;
use num_traits::Num;
use petgraph::algo::bellman_ford;
use petgraph::algo::find_negative_cycle;
use petgraph::dot::Dot;
use petgraph::graph::EdgeIndex;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct SPTree {
    t: Vec<EdgeIndex>,
    l: Vec<EdgeIndex>,
    u: Vec<EdgeIndex>,
}

//TODO functions getcost(graph, i, j)
//               getcapacity(graph, i, j)
//               getflow(graph, i, j)

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
    //debug_println!("initial_number_of_node = {:?}", initial_number_of_node);
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

    SPTree {
        t: tree_arcs,
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
fn par_update_node_potentials<'a, NUM: CloneableNum>(
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

//Compute reduced cost such that : for any edge (i, j) in T : C^pi_ij = C_ij - pi(i) + pi(j) = 0
//                                 for any edge (i, j) notin T : C^pi_ij = C_ij - pi(i) + pi(j)
fn compute_reduced_cost<'a, NUM: CloneableNum>(
    pi: &mut Vec<NUM>,
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    sptree: &mut SPTree,
) -> Vec<NUM> {
    let mut reduced_cost: Vec<NUM> = vec![zero(); graph.edge_count()];
    sptree.l.iter().chain(sptree.u.iter()).for_each(|&x| {
        reduced_cost[x.index()] = graph.edge_weight(x).unwrap().cost
            - pi[graph.edge_endpoints(x).unwrap().0.index()]
            + pi[graph.edge_endpoints(x).unwrap().1.index()];
    });
    reduced_cost
}

/*
fn par_find_entering_arc<'a, NUM: CloneableNum>(
    sptree: &mut SPTree,
    reduced_cost: &mut HashMap<EdgeIndex, NUM>,
) -> Option<EdgeIndex> {
    let min_l = sptree.l.iter().min_by(|a, b| {
        reduced_cost
            .get(&a)
            .unwrap()
            .partial_cmp(reduced_cost.get(&b).unwrap())
            .unwrap()
    });
    let max_u = sptree.u.iter().max_by(|a, b| {
        reduced_cost
            .get(&a)
            .unwrap()
            .partial_cmp(reduced_cost.get(&b).unwrap())
            .unwrap()
    });
    match (min_l, max_u) {
        (Some(min_l), Some(max_u)) => {
            if reduced_cost.get(&min_l) >= Some(&(zero()))
                && reduced_cost.get(&max_u) <= Some(&(zero()))
            {
                None
            } else if reduced_cost.get(&min_l) >= Some(&(zero())) {
                Some(*max_u)
            } else if reduced_cost.get(&max_u) <= Some(&(zero())) {
                Some(*min_l)
            } else if (zero::<NUM>() - *reduced_cost.get(&min_l).unwrap())
                >= *reduced_cost.get(&max_u).unwrap()
            {
                Some(*min_l)
            } else {
                Some(*max_u)
            }
        }
        (Some(min_l), None) => {
            if reduced_cost.get(&min_l) < Some(&(zero())) {
                Some(*min_l)
            } else {
                None
            }
        }
        (None, Some(max_u)) => {
            if reduced_cost.get(&max_u) > Some(&(zero())) {
                Some(*max_u)
            } else {
                None
            }
        }
        (None, None) => None,
    }
}*/

fn par_find_entering_arc<'a, NUM: CloneableNum>(
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
        (_, _) => if (zero::<NUM>() - reduced_cost[min_l.unwrap().index()]) >= reduced_cost[max_u.unwrap().index()]
            {
                min_l
            } else {
                max_u
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
    let is_in_u = sptree.u.iter().contains(&entering_arc);
    let mut edges: Vec<(u32, u32, f32)> = sptree
        .t
        .iter()
        .map(|&x| {
            (
                graph.edge_endpoints(x).unwrap().0.index() as u32,
                graph.edge_endpoints(x).unwrap().1.index() as u32,
                0f32,
            )
        })
        .collect();
    let mut edges1: Vec<(u32, u32, f32)> = sptree
        .t
        .iter()
        .map(|&x| {
            (
                graph.edge_endpoints(x).unwrap().1.index() as u32,
                graph.edge_endpoints(x).unwrap().0.index() as u32,
                0f32,
            )
        })
        .collect();
    edges.append(&mut edges1);
    if is_in_u {
        edges.push((
            graph.edge_endpoints(entering_arc).unwrap().1.index() as u32,
            graph.edge_endpoints(entering_arc).unwrap().0.index() as u32,
            -1f32,
        ));
    } else {
        edges.push((
            graph.edge_endpoints(entering_arc).unwrap().0.index() as u32,
            graph.edge_endpoints(entering_arc).unwrap().1.index() as u32,
            -1f32,
        ));
    }
    let g = Graph::<(), f32, Directed>::from_edges(edges);
    let mut cycle = find_negative_cycle(&g, NodeIndex::new(graph.node_count() - 1)).unwrap();

    while (cycle[0], cycle[1]) != graph.edge_endpoints(entering_arc).unwrap()
        && (cycle[1], cycle[0]) != graph.edge_endpoints(entering_arc).unwrap()
    {
        cycle.rotate_left(1);
    }
    cycle
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
    let farthest_blocking_edge: petgraph::graph::EdgeIndex = edge_in_cycle[flowchange.0];
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

//updating sptree structure according to the leaving arc
//removing it from T and putting in L or U depending on the updated flow on the arc
fn par_update_sptree_with_leaving<'a, NUM: CloneableNum>(
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    sptree: &mut SPTree,
    leaving_arc: petgraph::graph::EdgeIndex,
) {
    sptree.t = sptree
        .t
        .clone()
        .into_iter()
        .filter(|&x| x != leaving_arc)
        .collect();
    let at_min_capacity: bool = graph.edge_weight(leaving_arc).unwrap().flow == zero();

    if at_min_capacity {
        sptree.l.push(leaving_arc);
    } else {
        sptree.u.push(leaving_arc)
    }
}

//adding entering_arc to T arc set of sptree
fn par_update_sptree_with_entering(sptree: &mut SPTree, entering_arc: Option<EdgeIndex>) {
    sptree.t.push(entering_arc.unwrap());
    sptree.l = sptree
        .l
        .clone()
        .into_iter()
        .filter(|&x| x != entering_arc.unwrap())
        .collect();
    sptree.u = sptree
        .u
        .clone()
        .into_iter()
        .filter(|&x| x != entering_arc.unwrap())
        .collect();
}

fn debug_print_tlu<NUM: CloneableNum>(graph: DiGraph<u32, CustomEdgeIndices<NUM>>, tlu: SPTree) {
    print!("\nT = ");
    tlu.t.iter().for_each(|x| {
        let endspoints = graph.edge_endpoints(*x).unwrap();
        print!("{:?}, ", (endspoints.0.index(), endspoints.1.index()))
    });
    print!("\nL = ");
    tlu.l.iter().for_each(|x| {
        let endspoints = graph.edge_endpoints(*x).unwrap();
        print!("{:?}, ", (endspoints.0.index(), endspoints.1.index()))
    });
    print!("\nU = ");
    tlu.u.iter().for_each(|x| {
        let endspoints = graph.edge_endpoints(*x).unwrap();
        print!("{:?}, ", (endspoints.0.index(), endspoints.1.index()))
    });
    print!("\n");
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
    graph: DiGraph<u32, CustomEdgeIndices<NUM>>,
    demand: NUM,
) -> DiGraph<u32, CustomEdgeIndices<NUM>> {
    let mut min_cost_flow_graph = graph;
    let mut tlu_solution = __initialization::<NUM>(&mut min_cost_flow_graph, demand);

    //debug_print_tlu(min_cost_flow_graph.clone(), tlu_solution.clone());

    let mut potentials = compute_node_potentials(&mut min_cost_flow_graph, &mut tlu_solution);
    debug_println!("potentials = {:?}", potentials);

    let mut reduced_cost =
        compute_reduced_cost(&mut potentials, &min_cost_flow_graph, &mut tlu_solution);
    debug_print_reduced_cost(min_cost_flow_graph.clone(), reduced_cost.clone());

    let mut entering_arc = par_find_entering_arc(&mut tlu_solution, &mut reduced_cost);
    debug_println!("entering arc : {:?}", {
        if entering_arc.is_none() {
            (0, 0)
        } else {
            let endpoints = min_cost_flow_graph
                .edge_endpoints(entering_arc.unwrap())
                .unwrap();
            (endpoints.0.index(), endpoints.1.index())
        }
    });

    let mut iteration = 0;

    while entering_arc != None {
        println!("ITERATION {:?}", iteration);
        let mut cycle = find_cycle_with_arc(
            &min_cost_flow_graph,
            &mut tlu_solution,
            entering_arc.unwrap(),
        );
        par_update_sptree_with_entering(&mut tlu_solution, entering_arc);

        let leaving_arc = compute_flowchange(&mut min_cost_flow_graph, &mut cycle);

        par_update_sptree_with_leaving(&min_cost_flow_graph, &mut tlu_solution, leaving_arc);
        debug_println!("leaving arc : {:?}", {
            {
                let endpoints = min_cost_flow_graph.edge_endpoints(leaving_arc).unwrap();
                (endpoints.0.index(), endpoints.1.index())
            }
        });

        potentials = par_update_node_potentials(
            &min_cost_flow_graph,
            potentials,
            &mut tlu_solution,
            entering_arc.unwrap(),
            &mut reduced_cost,
        );
        debug_println!("potentials = {:?}", potentials);

        reduced_cost =
            compute_reduced_cost(&mut potentials, &min_cost_flow_graph, &mut tlu_solution);

        debug_print_reduced_cost(min_cost_flow_graph.clone(), reduced_cost.clone());

        entering_arc = par_find_entering_arc(&mut tlu_solution, &mut reduced_cost);

        debug_println!("entering arc : {:?}", {
            if entering_arc.is_none() {
                (0, 0)
            } else {
                let endpoints = min_cost_flow_graph
                    .edge_endpoints(entering_arc.unwrap())
                    .unwrap();
                (endpoints.0.index(), endpoints.1.index())
            }
        });

        iteration += 1;
        let mut cost: NUM = zero();
        min_cost_flow_graph
            .edge_references()
            .for_each(|x| cost += x.weight().flow * x.weight().cost);
        debug_println!("total cost = {:?}", cost);
    }
    debug_println!("_test = {:?}", Dot::new(&min_cost_flow_graph));
    min_cost_flow_graph
}
