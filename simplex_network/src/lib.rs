use debug_print::debug_println;
use itertools::Itertools;
use num_traits::bounds::Bounded;
use num_traits::identities::zero;
use num_traits::Num;
use petgraph::algo::bellman_ford;
use petgraph::algo::find_negative_cycle;
use petgraph::data::FromElements;
use petgraph::dot::Dot;
use petgraph::graph::node_index;
use petgraph::graph::EdgeReference;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct SPTree {
    t: Vec<(u32, u32)>,
    l: Vec<(u32, u32)>,
    u: Vec<(u32, u32)>,
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
    mut graph: DiGraph<u32, CustomEdgeIndices<NUM>>,
    demand: NUM,
) -> (SPTree, DiGraph<u32, CustomEdgeIndices<NUM>>) {
    let initial_number_of_node: u32 = (graph.node_count() - 1) as u32;
    debug_println!("initial_number_of_node = {:?}", initial_number_of_node);
    let sink_id = graph
        .node_indices()
        .find(|&x| (graph.node_weight(x).unwrap() == &initial_number_of_node))
        .unwrap();

    let mut tree_arcs: Vec<(u32, u32)> = Vec::new();
    let mut lower_arcs: Vec<(u32, u32)> = Vec::new();
    let upper_arcs: Vec<(u32, u32)> = Vec::new();

    let mut big_value: NUM = zero();
    for _ in 0..100 {
        big_value += demand;
    }
    let artificial_edge = graph.add_edge(
        NodeIndex::new(0),
        sink_id,
        CustomEdgeIndices {
            cost: Bounded::min_value(),
            capacity: big_value,
            flow: demand,
        },
    );
    let spanningtree = UnGraph::<_, _>::from_elements(petgraph::algo::min_spanning_tree(&graph));
    graph.edge_weight_mut(artificial_edge).unwrap().cost = big_value;

    debug_println!("{:?}", Dot::new(&spanningtree));
    spanningtree
        .edge_references()
        .for_each(|x| tree_arcs.push((x.source().index() as u32, x.target().index() as u32)));

    graph
        .edge_references()
        .map(|x| (x.source().index() as u32, x.target().index() as u32))
        .filter(|x| !tree_arcs.contains(x))
        .for_each(|x| lower_arcs.push(x));

    debug_println!("INITIAL SOLUTION : ");
    let tlu_solution = SPTree {
        t: tree_arcs,
        l: lower_arcs,
        u: upper_arcs,
    };
    debug_println!("T = {:?}", tlu_solution.t);
    debug_println!("L = {:?}", tlu_solution.l);
    debug_println!("U = {:?}", tlu_solution.u);
    (tlu_solution, graph)
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
fn compute_node_potentials<NUM: CloneableNum>(
    graph: &mut DiGraph<u32, CustomEdgeIndices<NUM>>,
    sptree: &mut SPTree,
) -> Vec<NUM> {
    let mut pi: Vec<NUM> = vec![zero(); graph.node_count()];
    let edges: Vec<(u32, u32, f32)> = sptree.t.iter().map(|&x| (x.0, x.1, 1.)).collect();
    let temp_graph = Graph::<(), f32, Undirected>::from_edges(edges);

    let path = bellman_ford(&temp_graph, NodeIndex::new(0)).unwrap();

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
fn par_update_node_potentials<NUM: CloneableNum>(
    potential: Vec<NUM>,
    sptree: &mut SPTree,
    entering_arc: (u32, u32),
    reduced_cost: &mut HashMap<(u32, u32), NUM>,
) -> Vec<NUM> {
    let mut edges: Vec<(u32, u32, f32)> = sptree
        .t
        .par_iter()
        .filter(|&&x| x != entering_arc)
        .map(|x| (x.0, x.1, 0.))
        .collect();
    edges.push((entering_arc.0, entering_arc.1, 1.)); // <-- edge separating T1 and T2
    let g = Graph::<(), f32, Undirected>::from_edges(edges);
    let path_cost = bellman_ford(&g, NodeIndex::new(0)).unwrap().distances;
    let potentials_to_update: Vec<usize> = path_cost
        .par_iter()
        .enumerate()
        .filter(|&(_, cost)| cost > &0.)
        .map(|x| x.0)
        .collect();
    let mut change: NUM = zero();
    if potentials_to_update.contains(&(entering_arc.1 as usize)) {
        change -= *reduced_cost.get(&(entering_arc.0, entering_arc.1)).unwrap();
    } else {
        change += *reduced_cost.get(&(entering_arc.0, entering_arc.1)).unwrap();
    }
    potential
        .par_iter()
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
fn compute_reduced_cost<NUM: CloneableNum>(
    pi: &mut Vec<NUM>,
    graph: &mut DiGraph<u32, CustomEdgeIndices<NUM>>,
    sptree: &mut SPTree,
) -> HashMap<(u32, u32), NUM> {
    let mut reduced_cost: HashMap<(u32, u32), NUM> = HashMap::new();
    sptree.l.iter().for_each(|&(u, v)| {
        reduced_cost.insert(
            (u, v),
            graph
                .edge_weight(
                    graph
                        .find_edge(node_index(u as usize), node_index(v as usize))
                        .unwrap(),
                )
                .unwrap()
                .cost
                - pi[u as usize]
                + pi[v as usize],
        );
    });
    sptree.u.iter().for_each(|&(u, v)| {
        reduced_cost.insert(
            (u, v),
            graph
                .edge_weight(
                    graph
                        .find_edge(node_index(u as usize), node_index(v as usize))
                        .unwrap(),
                )
                .unwrap()
                .cost
                - pi[u as usize]
                + pi[v as usize],
        );
    });
    sptree.t.iter().for_each(|&(u, v)| {
        reduced_cost.insert((u, v), zero());
    });
    reduced_cost
}

//Working but not sure about the way it is... (it work)
//probably better way to manage type Option<(_,_)>
fn _find_entering_arc<NUM: CloneableNum>(
    sptree: &mut SPTree,
    reduced_cost: &mut HashMap<(u32, u32), NUM>,
) -> Option<(u32, u32)> {
    let min_l = sptree.l.iter().min_by(|a, b| {
        reduced_cost
            .get(&(a.0, a.1))
            .unwrap()
            .partial_cmp(reduced_cost.get(&(b.0, b.1)).unwrap())
            .unwrap()
    });
    let max_u = sptree.u.iter().max_by(|a, b| {
        reduced_cost
            .get(&(a.0, a.1))
            .unwrap()
            .partial_cmp(reduced_cost.get(&(b.0, b.1)).unwrap())
            .unwrap()
    });
    let zero: NUM = zero();
    let rc_min_l = if min_l != None {
        reduced_cost.get(&(min_l.unwrap().0, min_l.unwrap().1))
    } else {
        Some(&zero)
    };
    let rc_max_u = if max_u != None {
        reduced_cost.get(&(max_u.unwrap().0, max_u.unwrap().1))
    } else {
        Some(&zero)
    };
    debug_println!("min_l = {:?}, rc = {:?}", min_l, rc_min_l);
    debug_println!("max_u = {:?} rc = {:?}", max_u, rc_max_u);
    //optimality conditions
    if rc_min_l >= Some(&zero) && rc_max_u <= Some(&zero) {
        debug_println!("OPTIMALITY CONDITIONS REACHED");
        return None;
    }
    if rc_max_u <= Some(&zero) {
        return Some(*min_l.unwrap());
    }
    if rc_min_l >= Some(&zero) {
        return Some(*max_u.unwrap());
    }
    let abs = zero - *rc_min_l.unwrap();
    if abs >= *rc_max_u.unwrap() {
        return Some(*min_l.unwrap());
    } else {
        return Some(*max_u.unwrap());
    }
}

fn par_find_entering_arc<NUM: CloneableNum>(
    sptree: &mut SPTree,
    reduced_cost: &mut HashMap<(u32, u32), NUM>,
) -> Option<(u32, u32)> {
    let min_l = sptree.l.par_iter().min_by(|a, b| {
        reduced_cost
            .get(&(a.0, a.1))
            .unwrap()
            .partial_cmp(reduced_cost.get(&(b.0, b.1)).unwrap())
            .unwrap()
    });
    let max_u = sptree.u.par_iter().max_by(|a, b| {
        reduced_cost
            .get(&(a.0, a.1))
            .unwrap()
            .partial_cmp(reduced_cost.get(&(b.0, b.1)).unwrap())
            .unwrap()
    });
    match (min_l, max_u) {
        (Some(min_l), Some(max_u)) => {
            if reduced_cost.get(min_l) >= Some(&(zero()))
                && reduced_cost.get(max_u) <= Some(&(zero()))
            {
                None
            } else if reduced_cost.get(min_l) >= Some(&(zero())) {
                Some(*max_u)
            } else if reduced_cost.get(max_u) <= Some(&(zero())) {
                Some(*min_l)
            } else if (zero::<NUM>() - *reduced_cost.get(min_l).unwrap())
                >= *reduced_cost.get(max_u).unwrap()
            {
                Some(*min_l)
            } else {
                Some(*max_u)
            }
        }
        (Some(min_l), None) => {
            if reduced_cost.get(min_l) < Some(&(zero())) {
                Some(*min_l)
            } else {
                None
            }
        }
        (None, Some(max_u)) => {
            if reduced_cost.get(max_u) > Some(&(zero())) {
                Some(*max_u)
            } else {
                None
            }
        }
        (None, None) => None,
    }
}

//Using Bellmanford algorithm to find negative weight cycle
//we build a graph with a cycle and forcing every arc weight to 0 except for one arc that we know
//in the cycle. BF then find negative weight cycle
fn find_cycle_with_arc(sptree: &mut SPTree, entering_arc: (u32, u32)) -> Vec<u32> {
    let is_in_u = sptree.u.iter().contains(&entering_arc);
    let mut edges: Vec<(u32, u32, f32)> = sptree.t.par_iter().map(|&x| (x.0, x.1, 0.)).collect();
    let mut edges1: Vec<(u32, u32, f32)> = sptree.t.par_iter().map(|&x| (x.1, x.0, 0.)).collect();
    edges.append(&mut edges1);
    if is_in_u {
        edges.push((entering_arc.1, entering_arc.0, -1.));
    } else {
        edges.push((entering_arc.0, entering_arc.1, -1.)); // <--- edge in the cycle we want
    }
    let g = Graph::<(), f32, Directed>::from_edges(edges);
    let cycle = find_negative_cycle(&g, NodeIndex::new(0));

    let mut vec_id_cycle = Vec::new();
    cycle
        .unwrap()
        .iter()
        .for_each(|&x| vec_id_cycle.push(x.index() as u32));
    while !((vec_id_cycle[0] == entering_arc.0 || vec_id_cycle[0] == entering_arc.1)
        && (vec_id_cycle[1] == entering_arc.0 || vec_id_cycle[1] == entering_arc.1))
    {
        vec_id_cycle.rotate_left(1);
    }
    vec_id_cycle
}

//checking if the specified edge is in forward direction according to a cycle
//needed in function compute_flowchange(...)
fn is_forward<NUM: CloneableNum>(
    edgeref: EdgeReference<CustomEdgeIndices<NUM>>,
    cycle: &mut Vec<u32>,
) -> bool {
    let (i, j) = (edgeref.source().index(), edgeref.target().index());
    let mut altered_cycle = cycle.clone();
    altered_cycle.push(*cycle.first().unwrap());
    let test: Vec<(u32, u32)> = altered_cycle
        .into_iter()
        .tuple_windows::<(u32, u32)>()
        .collect();
    test.contains(&(i as u32, j as u32))
}

//decompose cycle in tuple_altered_cycle variable ordered in distance to the entering arc
fn distances_in_cycle(cycle: &mut Vec<u32>) -> Vec<(u32, u32)> {
    let mut altered_cycle = cycle.clone();
    altered_cycle.push(*cycle.first().unwrap());
    let tuple_altered_cycle: Vec<(u32, u32)> = altered_cycle
        .into_iter()
        .tuple_windows::<(u32, u32)>()
        .collect();
    return tuple_altered_cycle;
}

//computing delta the amount of unit of flow we can augment through the cycle
//returning (the leaving edge:(u:u32, v:u32) , flow_change vector : Vec<(edge uv), delta(uv)>)
fn compute_flowchange<NUM: CloneableNum>(
    graph: &mut DiGraph<u32, CustomEdgeIndices<NUM>>,
    mut cycle: &mut Vec<u32>,
) -> ((u32, u32), Vec<((u32, u32), NUM)>) {
    let mut altered_cycle = cycle.clone();
    altered_cycle.push(*cycle.first().unwrap());
    let tuple_altered_cycle: Vec<(u32, u32)> = altered_cycle
        .into_iter()
        .tuple_windows::<(u32, u32)>()
        .collect();
    let mut edge_in_cycle: Vec<EdgeReference<CustomEdgeIndices<NUM>>> = graph
        .edge_references()
        .filter(|&x| {
            tuple_altered_cycle
                .iter()
                .contains(&((x.source().index() as u32), (x.target().index() as u32)))
                || tuple_altered_cycle
                    .iter()
                    .contains(&((x.target().index() as u32), (x.source().index() as u32)))
        })
        .collect();
    let distances = distances_in_cycle(&mut cycle);
    edge_in_cycle = edge_in_cycle
        .into_iter()
        .sorted_by_key(|&e| {
            distances
                .iter()
                .enumerate()
                .find(|x| {
                    x.1 == &(e.source().index() as u32, e.target().index() as u32)
                        || x.1 == &(e.target().index() as u32, e.source().index() as u32)
                })
                .unwrap()
                .0
        })
        .rev()
        .collect();
    debug_println!(
        "edge_in_cycle = {:?}",
        edge_in_cycle
            .iter()
            .map(|x| (x.source().index(), x.target().index()))
            .rev()
            .collect::<Vec<(usize, usize)>>()
    );
    let delta: Vec<NUM> = edge_in_cycle
        .iter()
        .map(|&x| {
            if is_forward(x, &mut cycle) {
                debug_println!(
                    "edge ({:?}, {:?}) is forward, delta_ij ={:?}",
                    x.source().index(),
                    x.target().index(),
                    x.weight().capacity - x.weight().flow
                );
                x.weight().capacity - x.weight().flow
            } else {
                debug_println!(
                    "edge ({:?}, {:?}) is backward delta_ij ={:?}",
                    x.source().index(),
                    x.target().index(),
                    x.weight().flow
                );
                x.weight().flow
            }
        })
        .collect();
    let flowchange = delta
        .iter()
        .enumerate()
        .min_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .unwrap();
    let farthest_blocking_edge: (u32, u32) = (
        edge_in_cycle[flowchange.0].source().index() as u32,
        edge_in_cycle[flowchange.0].target().index() as u32,
    );
    let edge_flow_change: Vec<((u32, u32), NUM)> = edge_in_cycle
        .iter()
        .map(|&x| {
            (
                (x.source().index() as u32, x.target().index() as u32),
                if is_forward(x, &mut cycle) {
                    *flowchange.1
                } else {
                    zero::<NUM>() - *flowchange.1
                },
            )
        })
        .collect();
    debug_println!(
        "farthest blocking edge = ({:?}, {:?})\namount of flow change = {:?}",
        edge_in_cycle[flowchange.0].source().index(),
        edge_in_cycle[flowchange.0].target().index(),
        flowchange.1
    );

    return (farthest_blocking_edge, edge_flow_change);
}

//updating the flow in every edge specified in flow_change vector
fn update_flow<NUM: CloneableNum>(
    graph: &mut DiGraph<u32, CustomEdgeIndices<NUM>>,
    flow_change: Vec<((u32, u32), NUM)>,
) {
    flow_change.into_iter().for_each(|((i, j), a)| {
        graph
            .edge_weight_mut(
                graph
                    .find_edge(node_index(i as usize), node_index(j as usize))
                    .unwrap(),
            )
            .unwrap()
            .flow += a;
    });
}

//updating sptree structure according to the leaving arc
//removing it from T and putting in L or U depending on the updated flow on the arc
fn par_update_sptree_with_leaving<NUM: CloneableNum>(
    sptree: &mut SPTree,
    leaving_arc: (u32, u32),
    graph: &mut DiGraph<u32, CustomEdgeIndices<NUM>>,
) {
    sptree.t = sptree
        .t
        .clone()
        .into_par_iter()
        .filter(|&x| x != leaving_arc)
        .collect();
    let at_min_capacity: bool = graph
        .edge_weight(
            graph
                .find_edge(
                    node_index(leaving_arc.0 as usize),
                    node_index(leaving_arc.1 as usize),
                )
                .unwrap(),
        )
        .unwrap()
        .flow
        == zero();
    if at_min_capacity {
        sptree.l.push(leaving_arc);
    } else {
        sptree.u.push(leaving_arc)
    }
}

//adding entering_arc to T arc set of sptree
fn par_update_sptree_with_entering(sptree: &mut SPTree, entering_arc: Option<(u32, u32)>) {
    sptree.t.push(entering_arc.unwrap());
    sptree.l = sptree
        .l
        .clone()
        .into_par_iter()
        .filter(|&x| x != entering_arc.unwrap())
        .collect();
    sptree.u = sptree
        .u
        .clone()
        .into_par_iter()
        .filter(|&x| x != entering_arc.unwrap())
        .collect();
}

//main algorithm function
pub fn min_cost<NUM: CloneableNum>(
    graph: DiGraph<u32, CustomEdgeIndices<NUM>>,
    demand: NUM,
) -> DiGraph<u32, CustomEdgeIndices<NUM>> {
    debug_println!("##################################### INITIALIZATION #####################################");

    let (mut tlu_solution, mut min_cost_flow_graph) =
        __initialization::<NUM>(graph.clone(), demand);

    let mut potentials = compute_node_potentials(&mut min_cost_flow_graph, &mut tlu_solution);
    debug_println!("potentials = {:?}", potentials);
    let mut reduced_cost =
        compute_reduced_cost(&mut potentials, &mut min_cost_flow_graph, &mut tlu_solution);
    debug_println!("tlu_solution = {:?}", tlu_solution);
    debug_println!("reduced_cost = {:?}", reduced_cost);

    let mut entering_arc = par_find_entering_arc(&mut tlu_solution, &mut reduced_cost);
    debug_println!("entering arc = {:?}", entering_arc);

    let mut iteration = 1;
    //let max_iteration = 100;

    while entering_arc != None {
        debug_println!("##################################### ITERATION {:?} #####################################", iteration);

        let mut cycle = find_cycle_with_arc(&mut tlu_solution, entering_arc.unwrap());
        debug_println!("T = {:?}", tlu_solution.t);
        debug_println!("L = {:?}", tlu_solution.l);
        debug_println!("U = {:?}", tlu_solution.u);
        println!("entering arc = {:?}", entering_arc.unwrap());

        par_update_sptree_with_entering(&mut tlu_solution, entering_arc);

        let (leaving_arc, vec_flow_change) =
            compute_flowchange(&mut min_cost_flow_graph, &mut cycle);

        println!("leaving arc = {:?}", leaving_arc);
        update_flow(&mut min_cost_flow_graph, vec_flow_change);
        par_update_sptree_with_leaving(&mut tlu_solution, leaving_arc, &mut min_cost_flow_graph);

        potentials = par_update_node_potentials(
            potentials,
            &mut tlu_solution,
            entering_arc.unwrap(),
            &mut reduced_cost,
        );

        debug_println!("node_potentials = {:?}", potentials);

        reduced_cost =
            compute_reduced_cost(&mut potentials, &mut min_cost_flow_graph, &mut tlu_solution);
        debug_println!("reduced_cost = {:?}", reduced_cost);

        debug_println!("{:?}", Dot::new(&min_cost_flow_graph));

        entering_arc = par_find_entering_arc(&mut tlu_solution, &mut reduced_cost);
        debug_println!("entering arc = {:?}", entering_arc);
        iteration += 1;

        let mut cost: NUM = zero();
        min_cost_flow_graph
            .edge_references()
            .for_each(|x| cost += x.weight().flow * x.weight().cost);
        println!("total cost = {:?}", cost);
    }
    /*if iteration == max_iteration {
        debug_println!("MAX iteration reached");
    }*/
    min_cost_flow_graph
}
