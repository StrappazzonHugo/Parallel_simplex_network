use debug_print::debug_println;
use itertools::Itertools;
use num_traits::bounds::Bounded;
use num_traits::identities::zero;
use num_traits::Num;
use petgraph::algo::bellman_ford;
use petgraph::algo::find_negative_cycle;
use petgraph::algo::is_cyclic_undirected;
use petgraph::dot::Dot;
use petgraph::graph::node_index;
use petgraph::graph::DefaultIx;
use petgraph::graph::EdgeReference;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
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

#[derive(Clone, Debug, Copy)]
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

fn initialization<'a, NUM: CloneableNum, Ix>(
    mut graph: DiGraph<u32, CustomEdgeIndices<NUM>>,
    demand: NUM,
) -> (SPTree, DiGraph<u32, CustomEdgeIndices<NUM>>) {
    let source_id = 0;
    let mut sink_id = graph
        .node_indices()
        .find(|&x| (graph.node_weight(x).unwrap() == &99u32)) //Temporary the sink is labelled with
        //id 99, easier to find after removing
        //nodes
        .unwrap();
    let mut _max_demand: NUM = zero();
    (graph, _max_demand) = max_flow(source_id, sink_id.index(), graph);
    debug_println!("_test = {:?}", Dot::new(&graph));

    debug_println!("sink id = {:?}", sink_id);

    //removing orphan nodes from the graph
    find_orphan_nodes(&mut graph).iter().for_each(|&x| {
        debug_println!("find_orphan nodes : {:?}", x);
        graph.remove_node(x).unwrap();
    });

    sink_id = graph
        .node_indices()
        .find(|&x| (graph.node_weight(x).unwrap() == &99u32))
        .unwrap();
    debug_println!("sink id = {:?}", sink_id);

    debug_println!("{:?}", Dot::new(&graph));

    assert!(demand <= _max_demand);
    resize_flow(&mut graph, demand, _max_demand, sink_id);

    let mut tree_arcs: Vec<(u32, u32)> = Vec::new();
    let mut lower_arcs: Vec<(u32, u32)> = Vec::new();
    let mut upper_arcs: Vec<(u32, u32)> = Vec::new();

    init_tlu(&mut graph, &mut tree_arcs, &mut lower_arcs, &mut upper_arcs);

    debug_println!("T = {:?}", tree_arcs);
    debug_println!("L = {:?}", lower_arcs);
    debug_println!("U = {:?}", upper_arcs);

    debug_println!("is cyclic {:?}", is_cyclic(&mut tree_arcs));

    if is_cyclic(&mut tree_arcs) {
        //let arc = T.pop().unwrap();
        let mut cycle = find_cycle(&mut SPTree {
            t: tree_arcs.clone(),
            l: lower_arcs.clone(),
            u: upper_arcs.clone(),
        });
        let (_useless, vec_flow_change) = compute_flowchange(&mut graph, &mut cycle);
        update_flow(&mut graph, vec_flow_change);

        init_tlu(&mut graph, &mut tree_arcs, &mut lower_arcs, &mut upper_arcs);
    }
    debug_println!("is cyclic {:?}", is_cyclic(&mut tree_arcs));

    // while T isnt a tree : we add one edge from U to T
    // we cannot obtain a cycle at iteration n since we necessarily
    // have a spanning tree at the iteration n-1
    while !is_tree(&mut tree_arcs, graph.node_count()) {
        for (index, edge) in upper_arcs
            .iter()
            .enumerate()
            .chain(lower_arcs.iter().enumerate())
        {
            if !is_cyclic_with_arc(&mut tree_arcs, *edge) {
                tree_arcs.push(*edge);
                if upper_arcs.contains(edge) {
                    upper_arcs.remove(index);
                } else if lower_arcs.contains(edge) {
                    lower_arcs.remove(index);
                }
                break;
            }
        }
    }

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

//Filling T arc-set with free arc of the initial feasible solution
//L with restricted at lowerbound arcs
//u with restricted at upperbound arcs
fn init_tlu<N, NUM: CloneableNum>(
    graph: &mut DiGraph<N, CustomEdgeIndices<NUM>>,
    tree_arcs: &mut Vec<(u32, u32)>,
    lower_arcs: &mut Vec<(u32, u32)>,
    upper_arcs: &mut Vec<(u32, u32)>,
) {
    let freearcs = graph
        .edge_references()
        .filter(|&x| x.weight().flow > zero() && x.weight().flow < x.weight().capacity);

    let low_bound_restricted = graph
        .edge_references()
        .filter(|&x| x.weight().flow == zero() && x.weight().capacity > zero());

    let up_bound_restricted = graph
        .edge_references()
        .filter(|&x| x.weight().flow == x.weight().capacity && x.weight().capacity > zero());

    *tree_arcs = freearcs
        .map(|x| (x.source().index() as u32, x.target().index() as u32))
        .collect();

    *lower_arcs = low_bound_restricted
        .map(|x| (x.source().index() as u32, x.target().index() as u32))
        .collect();

    *upper_arcs = up_bound_restricted
        .map(|x| (x.source().index() as u32, x.target().index() as u32))
        .collect();
}

//checking tree using number of edge property in spanning tree
fn is_tree<E>(edges: &mut Vec<E>, n: usize) -> bool {
    edges.len() + 1 == n
}

fn is_cyclic_with_arc(t: &mut Vec<(u32, u32)>, arc: (u32, u32)) -> bool {
    let mut g = Graph::<(), (), Undirected>::from_edges(t.clone());
    g.extend_with_edges(&[arc]);
    petgraph::algo::is_cyclic_undirected(&g)
}

fn is_cyclic(t: &mut Vec<(u32, u32)>) -> bool {
    let g = Graph::<(), (), Undirected>::from_edges(t.clone());
    petgraph::algo::is_cyclic_undirected(&g)
}
//iterate over nodes and checking : sum(flow in adjacent edges) > 0
fn find_orphan_nodes<N, NUM: CloneableNum>(
    graph: &mut DiGraph<N, CustomEdgeIndices<NUM>>,
) -> Vec<petgraph::graph::NodeIndex> {
    graph
        .node_indices()
        .rev()
        .skip(1)
        .filter(|&u| {
            graph
                .edges(u)
                .fold(true, |check, edge| check && edge.weight().flow == zero())
        })
        .collect()
}

//resize the flow such that it match the demand
fn resize_flow<NUM: CloneableNum>(
    graph: &mut DiGraph<u32, CustomEdgeIndices<NUM>>,
    demand: NUM,
    max_demand: NUM,
    sink_id: NodeIndex,
) {
    if demand > max_demand {
        return; //TODO flow not feasible
    }
    if demand == max_demand {
        return;
    }

    let graph_before_resize = graph.clone();
    let mut restricted_arcs: Vec<EdgeReference<CustomEdgeIndices<NUM>>> = graph_before_resize
        .edge_references()
        .filter(|x| x.target() == sink_id || x.source() == sink_id)
        .collect();
    let free_arc: Option<(usize, EdgeReference<CustomEdgeIndices<NUM>>)> = restricted_arcs
        .clone()
        .into_iter()
        .find_position(|x| x.weight().flow < x.weight().capacity);
    if !free_arc.is_none() {
        restricted_arcs.remove(free_arc.unwrap().0);
    }
    let mut sum_capacity: NUM = zero();
    let mut bool = false;

    for edge in restricted_arcs.iter() {
        sum_capacity += edge.weight().flow;

        if bool {
            graph.edge_weight_mut(edge.id()).unwrap().capacity = zero();
        }
        if sum_capacity >= demand {
            bool = true;
            graph.edge_weight_mut(edge.id()).unwrap().capacity =
                demand - (sum_capacity - graph.edge_weight_mut(edge.id()).unwrap().flow);
        }
    }

    if !free_arc.is_none() {
        if bool {
            graph
                .edge_weight_mut(free_arc.unwrap().1.id())
                .unwrap()
                .capacity = zero();
            debug_println!(
                "free_arc = {:?}",
                graph.edge_weight(free_arc.unwrap().1.id()).unwrap()
            );
        } else {
            graph
                .edge_weight_mut(free_arc.unwrap().1.id())
                .unwrap()
                .capacity = demand - sum_capacity;
        }
    }

    graph
        .clone()
        .edge_references()
        .for_each(|x| graph.edge_weight_mut(x.id()).unwrap().flow = zero());
    debug_println!();
    let (resized_flow, max_demand) = max_flow(0, sink_id.index(), graph.clone());

    debug_println!("RESIZED_FLOW =  {:?}", Dot::new(&(resized_flow)));
    resized_flow.edge_references().for_each(|x| {
        graph.edge_weight_mut(x.id()).unwrap().flow = x.weight().flow;

        graph.edge_weight_mut(x.id()).unwrap().capacity =
            graph_before_resize.edge_weight(x.id()).unwrap().capacity;
    });

    debug_println!("max_demand = {:?}", max_demand);
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
fn compute_node_potentials<N, NUM: CloneableNum>(
    graph: &mut DiGraph<N, CustomEdgeIndices<NUM>>,
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

//Updating potential inspired by : NETWORK FLOWS Theory, Algorithms, and Applications (p.419)
//modified such that the difference from T1 and T2 is done with a BellmanFord algorithm
//detecting the single edge cost put to 1 that divide T1 and T2 in T
fn update_node_potentials<NUM: CloneableNum>(
    potential: Vec<NUM>,
    sptree: &mut SPTree,
    entering_arc: (u32, u32),
    reduced_cost: &mut HashMap<(i32, i32), NUM>,
) -> Vec<NUM> {
    let mut edges: Vec<(u32, u32, f32)> = sptree
        .t
        .iter()
        .filter(|&&x| x != entering_arc)
        .map(|x| (x.0, x.1, 0.))
        .collect();
    edges.push((entering_arc.0, entering_arc.1, 1.)); // <-- edge separating T1 and T2
    let g = Graph::<(), f32, Undirected>::from_edges(edges);
    let path_cost = bellman_ford(&g, NodeIndex::new(0)).unwrap().distances;
    let potentials_to_update: Vec<usize> = path_cost
        .iter()
        .enumerate()
        .filter(|&(_, cost)| cost > &0.)
        .map(|x| x.0)
        .collect();
    let mut change: NUM = zero();
    if potentials_to_update.contains(&(entering_arc.1 as usize)) {
        change -= *reduced_cost
            .get(&(entering_arc.0 as i32, entering_arc.1 as i32))
            .unwrap();
    } else {
        change += *reduced_cost
            .get(&(entering_arc.0 as i32, entering_arc.1 as i32))
            .unwrap();
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
fn compute_reduced_cost<N, NUM: CloneableNum>(
    pi: &mut Vec<NUM>,
    graph: &mut DiGraph<N, CustomEdgeIndices<NUM>>,
    sptree: &mut SPTree,
) -> HashMap<(i32, i32), NUM> {
    let mut reduced_cost: HashMap<(i32, i32), NUM> = HashMap::new();
    sptree.l.iter().for_each(|&(u, v)| {
        reduced_cost.insert(
            (u as i32, v as i32),
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
            (u as i32, v as i32),
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
        reduced_cost.insert((u as i32, v as i32), zero());
    });
    reduced_cost
}

//Working but not sure about the way it is... (it work)
//probably better way to manage type Option<(_,_)>
fn find_entering_arc<NUM: CloneableNum>(
    sptree: &mut SPTree,
    reduced_cost: &mut HashMap<(i32, i32), NUM>,
) -> Option<(u32, u32)> {
    let min_l = sptree.l.iter().min_by(|a, b| {
        reduced_cost
            .get(&(a.0 as i32, a.1 as i32))
            .unwrap()
            .partial_cmp(reduced_cost.get(&(b.0 as i32, b.1 as i32)).unwrap())
            .unwrap()
    });
    let max_u = sptree.u.iter().max_by(|a, b| {
        reduced_cost
            .get(&(a.0 as i32, a.1 as i32))
            .unwrap()
            .partial_cmp(reduced_cost.get(&(b.0 as i32, b.1 as i32)).unwrap())
            .unwrap()
    });
    let zero: NUM = zero();
    let rc_min_l = if min_l != None {
        reduced_cost.get(&(min_l.unwrap().0 as i32, min_l.unwrap().1 as i32))
    } else {
        Some(&zero)
    };
    let rc_max_u = if max_u != None {
        reduced_cost.get(&(max_u.unwrap().0 as i32, max_u.unwrap().1 as i32))
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

//Using Bellmanford algorithm to find negative weight cycle
//we build a graph with a cycle and forcing every arc weight to 0 except for one arc that we know
//in the cycle. BF then find negative weight cycle
fn find_cycle_with_arc(sptree: &mut SPTree, entering_arc: (u32, u32)) -> Vec<u32> {
    let is_in_u = sptree.u.iter().contains(&entering_arc);
    let mut edges: Vec<(u32, u32, f32)> = sptree.t.iter().map(|&x| (x.0, x.1, 0.)).collect();
    let mut edges1: Vec<(u32, u32, f32)> = sptree.t.iter().map(|&x| (x.1, x.0, 0.)).collect();
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

//Using Bellmanford algorithm to find negative weight cycle
//we build a graph with a cycle and forcing every arc weight to 0 except for one arc that we know
//in the cycle. BF then find negative weight cycle
fn find_cycle(sptree: &mut SPTree) -> Vec<u32> {
    let mut edges: Vec<(u32, u32, f32)> = sptree.t.iter().map(|&x| (x.0, x.1, 0.)).collect();
    let mut g = Graph::<(), f32, Undirected>::from_edges(edges.clone());
    let mut edge: (usize, &(u32, u32, f32)) = (0, &(0, 0, 0.));
    for x in edges.iter().enumerate() {
        edge = x;
        g.remove_edge(
            g.find_edge(
                NodeIndex::new(edge.1 .0 as usize),
                NodeIndex::new(edge.1 .1 as usize),
            )
            .unwrap(),
        );
        if !is_cyclic_undirected(&g) {
            break;
        }
    }
    let new_edge = (edge.1 .0, edge.1 .1, -1.);
    let index = edge.0;
    edges.remove(index);
    let mut reverse_edge: Vec<(u32, u32, f32)> = sptree.t.iter().map(|&x| (x.1, x.0, 0.)).collect();
    reverse_edge.remove(index);
    edges.append(&mut reverse_edge);
    edges.push(new_edge);
    let g = Graph::<(), f32, Directed>::from_edges(edges);
    let cycle = find_negative_cycle(&g, NodeIndex::new(0));

    let mut vec_id_cycle = Vec::new();
    cycle
        .unwrap()
        .iter()
        .for_each(|&x| vec_id_cycle.push(x.index() as u32));
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

//decompose cycle in tuple_altered_cycle variable and count distance using index in vector
fn distances_in_cycle(cycle: &mut Vec<u32>) -> Vec<(u32, u32)> {
    let mut altered_cycle = cycle.clone();
    altered_cycle.push(*cycle.first().unwrap());
    let tuple_altered_cycle: Vec<(u32, u32)> = altered_cycle
        .into_iter()
        .tuple_windows::<(u32, u32)>()
        .collect();
    /*let res = tuple_altered_cycle
    .iter()
    .enumerate()
    .find(|&(_, x)| x == &(i as u32, j as u32) || x == &(j as u32, i as u32))
    .unwrap();*/
    return tuple_altered_cycle;
}

//computing delta the amount of unit of flow we can augment through the cycle
//returning (the leaving edge:(u:u32, v:u32) , flow_change vector : Vec<(edge uv), delta(uv)>)
fn compute_flowchange<N, NUM: CloneableNum>(
    graph: &mut DiGraph<N, CustomEdgeIndices<NUM>>,
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
fn update_flow<N: Clone, NUM: CloneableNum>(
    graph: &mut DiGraph<N, CustomEdgeIndices<NUM>>,
    flow_change: Vec<((u32, u32), NUM)>,
) {
    flow_change.iter().for_each(|((i, j), a)| {
        graph
            .edge_weight_mut(
                graph
                    .find_edge(node_index(*i as usize), node_index(*j as usize))
                    .unwrap(),
            )
            .unwrap()
            .flow += *a;
    });
}

//updating sptree structure according to the leaving arc
//removing it from T and putting in L or U depending on the updated flow on the arc
fn update_sptree_with_leaving<N, NUM: CloneableNum>(
    sptree: &mut SPTree,
    leaving_arc: (u32, u32),
    graph: &mut DiGraph<N, CustomEdgeIndices<NUM>>,
) -> SPTree {
    let mut updated_sptree = sptree.clone();
    updated_sptree.t = updated_sptree
        .t
        .into_iter()
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
        updated_sptree.l.push(leaving_arc);
    } else {
        updated_sptree.u.push(leaving_arc)
    }
    updated_sptree
}

//adding entering_arc to T arc set of sptree
fn update_sptree_with_entering(sptree: &mut SPTree, entering_arc: Option<(u32, u32)>) -> SPTree {
    let mut updated_sptree = sptree.clone();
    updated_sptree.t.push(entering_arc.unwrap());

    updated_sptree.l = updated_sptree
        .l
        .into_iter()
        .filter(|&x| x != entering_arc.unwrap())
        .collect();
    updated_sptree.u = updated_sptree
        .u
        .into_iter()
        .filter(|&x| x != entering_arc.unwrap())
        .collect();
    updated_sptree
}

///////////////////////////
//Push-Relabel algorithm
///////////////////////////
fn push<N, NUM: CloneableNum>(
    edgeref: &mut EdgeReference<CustomEdgeIndices<NUM>>,
    graph: &mut DiGraph<N, CustomEdgeIndices<NUM>>,
    excess: &mut Vec<NUM>,
    excess_vertices: &mut Vec<usize>,
) {
    let d_vec = vec![
        excess[edgeref.source().index()],
        graph.edge_weight(edgeref.id()).unwrap().capacity
            - graph.edge_weight(edgeref.id()).unwrap().flow,
    ];
    let d: NUM = *d_vec
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    graph.edge_weight_mut(edgeref.id()).unwrap().flow += d;
    graph
        .edge_weight_mut(graph.find_edge(edgeref.target(), edgeref.source()).unwrap())
        .unwrap()
        .flow -= d;
    excess[edgeref.source().index()] -= d;
    excess[edgeref.target().index()] += d;
    if d != zero() && excess[edgeref.target().index()] == d {
        excess_vertices.push(edgeref.target().index());
    }
}

fn relabel<N, NUM: CloneableNum>(
    u: usize,
    graph: &mut DiGraph<N, CustomEdgeIndices<NUM>>,
    label: &mut Vec<i32>,
) {
    let mut d = Bounded::max_value();
    graph
        .edges_directed(NodeIndex::new(u), Outgoing)
        .for_each(|x| {
            if (x.weight().capacity - x.weight().flow) > zero() {
                d = std::cmp::min(d, label[x.target().index()]);
            };
        });
    graph
        .edges_directed(NodeIndex::new(u), Incoming)
        .for_each(|x| {
            if (x.weight().capacity - x.weight().flow) > zero() {
                d = std::cmp::min(d, label[x.target().index()]);
            };
        });
    if d < Bounded::max_value() {
        label[u] = d + 1;
    }
}
fn discharge<NUM: CloneableNum>(
    u: usize,
    graph: &mut DiGraph<u32, CustomEdgeIndices<NUM>>,
    excess: &mut Vec<NUM>,
    excess_vertices: &mut Vec<usize>,
    seen: &mut Vec<usize>,
    label: &mut Vec<i32>,
) {
    //Maybe not necessary to clone here
    let temp_graph = graph.clone();

    while excess[u] > zero() {
        if seen[u] < graph.node_count() {
            let v = seen[u];
            let edge_uv = graph.find_edge(NodeIndex::new(u), NodeIndex::new(v));
            if edge_uv != None
                && (graph.edge_weight(edge_uv.unwrap()).unwrap().capacity
                    - graph.edge_weight(edge_uv.unwrap()).unwrap().flow)
                    > zero()
                && label[u] > label[v]
            {
                let mut edgeref = temp_graph
                    .edge_references()
                    .find(|x| x.source().index() == u && x.target().index() == v)
                    .unwrap();

                push(&mut edgeref, graph, excess, excess_vertices);
            } else {
                seen[u] += 1;
            }
        } else {
            relabel(u, graph, label);
            seen[u] = 0;
        }
    }
}

fn max_flow<NUM: CloneableNum>(
    s: usize,
    t: usize,
    graph: DiGraph<u32, CustomEdgeIndices<NUM>>,
) -> (DiGraph<u32, CustomEdgeIndices<NUM>>, NUM) {
    let mut label = vec![0i32; graph.node_count()];
    label[s] = graph.node_count() as i32;
    let mut excess = vec![zero(); graph.node_count()];
    excess[s] = Bounded::max_value();
    let mut seen = vec![0; graph.node_count()];
    let mut excess_vertices: Vec<usize> = vec![];
    let mut max_flow_graph = graph.clone();

    graph.edge_references().for_each(|x| {
        max_flow_graph.add_edge(
            x.target(),
            x.source(),
            CustomEdgeIndices {
                cost: (zero()),
                capacity: (zero()),
                flow: (zero()),
            },
        );
    });

    graph
        .edges_directed(NodeIndex::new(s), Outgoing)
        .for_each(|mut x| {
            push(
                &mut x,
                &mut max_flow_graph,
                &mut excess,
                &mut excess_vertices,
            )
        });

    while !excess_vertices.is_empty() {
        let u = excess_vertices.pop();
        if u.unwrap() != s && u.unwrap() != t {
            discharge(
                u.unwrap(),
                &mut max_flow_graph,
                &mut excess,
                &mut excess_vertices,
                &mut seen,
                &mut label,
            );
        }
    }

    let mut max_flow = zero();
    max_flow_graph
        .clone()
        .edge_references()
        .rev()
        .for_each(|x| {
            if graph.find_edge(x.source(), x.target()) == None {
                max_flow_graph.remove_edge(x.id());
            }
        });
    max_flow_graph
        .edges(NodeIndex::new(0))
        .for_each(|x| max_flow = x.weight().flow + max_flow);
    debug_println!("MAX_FLOW = {:?}", max_flow);
    (max_flow_graph, max_flow)
}

//main algorithm function
pub fn min_cost<NUM: CloneableNum>(
    graph: DiGraph<u32, CustomEdgeIndices<NUM>>,
    demand: NUM,
) -> DiGraph<u32, CustomEdgeIndices<NUM>> {
    debug_println!("##################################### INITIALIZATION #####################################");

    let (mut tlu_solution, mut min_cost_flow_graph) =
        initialization::<NUM, DefaultIx>(graph.clone(), demand);

    let mut potentials = compute_node_potentials(&mut min_cost_flow_graph, &mut tlu_solution);
    debug_println!("potentials = {:?}", potentials);
    let mut reduced_cost =
        compute_reduced_cost(&mut potentials, &mut min_cost_flow_graph, &mut tlu_solution);
    debug_println!("tlu_solution = {:?}", tlu_solution);
    debug_println!("reduced_cost = {:?}", reduced_cost);

    let mut entering_arc = find_entering_arc(&mut tlu_solution, &mut reduced_cost);
    debug_println!("entering arc = {:?}", entering_arc);

    let mut iteration = 1;
    let max_iteration = 100;

    while entering_arc != None && iteration < max_iteration {
        debug_println!("##################################### ITERATION {:?} #####################################", iteration);

        let mut cycle = find_cycle_with_arc(&mut tlu_solution, entering_arc.unwrap());
        debug_println!("T = {:?}", tlu_solution.t);
        debug_println!("L = {:?}", tlu_solution.l);
        debug_println!("U = {:?}", tlu_solution.u);
        debug_println!("entering arc = {:?}", entering_arc.unwrap());

        tlu_solution = update_sptree_with_entering(&mut tlu_solution, entering_arc);

        let (leaving_arc, vec_flow_change) =
            compute_flowchange(&mut min_cost_flow_graph, &mut cycle);

        update_flow(&mut min_cost_flow_graph, vec_flow_change);
        tlu_solution =
            update_sptree_with_leaving(&mut tlu_solution, leaving_arc, &mut min_cost_flow_graph);

        potentials = update_node_potentials(
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

        entering_arc = find_entering_arc(&mut tlu_solution, &mut reduced_cost);
        debug_println!("entering arc = {:?}", entering_arc);
        iteration += 1;

        let mut cost: NUM = zero();
        min_cost_flow_graph
            .edge_references()
            .for_each(|x| cost += x.weight().flow * x.weight().cost);
        debug_println!("total cost = {:?}", cost);
    }
    if iteration == max_iteration {
        debug_println!("MAX iteration reached");
    }
    min_cost_flow_graph
}
