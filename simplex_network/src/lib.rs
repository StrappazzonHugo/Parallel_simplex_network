use itertools::Itertools;
use num_traits::identities::zero;
use num_traits::Num;
use petgraph::algo::bellman_ford;
use petgraph::graph::NodeIndex;
use petgraph::matrix_graph::*;
use petgraph::prelude::*;
use petgraph::visit::*;
//use rayon::prelude::*;
use petgraph::dot::Dot;
use State::*;

#[derive(Clone, Debug, Copy, PartialEq, PartialOrd)]
pub struct CustomEdgeIndices<NUM: CloneableNum> {
    pub cost: NUM,
    pub capacity: NUM,
    pub flow: NUM,
    pub state: State,
}

#[derive(Clone, Debug, Copy, PartialEq, PartialOrd)]
pub enum State {
    TreeArc,
    LowerRestricted,
    UpperRestricted,
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
fn initialization<NUM: CloneableNum>(
    graph: &mut DiMatrix<u32, CustomEdgeIndices<NUM>>,
    demand: NUM,
) -> Vec<Option<NodeIndex<u16>>> {
    let initial_number_of_node: u32 = (graph.node_count() - 1) as u32;
    let sink_id = graph
        .node_identifiers()
        .find(|&x| (graph.node_weight(x) == &initial_number_of_node))
        .unwrap();

    let mut big_value: NUM = zero();
    for _ in 0..100 {
        big_value += demand;
    }

    graph
        .clone()
        .edge_references()
        .for_each(|x| graph.edge_weight_mut(x.0, x.1).state = LowerRestricted);

    let artificial_root = graph.add_node(graph.node_count() as u32);
    println!("artificial root = {:?}", artificial_root);
    for node in graph.clone().node_identifiers() {
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
                    state: TreeArc,
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
                    state: TreeArc,
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
                    state: TreeArc,
                },
            );
        }
    }

    let mut predecessors = vec![Some(artificial_root); graph.node_count() - 1];
    predecessors.push(None);
    predecessors
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
    graph: &mut DiMatrix<u32, CustomEdgeIndices<NUM>>,
) -> Vec<NUM> {
    let mut pi: Vec<NUM> = vec![zero(); graph.node_count()];
    let edges: Vec<(u32, u32, f32)> = graph
        .edge_references()
        .filter(|x| x.2.state == TreeArc)
        .map(|x| (x.0.index() as u32, x.1.index() as u32, 1f32))
        .collect();
    let temp_graph = Graph::<u16, f32, Undirected>::from_edges(edges);

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
        pi[id] = if graph.has_edge(NodeIndex::new(pred.unwrap().index()), NodeIndex::new(id)) {
            pi[pred.unwrap().index()]
                - graph
                    .edge_weight(NodeIndex::new(pred.unwrap().index()), NodeIndex::new(id))
                    .cost
        } else {
            graph
                .edge_weight(NodeIndex::new(id), NodeIndex::new(pred.unwrap().index()))
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
    graph: &DiMatrix<u32, CustomEdgeIndices<NUM>>,
    potential: Vec<NUM>,
    entering_arc: (NodeIndex<u16>, NodeIndex<u16>),
) -> Vec<NUM> {
    let mut edges: Vec<(u32, u32, f32)> = graph
        .edge_references()
        .filter(|x| x.2.state == TreeArc && !(x.0 == entering_arc.0 && x.1 == entering_arc.1))
        .map(|x| (x.0.index() as u32, x.1.index() as u32, 0f32))
        .collect();
    edges.push((
        entering_arc.0.index() as u32,
        entering_arc.1.index() as u32,
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
    let rc = get_reduced_cost_edge(graph, entering_arc, &potential);
    if potentials_to_update.contains(&entering_arc.1.index()) {
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

fn find_cycle_with_arc(
    entering_arc: (NodeIndex<u16>, NodeIndex<u16>),
    predecessors: Vec<Option<NodeIndex<u16>>>,
) -> Vec<NodeIndex<u16>> {
    let (i, j) = (entering_arc.0, entering_arc.1);
    let mut path_from_i: Vec<NodeIndex<u16>> = Vec::new();
    let mut path_from_j: Vec<NodeIndex<u16>> = Vec::new();

    let mut current_node: Option<NodeIndex<u16>> = Some(i);
    while !current_node.is_none() {
        path_from_i.push(current_node.unwrap());
        current_node = predecessors[current_node.unwrap().index()];
    }
    let mut current_node = Some(j);
    while !current_node.is_none() {
        path_from_j.push(current_node.unwrap());
        current_node = predecessors[current_node.unwrap().index()];
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
fn is_forward(edgeref: (NodeIndex<u16>, NodeIndex<u16>), cycle: &mut Vec<NodeIndex<u16>>) -> bool {
    distances_in_cycle(cycle).contains(&edgeref)
}

//decompose cycle in tuple_altered_cycle variable ordered in distance to the entering arc
fn distances_in_cycle(cycle: &mut Vec<NodeIndex<u16>>) -> Vec<(NodeIndex<u16>, NodeIndex<u16>)> {
    let mut altered_cycle = cycle.clone();
    altered_cycle.push(*cycle.first().unwrap());
    let tuple_altered_cycle: Vec<(NodeIndex<u16>, NodeIndex<u16>)> = altered_cycle
        .into_iter()
        .tuple_windows::<(NodeIndex<u16>, NodeIndex<u16>)>()
        .collect();
    tuple_altered_cycle
}

//computing delta the amount of unit of flow we can augment through the cycle
//returning (the leaving edge
fn compute_flowchange<NUM: CloneableNum>(
    graph: &mut DiMatrix<u32, CustomEdgeIndices<NUM>>,
    cycle: &mut Vec<NodeIndex<u16>>,
    entering_arc: (NodeIndex<u16>, NodeIndex<u16>),
) -> (NodeIndex<u16>, NodeIndex<u16>)
//Vec<(EdgeReference<'a, CustomEdgeIndices<NUM>>, NUM)>,
{
    let mut edge_in_cycle: Vec<(NodeIndex<u16>, NodeIndex<u16>)> = Vec::new();

    distances_in_cycle(cycle).iter().for_each(|(u, v)| {
        if graph.clone().has_edge(*u, *v) {
            edge_in_cycle.push((*u, *v));
        } else {
            edge_in_cycle.push((*v, *u));
        }
    });

    if graph.edge_weight(entering_arc.0, entering_arc.1).state == UpperRestricted {
        cycle.reverse();
    };

    let delta: Vec<NUM> = edge_in_cycle
        .iter()
        .map(|&x| {
            if is_forward((x.0, x.1), cycle) {
                graph.edge_weight(x.0, x.1).capacity - graph.edge_weight(x.0, x.1).flow
            } else {
                graph.edge_weight(x.0, x.1).flow
            }
        })
        .collect();
    let flowchange = delta
        .iter()
        .enumerate()
        .min_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .unwrap();
    let farthest_blocking_edge: (NodeIndex<u16>, NodeIndex<u16>) = edge_in_cycle[flowchange.0];
    if *flowchange.1 != zero::<NUM>() {
        edge_in_cycle.iter().for_each(|x| {
            graph.edge_weight_mut(x.0, x.1).flow += if is_forward((x.0, x.1), cycle) {
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
    graph: &mut DiMatrix<u32, CustomEdgeIndices<NUM>>,
    entering_arc: &mut (NodeIndex<u16>, NodeIndex<u16>),
    leaving_arc: &mut (NodeIndex<u16>, NodeIndex<u16>),
    predecessors: &mut Vec<Option<NodeIndex<u16>>>,
) {
    if entering_arc != leaving_arc {
        let (i, j) = (entering_arc.0, entering_arc.1);
        let (k, l) = (leaving_arc.0, leaving_arc.1);
        if predecessors[k.index()] == Some(l) {
            predecessors[k.index()] = None;
        } else {
            predecessors[l.index()] = None;
        }
        let mut path_from_i: Vec<NodeIndex<u16>> = Vec::new();
        let mut path_from_j: Vec<NodeIndex<u16>> = Vec::new();

        let mut current_node: Option<NodeIndex<u16>> = Some(i);
        while !current_node.is_none() {
            path_from_i.push(current_node.unwrap());
            current_node = predecessors[current_node.unwrap().index()];
        }
        current_node = Some(j);
        while !current_node.is_none() {
            path_from_j.push(current_node.unwrap());
            current_node = predecessors[current_node.unwrap().index()];
        }
        if path_from_i[path_from_i.len() - 1] != NodeIndex::new(graph.node_count() - 1) {
            predecessors[i.index()] = Some(j);
            path_from_i
                .iter()
                .enumerate()
                .skip(1)
                .for_each(|(index, &x)| predecessors[x.index()] = Some(path_from_i[index - 1]));
        } else {
            predecessors[j.index()] = Some(i);
            path_from_j
                .iter()
                .enumerate()
                .skip(1)
                .for_each(|(index, &x)| predecessors[x.index()] = Some(path_from_j[index - 1]));
        }
    }
    graph.edge_weight_mut(entering_arc.0, entering_arc.1).state = TreeArc;
    if graph.edge_weight(leaving_arc.0, leaving_arc.1).flow == zero() {
        graph.edge_weight_mut(leaving_arc.0, leaving_arc.1).state = LowerRestricted;
    } else {
        graph.edge_weight_mut(leaving_arc.0, leaving_arc.1).state = UpperRestricted;
    }
}

///////////////////////
///////////////////////

fn get_reduced_cost_edge<NUM: CloneableNum>(
    graph: &DiMatrix<u32, CustomEdgeIndices<NUM>>,
    edge: (NodeIndex<u16>, NodeIndex<u16>),
    potential: &Vec<NUM>,
) -> NUM {
    graph.edge_weight(edge.0, edge.1).cost - potential[edge.0.index()] + potential[edge.1.index()]
}

fn find_best_eligible_arc<NUM: CloneableNum>(
    graph: &DiMatrix<u32, CustomEdgeIndices<NUM>>,
    potential: &Vec<NUM>,
    block: &Vec<(usize, (NodeIndex<u16>, NodeIndex<u16>))>,
) -> Option<usize> {
    let (index, rc_entering_arc) = block
        .iter()
        .map(|(index, arc)| {
            (
                index,
                if graph.edge_weight(arc.0, arc.1).flow == zero() {
                    get_reduced_cost_edge(graph, *arc, potential)
                } else {
                    zero::<NUM>() - get_reduced_cost_edge(graph, *arc, potential)
                },
            )
        })
        .min_by(|&x, &y| x.1.partial_cmp(&y.1).unwrap())
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

//Best Eligible arc
fn _find_best_arc<'a, NUM: CloneableNum>(
    graph: &DiMatrix<u32, CustomEdgeIndices<NUM>>,
    potential: &Vec<NUM>,
) -> Option<(NodeIndex<u16>, NodeIndex<u16>)> {
    let map = graph.visit_map();
    let mut min_l = graph
        .edge_references()
        .filter(|e| e.2.state == LowerRestricted)
        .min_by(|&x, &y| {
            (get_reduced_cost_edge(graph, (x.0, x.1), potential))
                .partial_cmp(&(get_reduced_cost_edge(graph, (y.0, y.1), potential)))
                .unwrap()
        });
    let mut max_u = graph
        .edge_references()
        .filter(|e| e.2.state == UpperRestricted)
        .max_by(|&x, &y| {
            (get_reduced_cost_edge(graph, (x.0, x.1), potential))
                .partial_cmp(&(get_reduced_cost_edge(graph, (y.0, y.1), potential)))
                .unwrap()
        });
    let mut rcmin = None;
    let mut rcmax = None;
    if !min_l.is_none() {
        rcmin = Some(get_reduced_cost_edge(
            graph,
            (min_l.unwrap().0, min_l.unwrap().1),
            potential,
        ));
    };
    if !max_u.is_none() {
        rcmax = Some(get_reduced_cost_edge(
            graph,
            (max_u.unwrap().0, max_u.unwrap().1),
            potential,
        ));
    };
    if !min_l.is_none() && rcmin.unwrap() >= zero() {
        min_l = None;
    }
    if !max_u.is_none() && rcmax.unwrap() <= zero() {
        max_u = None;
    }
    match (min_l, max_u) {
        (None, None) => None,
        (_, None) => Some((min_l.unwrap().0, min_l.unwrap().1)),
        (None, _) => Some((max_u.unwrap().0, max_u.unwrap().1)),
        (_, _) => {
            if (zero::<NUM>() - rcmin.unwrap()) >= rcmax.unwrap() {
                Some((min_l.unwrap().0, min_l.unwrap().1))
            } else {
                Some((max_u.unwrap().0, max_u.unwrap().1))
            }
        }
    }
}

//First eligible
fn _find_first_arc<'a, NUM: CloneableNum>(
    graph: &'a DiMatrix<u32, CustomEdgeIndices<NUM>>,
    potential: &Vec<NUM>,
) -> Option<(NodeIndex<u16>, NodeIndex<u16>)> {
    let edge_vec: Vec<(NodeIndex<u16>, NodeIndex<u16>, &CustomEdgeIndices<NUM>)> = graph
        .edge_references()
        .filter(|x| x.2.state != TreeArc)
        .collect();
    let entering_arc = edge_vec.iter().find(|&&arc| {
        let rc = if graph.edge_weight(arc.0, arc.1).flow == zero() {
            get_reduced_cost_edge(graph, (arc.0, arc.1), potential)
        } else {
            zero::<NUM>() - get_reduced_cost_edge(graph, (arc.0, arc.1), potential)
        };
        rc < zero()
    });
    if entering_arc.is_none() {
        None
    } else {
        Some((entering_arc.unwrap().0, entering_arc.unwrap().1))
    }
}

//Block search
fn _find_block_search<'a, NUM: CloneableNum>(
    graph: &DiMatrix<u32, CustomEdgeIndices<NUM>>,
    potential: &Vec<NUM>,
    mut index: usize,
    block_size: usize,
) -> (
    Option<usize>,
    Option<(NodeIndex<u16>, NodeIndex<u16>)>,
) {
    let mut block_number = 0;
    let mut arc_index = None;
    let edge_vec: Vec<(NodeIndex<u16>, NodeIndex<u16>, &CustomEdgeIndices<NUM>)> = graph.edge_references().collect();

    while block_number * block_size <= graph.edge_count() {
        arc_index = find_best_eligible_arc(
            graph,
            &potential,
            &(edge_vec
                .iter()
                .enumerate()
                .cycle()
                .skip(index)
                .take(block_size)
                .map(|(index, x)| (index, (x.0, x.1)))
                .collect()),
        );
        if arc_index.is_none() {
            index += block_size;
            block_number += 1;
        } else {
            break;
        }
    }
    if arc_index.is_none() {
        (None, None)
    } else {
        (arc_index, Some((edge_vec[arc_index.unwrap()].0, edge_vec[arc_index.unwrap()].1)))
    }
}

//main algorithm function
pub fn min_cost<NUM: CloneableNum>(
    mut graph: DiMatrix<u32, CustomEdgeIndices<NUM>>,
    demand: NUM,
) -> DiMatrix<u32, CustomEdgeIndices<NUM>> {
    let mut predecessors = initialization::<NUM>(&mut graph, demand);
    let mut potentials = compute_node_potentials(&mut graph);

    /*let block_size = graph.edge_count() / 17 as usize;
    let (mut index, mut entering_arc) = _find_block_search(
        &mut graph,
        &mut potentials,
        0,
        block_size,
    );*/

    let mut entering_arc = _find_best_arc(&mut graph, &mut potentials);
    //let mut entering_arc = _find_first_arc(&mut graph, &mut potentials);

    let mut _iteration = 0;
    while entering_arc != None {
        //println!("iteration {:?}", _iteration);
        let mut cycle = find_cycle_with_arc(entering_arc.unwrap(), predecessors.clone());
        //println!("cycle {:?}", cycle);
        //println!("entering arc {:?}", entering_arc);
        let mut leaving_arc = compute_flowchange(&mut graph, &mut cycle, entering_arc.unwrap());
        //println!("leaving_arc {:?}", leaving_arc);

        update_sptree(
            &mut graph,
            &mut entering_arc.unwrap(),
            &mut leaving_arc,
            &mut predecessors,
        );

        //println!("predecessor {:?}", predecessors);
        potentials = update_node_potentials(&graph, potentials, entering_arc.unwrap());

        entering_arc = _find_best_arc(&mut graph, &mut potentials);

        //entering_arc = _find_first_arc(&mut graph, &mut potentials);

        /*(index, entering_arc) = _find_block_search(
            &mut graph,
            &mut potentials,
            index.unwrap(),
            block_size,
        );*/

        _iteration += 1;
    }
    let mut cost: NUM = zero();
    println!("ITERATION {:?}", _iteration);
    graph
        .edge_references()
        .for_each(|x| cost += x.weight().flow * x.weight().cost);
    println!("total cost = {:?}", cost);
    graph
}
