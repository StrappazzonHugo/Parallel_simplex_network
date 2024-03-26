use itertools::Itertools;
use num_traits::identities::one;
use num_traits::identities::zero;
use num_traits::Num;
use petgraph::algo::bellman_ford;
use petgraph::graph::EdgeIndex;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::stable_graph::IndexType;
use petgraph::visit::depth_first_search;
use petgraph::visit::Control;
use petgraph::visit::DfsEvent;
//use rayon::prelude::*;

#[derive(Debug, Clone)]
struct SPTree {
    in_base: Vec<EdgeIndex>,
    out_base: Vec<EdgeIndex>,
    pred: Vec<Option<NodeIndex>>,
    thread: Vec<NodeIndex>,
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

    let mut tree_arcs: Vec<EdgeIndex> = Vec::new();
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
                    state: zero(),
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
                    state: zero(),
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
                    state: zero(),
                },
            );
        }
        tree_arcs.push(edge);
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

    SPTree {
        in_base: tree_arcs,
        out_base: non_tree_arcs,
        pred: predecessors,
        thread: thread_id,
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
    sptree: &mut SPTree,
) -> Vec<NUM> {
    let mut pi: Vec<NUM> = vec![zero(); graph.node_count()];
    let edges: Vec<(u32, u32, f32)> = sptree
        .in_base
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

//Updating potential inspired by : NETWORK FLOWS Theory, Algorithms, and Applications (p.19)
//modified such that the difference from T1 and T2 is done with a BellmanFord algorithm
//detecting the single edge cost put to 1 that divide T1 and T2 in T
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

    let edges: Vec<(u32, u32, f32)> = sptree
        .in_base
        .iter()
        .filter(|&&x| x != entering_arc)
        .map(|&x| {
            let (i, j) = (
                graph.raw_edges()[x.index()].source(),
                graph.raw_edges()[x.index()].target(),
            );
            (i.index() as u32, j.index() as u32, 0f32)
        })
        .collect();
    let g = Graph::<(), f32, Undirected>::from_edges(edges);
    let path_cost: Vec<f32> = bellman_ford(&g, start).unwrap().distances;
    path_cost.iter().enumerate().for_each(|(index, x)| {
        if x == &0. {
            potential[index] += change
        } else {
        }
    });
    potential
}

fn __update_node_potentials<'a, NUM: CloneableNum>(
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

fn _update_node_potentials<'a, NUM: CloneableNum>(
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
    potentials_to_update
        .iter()
        .for_each(|&x| potential[x] += change);
    potential
}

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
    let mut current_node = Some(j);
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

/*
//checking if the specified edge is in forward direction according to a cycle
//needed in function compute_flowchange(...)
fn is_forward(edgeref: (NodeIndex, NodeIndex), cycle: &mut Vec<NodeIndex>) -> bool {
    distances_in_cycle(cycle).contains(&edgeref)
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
}*/

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
) {
    if entering_arc != leaving_arc {
        //Predecessors update
        let (i, j) = graph.edge_endpoints(entering_arc).unwrap();
        let (k, l) = graph.edge_endpoints(leaving_arc).unwrap();
        let path_to_change: &Vec<NodeIndex>;
        let path_to_root: &Vec<NodeIndex>;
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
        let mut current_node: NodeIndex = sptree.thread[path_to_change.last().unwrap().index()];
        let mut thread_from_change: Vec<NodeIndex> = vec![path_to_change[0]];
        while sptree.depth[current_node.index()]
            > sptree.depth[path_to_change.last().unwrap().index()]
        {
            thread_from_change.push(current_node);
            current_node = sptree.thread[current_node.index()];
        }
        sptree.depth[path_to_change[0].index()] = sptree.depth[path_to_root[0].index()] + 1;
        path_to_change
            .iter()
            .enumerate()
            .skip(1)
            .for_each(|(index, x)| {
                sptree.depth[x.index()] = sptree.depth[path_to_change[0].index()] + index
            });
        thread_from_change.iter().for_each(|&x| {
            sptree.depth[x.index()] = sptree.depth[sptree.pred[x.index()].unwrap().index()] + 1
        });

        
    }
    sptree.in_base.push(entering_arc);

    /*if index.is_none() {
        let id_l = sptree.l.iter().position(|&x| x == entering_arc);
        let id_u = sptree.u.iter().position(|&x| x == entering_arc);
        if id_u.is_none() {
            sptree.l.remove(id_l.unwrap());
        } else {
            sptree.u.remove(id_u.unwrap());
        }
    } else {
        if index.unwrap() < sptree.l.len() {
            sptree.l.remove(index.unwrap());
        } else {
            sptree.u.remove(index.unwrap() - sptree.l.len());
        }
    }*/

    let id_t = sptree.in_base.iter().position(|&x| x == leaving_arc);
    sptree.in_base.remove(id_t.unwrap());

    sptree.out_base.push(leaving_arc);
    //update thread_id
    if entering_arc != leaving_arc {
        let edges: Vec<(u32, u32, f32)> = sptree
            .in_base
            .iter()
            .map(|&x| {
                let (i, j) = (
                    graph.raw_edges()[x.index()].source(),
                    graph.raw_edges()[x.index()].target(),
                );
                (i.index() as u32, j.index() as u32, 0f32)
            })
            .collect();
        let mut dfs: Vec<NodeIndex> = Vec::new();
        let g = Graph::<(), _, Undirected>::from_edges(edges);
        dfs.push(NodeIndex::new(g.node_count() - 1));
        depth_first_search(&g, Some(NodeIndex::new(g.node_count() - 1)), |event| {
            if let DfsEvent::TreeEdge(_, v) = event {
                if v == NodeIndex::new(g.node_count() - 1) {
                    return Control::Break(());
                }
                dfs.push(v);
            }
            Control::Continue
        });
        dfs.push(NodeIndex::new(g.node_count() - 1));
        dfs.iter()
            .tuple_windows()
            .for_each(|(x, &y)| sptree.thread[x.index()] = y);
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
) -> Option<EdgeIndex> {
    let min_arc: Option<&EdgeIndex> = out_base.iter().min_by(|&x, &y| {
        (graph[*x].state * get_reduced_cost_edgeindex(graph, *x, potential))
            .partial_cmp(&(graph[*y].state * get_reduced_cost_edgeindex(graph, *y, potential)))
            .unwrap()
    });
    if min_arc.is_none() {
        None
    } else {
        if graph[*min_arc.unwrap()].state
            * get_reduced_cost_edgeindex(graph, *min_arc.unwrap(), potential)
            >= zero()
        {
            None
        } else {
            Some(*min_arc.unwrap())
        }
    }
}

//First eligible
fn _find_first_arc<NUM: CloneableNum>(
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    out_base: &Vec<EdgeIndex>,
    potential: &Vec<NUM>,
) -> Option<EdgeIndex> {
    let entering_arc = out_base.iter().find(|&arc| {
        graph[*arc].state * get_reduced_cost_edgeindex(graph, *arc, potential) < zero()
    });
    if entering_arc.is_none() {
        None
    } else {
        Some(*entering_arc.unwrap())
    }
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
    let mut potentials = compute_node_potentials(&mut graph, &mut tlu_solution);

    let block_size = (graph.edge_count() / 6) as usize;
    let mut _index: Option<usize> = None;
    let mut entering_arc: Option<EdgeIndex>;

    //ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();
    (_index, entering_arc) = _find_block_search(
        &mut graph,
        &tlu_solution.out_base,
        &mut potentials,
        Some(0),
        block_size,
    );
    //entering_arc = _find_best_arc(&mut graph, &tlu_solution.out_base, &mut potentials);

    let mut _iteration = 0;
    while entering_arc != None {
        let mut cycle = find_cycle_with_arc(&graph, &mut tlu_solution, entering_arc.unwrap());

        let leaving_arc = compute_flowchange(&mut graph, &mut cycle, entering_arc.unwrap());

        update_sptree(
            &mut graph,
            &mut tlu_solution,
            entering_arc.unwrap(),
            leaving_arc,
        );

        potentials = __update_node_potentials(
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

    //entering_arc = _find_best_arc(&mut graph, &tlu_solution.out_base, &mut potentials);

        entering_arc = _find_first_arc(&mut graph, &tlu_solution.out_base, &mut potentials);

        _iteration += 1;
    }
    println!("ITERATION {:?}", _iteration);
    let mut cost: NUM = zero();
    graph
        .edge_references()
        .for_each(|x| cost += x.weight().flow * x.weight().cost);
    println!("total cost = {:?}", cost);
    graph
}
