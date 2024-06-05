use itertools::Itertools;
use num_traits::identities::one;
use num_traits::identities::zero;
use num_traits::Num;
use num_traits::Signed;
use petgraph::algo::bellman_ford;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::stable_graph::IndexType;
use rayon::prelude::*;
//use std::time::SystemTime;
use petgraph::dot::*;
use rayon::ThreadPoolBuilder;

#[derive(Debug, Clone)]
struct Edges<NUM: CloneableNum> {
    out_base: Vec<usize>,
    source: Vec<usize>,
    target: Vec<usize>,
    flow: Vec<NUM>,
    cost: Vec<NUM>,
    capacity: Vec<NUM>,
    state: Vec<NUM>,
}

#[derive(Debug, Clone)]
struct Nodes<NUM: CloneableNum> {
    potential: Vec<NUM>,
    thread: Vec<usize>,
    revthread: Vec<usize>,
    predecessor: Vec<Option<usize>>,
    depth: Vec<usize>,
    edge_tree: Vec<usize>,
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
    + Signed
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
) -> (Nodes<NUM>, Edges<NUM>) {
    let max_node_id: u32 = (graph.node_count() - 1) as u32;

    let sink_id = graph
        .node_indices()
        .find(|&x| (graph.node_weight(x).unwrap() == &max_node_id)) // SINK_ID
        .unwrap();

    println!("sink_id = {:?}", sink_id);
    let source_id = graph
        .node_indices()
        .find(|&x| (graph.node_weight(x).unwrap() == &1048577)) // SINK_ID
        .unwrap();

    println!("source_id = {:?}", source_id);
    let mut big_value: NUM;
    big_value = num_traits::Bounded::max_value();
    big_value = big_value / (one::<NUM>() + one());
    big_value = big_value / (one::<NUM>() + one());
    big_value = big_value / (one::<NUM>() + one());
    big_value = big_value / (one::<NUM>() + one());
    big_value = big_value / (one::<NUM>() + one());
    big_value = big_value / (one::<NUM>() + one());
    big_value = big_value / (one::<NUM>() + one());
    big_value = big_value / (one::<NUM>() + one());
    big_value = big_value / (one::<NUM>() + one());
    big_value = big_value / (one::<NUM>() + one());
    big_value = big_value / (one::<NUM>() + one());
    big_value = big_value / (one::<NUM>() + one());
    big_value = big_value / (one::<NUM>() + one());
    big_value = big_value / (one::<NUM>() + one());
    big_value = big_value / (one::<NUM>() + one());
    big_value = big_value / (one::<NUM>() + one());
    big_value = big_value / (one::<NUM>() + one());

    println!("big_value = {:?}", big_value);
    let mut edge_tree: Vec<usize> = vec![0; graph.node_count() + 1];

    let artificial_root = graph.add_node(graph.node_count() as u32);
    for node in graph.node_indices() {
        if node == artificial_root {
            continue;
        }
        if node == sink_id {
            let arc = graph.add_edge(
                artificial_root,
                node,
                CustomEdgeIndices {
                    cost: big_value,
                    capacity: big_value,
                    flow: demand,
                },
            );
            edge_tree[node.index()] = arc.index();
        } else if node == source_id {
            // SOURCE_ID
            let arc = graph.add_edge(
                node,
                artificial_root,
                CustomEdgeIndices {
                    cost: big_value,
                    capacity: big_value,
                    flow: demand,
                },
            );
            edge_tree[node.index()] = arc.index();
        } else {
            let arc = graph.add_edge(
                node,
                artificial_root,
                CustomEdgeIndices {
                    cost: big_value,
                    capacity: big_value,
                    flow: zero(),
                },
            );
            edge_tree[node.index()] = arc.index();
        }
    }

    let potentials: Vec<NUM> = compute_node_potentials(graph);

    let mut thread_id: Vec<usize> = vec![0; graph.node_count()];
    for i in 0..thread_id.len() - 1 {
        thread_id[i] = i + 1;
    }
    let mut rev_thread_id: Vec<usize> = vec![graph.node_count() - 1; graph.node_count()];
    for i in 1..rev_thread_id.len() {
        rev_thread_id[i] = i - 1;
    }

    let mut predecessors: Vec<Option<usize>> =
        vec![Some(graph.node_count() - 1); graph.node_count()];
    let last = predecessors.len() - 1;
    predecessors[last] = None;

    let mut depths: Vec<usize> = vec![1; graph.node_count()];
    depths[last] = 0;

    let nodes: Nodes<NUM> = Nodes {
        potential: (potentials),
        thread: (thread_id),
        revthread: (rev_thread_id),
        predecessor: (predecessors),
        depth: (depths),
        edge_tree: (edge_tree),
    };

    let mut outbase: Vec<usize> = vec![];
    let mut source: Vec<usize> = vec![0; graph.edge_count()];
    let mut target: Vec<usize> = vec![0; graph.edge_count()];
    let mut flow: Vec<NUM> = vec![zero(); graph.edge_count()];
    let mut cost: Vec<NUM> = vec![zero(); graph.edge_count()];
    let mut capacity: Vec<NUM> = vec![zero(); graph.edge_count()];
    let mut state: Vec<NUM> = vec![zero(); graph.edge_count()];

    graph.edge_references().for_each(|x| {
        let id = x.id().index();
        source[id] = x.source().index();
        target[id] = x.target().index();
        flow[id] = x.weight().flow;
        cost[id] = x.weight().cost;
        capacity[id] = x.weight().capacity;
        state[id] = if flow[id] == capacity[id] {
            zero::<NUM>() - one()
        } else if flow[id] == zero() {
            one()
        } else {
            zero()
        };
        if !(source[id] == artificial_root.index() || target[id] == artificial_root.index()) {
            outbase.push(id);
        }
    });

    let edges: Edges<NUM> = Edges {
        out_base: (outbase),
        source: (source),
        target: (target),
        flow: (flow),
        cost: (cost),
        capacity: (capacity),
        state: (state),
    };
    (nodes, edges)
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

    let dist_pred: Vec<(&i32, &Option<NodeIndex>)> =
        distances.iter().zip(path.predecessors.iter()).collect();

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
    edges: &Edges<NUM>,
    nodes: &mut Nodes<NUM>,
    entering_arc: usize,
    leaving_arc: usize,
) {
    if entering_arc == leaving_arc {
        return;
    }
    let (k, l) = (edges.source[entering_arc], edges.target[entering_arc]);
    let mut change: NUM = zero();
    let start: usize;
    if nodes.predecessor[k] == Some(l) {
        change += get_reduced_cost_edgeindex(edges, nodes, entering_arc);
        start = k;
    } else {
        change -= get_reduced_cost_edgeindex(edges, nodes, entering_arc);
        start = l;
    }
    let mut current_node = nodes.thread[start];
    nodes.potential[start] += change;
    while nodes.depth[current_node] > nodes.depth[start] {
        nodes.potential[current_node] += change;
        current_node = nodes.thread[current_node];
    }
}

//computing delta the amount of flow we can augment through the cycle
//returning the leaving edge
fn __compute_flowchange<'a, NUM: CloneableNum>(
    edges: &mut Edges<NUM>,
    nodes: &mut Nodes<NUM>,
    entering_arc: usize,
    //(leaving_arc_id:usize, branch: 1 or 2)
) -> (usize, usize) {
    let (i, j) = (edges.source[entering_arc], edges.target[entering_arc]);
    let up_restricted = edges.flow[entering_arc] != zero();
    let (node1, node2) = if up_restricted { (i, j) } else { (j, i) };
    let (mut current1, mut current2) = (node1, node2);
    let (mut cycle_part1, mut cycle_part2) = (vec![], vec![]);

    //fill both vector part1 and part2 of the cycle
    while current1 != current2 {
        if nodes.depth[current1] < nodes.depth[current2] {
            cycle_part2.push(nodes.edge_tree[current2]);
            current2 = nodes.predecessor[current2].expect("found");
        } else {
            cycle_part1.push(nodes.edge_tree[current1]);
            current1 = nodes.predecessor[current1].expect("found");
        }
        if nodes.predecessor[current1].is_some() && nodes.predecessor[current2].is_some() {
            assert_eq!(
                nodes.depth[current1],
                nodes.depth[nodes.predecessor[current1].unwrap()] + 1
            );
            assert_eq!(
                nodes.depth[current2],
                nodes.depth[nodes.predecessor[current2].unwrap()] + 1
            );
        }
    }

    //fill vector of delta of arc in part 1 and 2
    let (mut delta_p1, mut delta_p2): (Vec<(usize, NUM, NUM)>, Vec<(usize, NUM, NUM)>) = (
        vec![(0, zero(), one()); cycle_part1.len()],
        vec![(0, zero(), one()); cycle_part2.len()],
    );
    cycle_part1.iter().enumerate().for_each(|(index, &x)| {
        let pred = nodes.predecessor[edges.source[x]];
        delta_p1[index] = if pred.is_some() && edges.target[x] == pred.unwrap() {
            (index, edges.capacity[x] - edges.flow[x], one())
        } else {
            (index, edges.flow[x], zero::<NUM>() - one())
        }
    });
    cycle_part2.iter().enumerate().for_each(|(index, &x)| {
        let pred = nodes.predecessor[edges.target[x]];
        delta_p2[index] = if pred.is_some() && edges.source[x] == pred.unwrap() {
            (index, edges.capacity[x] - edges.flow[x], one())
        } else {
            (index, edges.flow[x], zero::<NUM>() - one())
        }
    });
    let min_d1 = delta_p1
        .iter()
        .min_set_by(|(_, delta1, _), (_, delta2, _)| delta1.partial_cmp(&delta2).unwrap());
    let min_d2 = delta_p2
        .iter()
        .min_set_by(|(_, delta1, _), (_, delta2, _)| delta1.partial_cmp(&delta2).unwrap());

    let leaving_p1 = min_d1
        .into_iter()
        .max_by(|(pos1, _, _), (pos2, _, _)| pos1.cmp(&pos2));
    let leaving_p2 = min_d2
        .into_iter()
        .min_by(|(pos1, _, _), (pos2, _, _)| pos1.cmp(&pos2));

    let leaving_set: usize;
    let leaving_arc: usize;

    let min_p1_p2 = if leaving_p1.is_none() {
        leaving_set = 2;
        leaving_p2.unwrap()
    } else if leaving_p2.is_none() {
        leaving_set = 1;
        leaving_p1.unwrap()
    } else if leaving_p1.unwrap().1 >= leaving_p2.unwrap().1 {
        leaving_set = 2;
        leaving_p2.unwrap()
    } else {
        leaving_set = 1;
        leaving_p1.unwrap()
    };

    let delta_entering = if up_restricted {
        edges.flow[entering_arc]
    } else {
        edges.capacity[entering_arc]
    };

    let mut final_delta = min_p1_p2.1;
    if min_p1_p2.1 > delta_entering {
        leaving_arc = entering_arc;
        final_delta = delta_entering;
    } else {
        if leaving_set == 1 {
            leaving_arc = cycle_part1[min_p1_p2.0];
        } else {
            // leaving_set == 2
            leaving_arc = cycle_part2[min_p1_p2.0];
        }
    }

    //Flow update
    if final_delta != zero() {
        cycle_part1
            .iter()
            .zip(delta_p1.iter())
            .for_each(|(&edge_cycle, (_, _, dir))| {
                edges.flow[edge_cycle] += *dir * final_delta;
                if edges.flow[edge_cycle] == edges.capacity[edge_cycle] {
                    edges.state[edge_cycle] = zero::<NUM>() - one()
                }
                if edges.flow[edge_cycle] == zero() {
                    edges.state[edge_cycle] = one()
                };
            });
        cycle_part2
            .iter()
            .zip(delta_p2.iter())
            .for_each(|(&edge_cycle, (_, _, dir))| {
                edges.flow[edge_cycle] += *dir * final_delta;
                if edges.flow[edge_cycle] == edges.capacity[edge_cycle] {
                    edges.state[edge_cycle] = zero::<NUM>() - one()
                }
                if edges.flow[edge_cycle] == zero() {
                    edges.state[edge_cycle] = one()
                };
            });
        if up_restricted {
            edges.flow[entering_arc] -= final_delta;
        } else {
            edges.flow[entering_arc] += final_delta;
        }
    }
    if edges.flow[entering_arc] == edges.capacity[entering_arc] {
        edges.state[entering_arc] = zero::<NUM>() - one()
    }
    if edges.flow[entering_arc] == zero() {
        edges.state[entering_arc] = one()
    }

    let branch = if up_restricted {
        if leaving_set == 1 {
            2
        } else {
            1
        }
    } else {
        leaving_set
    };

    (leaving_arc, branch)
}

fn _compute_flowchange<'a, NUM: CloneableNum>(
    edges: &mut Edges<NUM>,
    nodes: &mut Nodes<NUM>,
    entering_arc: usize,
) -> (usize, usize) {
    let (i, j) = (edges.source[entering_arc], edges.target[entering_arc]);
    let up_restricted = edges.flow[entering_arc] != zero();

    let mut current_i = i;
    let mut current_j = j;

    let mut min_delta = if up_restricted {
        (edges.flow[entering_arc], entering_arc)
    } else {
        (edges.capacity[entering_arc], entering_arc)
    };

    let mut min_delta_i = min_delta;
    let mut min_delta_j = min_delta;

    while current_j != current_i {
        let arc_i = nodes.edge_tree[current_i];
        let arc_j = nodes.edge_tree[current_j];
        let delta: (NUM, usize);
        if nodes.depth[current_i] < nodes.depth[current_j] {
            if up_restricted {
                if current_j == edges.target[arc_j] {
                    delta = (edges.capacity[arc_j] - edges.flow[arc_j], arc_j);
                } else {
                    delta = (edges.flow[arc_j], arc_j);
                }
                if delta.0 < min_delta_j.0 {
                    min_delta_j = delta
                };
            } else {
                if current_j == edges.source[arc_j] {
                    delta = (edges.capacity[arc_j] - edges.flow[arc_j], arc_j);
                } else {
                    delta = (edges.flow[nodes.edge_tree[current_j]], arc_j);
                }
                if delta.0 <= min_delta_j.0 {
                    min_delta_j = delta
                };
            }
            current_j = nodes.predecessor[current_j].unwrap();
        } else {
            if up_restricted {
                if current_i == edges.source[arc_i] {
                    delta = (edges.capacity[arc_i] - edges.flow[arc_i], arc_i);
                } else {
                    delta = (edges.flow[arc_i], arc_i);
                }
                if delta.0 <= min_delta_i.0 {
                    min_delta_i = delta
                };
            } else {
                if current_i == edges.target[arc_i] {
                    delta = (edges.capacity[arc_i] - edges.flow[arc_i], arc_i);
                } else {
                    delta = (edges.flow[arc_i], arc_i);
                }
                if delta.0 < min_delta_i.0 {
                    min_delta_i = delta
                };
            }
            current_i = nodes.predecessor[current_i].unwrap();
        }
    }

    let mut branch: usize = 0;
    if min_delta.0 > min_delta_i.0 {
        min_delta = min_delta_i;
        branch = 1;
    }
    if min_delta.0 > min_delta_j.0 {
        min_delta = min_delta_j;
        branch = 2;
    }
    if min_delta_j.0 == min_delta_i.0 {
        min_delta = if up_restricted {
            branch = 2;
            min_delta_j
        } else {
            branch = 1;
            min_delta_i
        }
    }

    if min_delta.0 != zero() {
        current_i = i;
        current_j = j;
        if up_restricted {
            edges.flow[entering_arc] -= min_delta.0;
        } else {
            edges.flow[entering_arc] += min_delta.0;
        }
        if edges.flow[entering_arc] == zero() {
            edges.state[entering_arc] = one();
        } else if edges.flow[entering_arc] == edges.capacity[entering_arc] {
            edges.state[entering_arc] = zero::<NUM>() - one();
        }
        while current_j != current_i {
            let arc_i = nodes.edge_tree[current_i];
            let arc_j = nodes.edge_tree[current_j];
            if nodes.depth[current_i] < nodes.depth[current_j] {
                if up_restricted {
                    if current_j == edges.target[arc_j] {
                        edges.flow[arc_j] += min_delta.0;
                    } else {
                        edges.flow[arc_j] -= min_delta.0;
                    }
                } else {
                    if current_j == edges.source[arc_j] {
                        edges.flow[arc_j] += min_delta.0;
                    } else {
                        edges.flow[arc_j] -= min_delta.0;
                    }
                }
                if edges.flow[arc_j] == zero() {
                    edges.state[arc_j] = one();
                } else if edges.flow[arc_j] == edges.capacity[arc_j] {
                    edges.state[arc_j] = zero::<NUM>() - one();
                }
                current_j = nodes.predecessor[current_j].unwrap();
            } else {
                if up_restricted {
                    if current_i == edges.source[arc_i] {
                        edges.flow[arc_i] += min_delta.0;
                    } else {
                        edges.flow[arc_i] -= min_delta.0;
                    }
                } else {
                    if current_i == edges.target[arc_i] {
                        edges.flow[arc_i] += min_delta.0;
                    } else {
                        edges.flow[arc_i] -= min_delta.0;
                    }
                }
                if edges.flow[arc_i] == zero() {
                    edges.state[arc_i] = one();
                } else if edges.flow[arc_i] == edges.capacity[arc_i] {
                    edges.state[arc_i] = zero::<NUM>() - one();
                }
                current_i = nodes.predecessor[current_i].unwrap();
            }
        }
    }

    (min_delta.1, branch)
}

/* Update sptree structure according to entering arc and leaving arc,
* reorder predecessors to keep tree coherent tree structure from one basis
* to another.
*/
fn update_sptree<NUM: CloneableNum>(
    edges: &mut Edges<NUM>,
    nodes: &mut Nodes<NUM>,
    entering_arc: usize,
    leaving_arc: usize,
    position: Option<usize>,
    branch: usize,
) {
    if entering_arc == leaving_arc {
        return;
    }
    //useful structure init
    let node_nb = nodes.potential.len();
    let (i, j) = (edges.source[entering_arc], edges.target[entering_arc]);
    let (k, l) = (edges.source[leaving_arc], edges.target[leaving_arc]);

    let mut path_to_change: &Vec<usize>;
    let mut path_to_root: &Vec<usize>;

    //used to get length of vector path_from_*
    let cutting_depth: usize;
    if nodes.predecessor[k] == Some(l) {
        nodes.predecessor[k] = None;
        cutting_depth = nodes.depth[k]
    } else {
        nodes.predecessor[l] = None;
        cutting_depth = nodes.depth[l]
    }

    //vectors contain id of arcs from i/j to root or removed arc
    let mut path_from_i: Vec<usize>;
    let mut path_from_j: Vec<usize>;
    if branch == 1 {
        path_from_i = vec![i; nodes.depth[i] + 1 - cutting_depth];
        path_from_j = vec![j; nodes.depth[j] + 1];
    } else {
        // branch == 1
        path_from_i = vec![i; nodes.depth[i] + 1];
        path_from_j = vec![j; nodes.depth[j] + 1 - cutting_depth];
    }

    let mut current_node: Option<usize> = Some(i);
    for index in 0..path_from_i.len() {
        path_from_i[index] = current_node.unwrap();
        current_node = nodes.predecessor[current_node.unwrap()];
    }
    current_node = Some(j);
    for index in 0..path_from_j.len() {
        path_from_j[index] = current_node.unwrap();
        current_node = nodes.predecessor[current_node.unwrap()];
    }

    if path_from_i[path_from_i.len() - 1] != node_nb - 1 {
        path_to_change = &path_from_i;
        path_to_root = &path_from_j;
    } else {
        path_to_change = &path_from_j;
        path_to_root = &path_from_i;
    }

    // update thread_id
    let mut current_node = nodes.thread[path_to_change.last().unwrap().index()];
    let mut block_parcour = vec![*path_to_change.last().unwrap()];
    while nodes.depth[current_node.index()] > nodes.depth[path_to_change.last().unwrap().index()] {
        block_parcour.push(current_node);
        current_node = nodes.thread[current_node.index()];
    }

    let mut dirty_rev_thread: Vec<usize> = vec![];
    let nodeid_to_block = nodes.revthread[block_parcour[0].index()];
    nodes.thread[nodeid_to_block.index()] = nodes.thread[block_parcour.last().unwrap().index()];
    dirty_rev_thread.push(nodeid_to_block);

    path_to_change
        .iter()
        .take(path_to_change.len() - 1)
        .for_each(|&x| {
            let mut current = nodes.thread[x.index()];
            let mut old_last = x;
            while nodes.depth[current.index()] > nodes.depth[x.index()] {
                old_last = current;
                current = nodes.thread[current.index()];
            }
            nodes.thread[nodes.revthread[x.index()].index()] = current;
            dirty_rev_thread.push(nodes.revthread[x.index()]);
            nodes.thread[old_last.index()] = nodes.predecessor[x.index()].unwrap();
            dirty_rev_thread.push(old_last);
        });

    let mut current = nodes.thread[path_to_change.last().unwrap().index()];
    let mut old = *path_to_change.last().unwrap();
    while nodes.depth[current.index()] > nodes.depth[path_to_change.last().unwrap().index()] {
        old = current;
        current = nodes.thread[current.index()];
    }
    nodes.thread[old.index()] = nodes.thread[path_to_root[0].index()];
    dirty_rev_thread.push(old);
    nodes.thread[path_to_root[0].index()] = path_to_change[0];
    dirty_rev_thread.push(path_to_root[0]);

    dirty_rev_thread
        .into_iter()
        .for_each(|new_rev| nodes.revthread[nodes.thread[new_rev.index()].index()] = new_rev);

    //Predecessors update + edge_tree
    if path_from_i[path_from_i.len() - 1] != node_nb - 1 {
        nodes.predecessor[i.index()] = Some(j);
        path_from_i
            .iter()
            .enumerate()
            .skip(1)
            .for_each(|(index, &x)| nodes.predecessor[x.index()] = Some(path_from_i[index - 1]));
        path_to_change = &path_from_i;
        path_to_root = &path_from_j;
    } else {
        nodes.predecessor[j.index()] = Some(i);
        path_from_j
            .iter()
            .enumerate()
            .skip(1)
            .for_each(|(index, &x)| nodes.predecessor[x.index()] = Some(path_from_j[index - 1]));
        path_to_root = &path_from_i;
        path_to_change = &path_from_j;
    }

    let temp: Vec<usize> = path_to_change.iter().map(|&x| nodes.edge_tree[x]).collect();
    nodes.edge_tree[path_to_change[0]] = entering_arc;
    path_to_change
        .iter()
        .enumerate()
        .skip(1)
        .for_each(|(index, &x)| {
            nodes.edge_tree[x] = temp[index - 1];
        });

    //update depth
    nodes.depth[path_to_change[0].index()] = nodes.depth[path_to_root[0].index()] + 1;
    path_to_change.iter().skip(1).for_each(|x| {
        nodes.depth[x.index()] = nodes.depth[nodes.predecessor[x.index()].unwrap().index()] + 1
    });
    block_parcour.iter().for_each(|&x| {
        nodes.depth[x.index()] = nodes.depth[nodes.predecessor[x.index()].unwrap().index()] + 1
    });

    edges.out_base[position.unwrap()] = leaving_arc;
}

///////////////////////
///////////////////////

fn get_reduced_cost_edgeindex<NUM: CloneableNum>(
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
    edgeindex: usize,
) -> NUM {
    edges.cost[edgeindex] - nodes.potential[edges.source[edgeindex]]
        + nodes.potential[edges.target[edgeindex]]
}

///////////////////////
///// Pivot rules /////
///////////////////////

//Best Eligible arc
fn _best_arc<NUM: CloneableNum>(
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
) -> (Option<usize>, Option<usize>) {
    let mut min = zero();
    let mut entering_arc = None;
    let mut index = None;
    for i in 0..edges.out_base.len() {
        let arc = unsafe { *edges.out_base.get_unchecked(i) };
        let rcplus = unsafe {
            *edges.cost.get_unchecked(arc)
                + *nodes
                    .potential
                    .get_unchecked(*edges.target.get_unchecked(arc))
        };
        let rcminus = unsafe {
            *nodes
                .potential
                .get_unchecked(*edges.source.get_unchecked(arc))
        };
        let s: NUM = unsafe { *edges.state.get_unchecked(arc) };
        if (rcplus < rcminus) ^ (s.is_negative()) {
            let rc = s * (rcplus - rcminus);
            if rc < min {
                min = rc;
                entering_arc = Some(arc);
                index = Some(i);
            }
        } else {
            continue;
        }
    }
    (index, entering_arc)

    /*let (index, candidate) = edges.out_base.iter().enumerate().min_by(|(_, &x), (_, &y)|
        {let rc_x = edges.state[x] * get_reduced_cost_edgeindex(edges, nodes, x);
         let rc_y = edges.state[y] * get_reduced_cost_edgeindex(edges, nodes, y);
         rc_x.partial_cmp(&rc_y).unwrap()}).expect("found");
    let rc_cand = edges.state[*candidate] * get_reduced_cost_edgeindex(edges, nodes, *candidate);
    if rc_cand >= zero() {
        return (None, None);
    }
    (Some(index), Some(*candidate))*/
}

//Parallel Best Eligible arc
fn _par_best_arc_v1<NUM: CloneableNum>(
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
    thread_nb: usize,
) -> (Option<usize>, Option<usize>) {
    let mut mins = vec![zero(); thread_nb];
    let mut arcs: Vec<(Option<usize>, Option<usize>)> = vec![(None, None); thread_nb];
    let chunk_size: usize = (edges.out_base.len() / thread_nb) + 1;
    let chunks: &Vec<&[usize]> = &edges.out_base.chunks(chunk_size).collect();
    std::thread::scope(|s| {
        for (i, (rc_cand, candidate)) in std::iter::zip(&mut mins, &mut arcs).enumerate() {
            s.spawn(move || {
                for (index, &arc) in chunks[i].iter().enumerate() {
                    let rc = edges.state[arc]
                        * (edges.cost[arc] - nodes.potential[edges.source[arc]]
                            + nodes.potential[edges.target[arc]]);
                    //println!("testrc = {:?}", rc);
                    if rc < *rc_cand {
                        *rc_cand = rc;
                        *candidate = (Some(chunk_size * i + index), Some(arc));
                    }
                }
            });
        }
    });
    let mut min = mins[0];
    let mut id = 0;
    for (index, rc) in mins.iter().enumerate() {
        if rc < &min {
            min = *rc;
            id = index;
        }
    }

    if min != zero() {
        return arcs[id];
    }
    (None, None)
}

//First eligible
fn _find_first_arc<NUM: CloneableNum>(
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
    mut index: Option<usize>,
) -> (Option<usize>, Option<usize>) {
    if index.is_none() {
        index = Some(0);
    }
    for i in index.unwrap() + 1..edges.out_base.len() {
        let arc = edges.out_base[i];
        let rc = edges.state[arc]
            * (edges.cost[arc] - nodes.potential[edges.source[arc]]
                + nodes.potential[edges.target[arc]]);
        if rc < zero() {
            return (Some(i), Some(arc));
        }
    }
    for i in 0..index.unwrap() + 1 {
        let arc = edges.out_base[i];
        let rc = edges.state[arc]
            * (edges.cost[arc] - nodes.potential[edges.source[arc]]
                + nodes.potential[edges.target[arc]]);
        if rc < zero() {
            return (Some(i), Some(arc));
        }
    }
    (None, None)
}

///////////////////////////////
/// SEQUENTIAL BLOCK SEARCH ///
///////////////////////////////

//Block_search basic_version
fn _block_search_v1<NUM: CloneableNum>(
    out_base: &Vec<usize>,
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
    mut index: usize,
    block_size: usize,
    //return (arc_index, arc_id)
) -> (Option<usize>, Option<usize>) {
    let mut min: NUM = zero();
    let mut entering_arc: Option<usize> = None;
    let mut nb_block_checked = 0;
    //let start = SystemTime::now();
    while nb_block_checked <= (out_base.len() / block_size) + 1 {
        nb_block_checked += 1;
        for i in index..(index + std::cmp::min(block_size, out_base.len() - index)) {
            let arc = unsafe { *edges.out_base.get_unchecked(i) };
            let rcplus = unsafe {
                *edges.cost.get_unchecked(arc)
                    + *nodes
                        .potential
                        .get_unchecked(*edges.target.get_unchecked(arc))
            };
            let rcminus = unsafe {
                *nodes
                    .potential
                    .get_unchecked(*edges.source.get_unchecked(arc))
            };
            let s: NUM = unsafe { *edges.state.get_unchecked(arc) };
            if (rcplus < rcminus) ^ (s.is_negative()) {
                let rc = s * (rcplus - rcminus);
                if rc < min {
                    min = rc;
                    entering_arc = Some(arc);
                    index = i;
                }
            } else {
                continue;
            }
        }
        if entering_arc.is_some() {
            /*match start.elapsed() {
                Ok(elapsed) => {
                    println!("{:?}", elapsed.as_nanos());
                }
                Err(e) => {
                    println!("Error: {e:?}");
                }
            }*/
            /*rc = edges.state[entering_arc.unwrap()]
                * (edges.cost[entering_arc.unwrap()]
                    - nodes.potential[edges.source[entering_arc.unwrap()]]
                    + nodes.potential[edges.target[entering_arc.unwrap()]]);
            println!("{:?}", rc);*/
            return (Some(index), entering_arc);
        }
        index = index + block_size;
        if index > out_base.len() {
            index = 0;
        }
    }
    (None, None)
}

fn _block_search_v2<NUM: CloneableNum>(
    out_base: &Vec<usize>,
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
    mut index: usize,
    block_size: usize,
    //return (arc_index, arc_id)
) -> (Option<usize>, Option<usize>) {
    let (mut cand_index, mut cand_id): (Option<usize>, Option<usize>) = (None, None);
    let mut nb_block_checked = 0;
    while nb_block_checked < (out_base.len() / block_size) + 1 {
        nb_block_checked += 1;
        let Some((arc_index, arc_id, rc)) = out_base
            .iter()
            .enumerate()
            .cycle()
            .skip(index)
            .take(block_size)
            .map(|(pos, &arc)| {
                (pos, arc, unsafe {
                    *edges.state.get_unchecked(arc)
                        * (*edges.cost.get_unchecked(arc)
                            - *nodes
                                .potential
                                .get_unchecked(*edges.source.get_unchecked(arc))
                            + *nodes
                                .potential
                                .get_unchecked(*edges.target.get_unchecked(arc)))
                })
            })
            .min_by(|(_, _, rca), (_, _, rcb)| rca.partial_cmp(&rcb).unwrap())
        else {
            unreachable!()
        };
        if rc < zero() {
            let new_index = arc_index;
            (cand_index, cand_id) = (Some(new_index), Some(arc_id));
        } else {
            index = if index + block_size >= out_base.len() {
                0
            } else {
                index + block_size
            };
        }
    }
    (cand_index, cand_id)
}

/*
unsafe fn _best_eligible_in_block<NUM: CloneableNum>(
    block: &Vec<usize>,
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
) -> Option<(usize, usize)> {
    let (best_index, rc) = block
        .iter()
        .enumerate()
        .map(|(index, &arc)| {
            (
                index,
                *edges.state.get_unchecked(arc)
                    * (*edges.cost.get_unchecked(arc)
                        - *nodes
                            .potential
                            .get_unchecked(*edges.source.get_unchecked(arc))
                        + *nodes
                            .potential
                            .get_unchecked(*edges.target.get_unchecked(arc))),
            )
        })
        .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
        .expect("found");
    if rc >= zero() {
        None
    } else {
        Some((best_index, block[best_index]))
    }
}*/

/////////////////////////////
/// PARALLEL BLOCK SEARCH ///
/////////////////////////////

//parallel iterator inside of the block perf are fine
fn _parallel_block_search_v1<NUM: CloneableNum>(
    out_base: &Vec<usize>,
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
    mut index: usize,
    block_size: usize,
    //return (arc_index, arc_id)
) -> (Option<usize>, Option<usize>) {
    let mut entering_arc: Option<(usize, usize, NUM)>;
    let mut nb_block_checked = 0;

    //let start = SystemTime::now();
    while nb_block_checked <= (out_base.len() / block_size) + 1 {
        nb_block_checked += 1;

        entering_arc = out_base[index..(index + std::cmp::min(block_size, out_base.len() - index))]
            .par_iter()
            .enumerate()
            .map(|(pos, &arc)| {
                let rcplus = unsafe {
                    *edges.cost.get_unchecked(arc)
                        + *nodes
                            .potential
                            .get_unchecked(*edges.target.get_unchecked(arc))
                };
                let rcminus = unsafe {
                    *nodes
                        .potential
                        .get_unchecked(*edges.source.get_unchecked(arc))
                };
                let s: NUM = unsafe { *edges.state.get_unchecked(arc) };
                (pos, arc, (s * (rcplus - rcminus)))
            })
            .min_by(|(_, _, rc1), (_, _, rc2)| (rc1).partial_cmp(&(rc2)).unwrap());
        //.filter(|(_, _, rc)| *rc < zero())

        if entering_arc.is_some() && entering_arc.unwrap().2 < zero() {
            /*match start.elapsed() {
                Ok(elapsed) => {
                    println!("{:?}", elapsed.as_nanos());
                }
                Err(e) => {
                    println!("Error: {e:?}");
                }
            }*/
            return (
                Some(index + entering_arc.unwrap().0),
                Some(entering_arc.unwrap().1),
            );
        }
        index = index + block_size;
        if index > out_base.len() {
            index = 0;
        }
    }
    (None, None)
}

//use of iterator to generate block
unsafe fn _parallel_block_search_v2<NUM: CloneableNum>(
    out_base: &Vec<usize>,
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
    mut index: usize,
    block_size: usize,
) -> (Option<usize>, Option<usize>) {
    let mut block_number = 0;
    let mut arc_index = None;

    while block_number * block_size <= out_base.len() {
        let (index_, rc_entering_arc) = out_base
            .par_iter()
            .enumerate()
            .skip(index)
            .chain(out_base[..index].par_iter().enumerate())
            .take(block_size)
            .map(|(pos, &arc)| {
                (
                    pos,
                    *edges.state.get_unchecked(arc)
                        * (*edges.cost.get_unchecked(arc)
                            - *nodes
                                .potential
                                .get_unchecked(*edges.source.get_unchecked(arc))
                            + *nodes
                                .potential
                                .get_unchecked(*edges.target.get_unchecked(arc))),
                )
            })
            .min_by(|&x, &y| x.1.partial_cmp(&y.1).unwrap())
            .unwrap();
        if rc_entering_arc >= zero::<NUM>() {
            arc_index = None
        } else {
            arc_index = Some(index_)
        }

        if arc_index.is_none() {
            index += block_size;
            block_number += 1;
            if index > out_base.len() {
                index = 0;
            }
        } else {
            break;
        }
    }

    if arc_index.is_none() {
        (None, None)
    } else {
        (arc_index, Some(out_base[arc_index.unwrap()]))
    }
}

//Same search than V2 but use of filter -> Bad performance
fn _parallel_block_search_v3<NUM: CloneableNum>(
    out_base: &Vec<usize>,
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
    mut index: usize,
    block_size: usize,
    //return (arc_index, arc_id)
) -> (Option<usize>, Option<usize>) {
    let (mut cand_index, mut cand_id): (Option<usize>, Option<usize>) = (None, None);
    let mut nb_block_checked = 0;
    while nb_block_checked < (out_base.len() / block_size) + 1 {
        nb_block_checked += 1;
        let candidate:Option<(usize, usize, NUM)> = //(arc_index, arc_id, rc)
            out_base[index..]
            .par_iter()
            .chain(&out_base[..index])
            .enumerate()
            .take(block_size)
            .map(|(pos, &arc)| {
                (pos, arc, unsafe {
                    *edges.state.get_unchecked(arc)
                        * (*edges.cost.get_unchecked(arc)
                            - *nodes
                                .potential
                                .get_unchecked(*edges.source.get_unchecked(arc))
                            + *nodes
                                .potential
                                .get_unchecked(*edges.target.get_unchecked(arc)))
                })
            })
            .filter(|(_,_,rc)| *rc < zero())
            .min_by(|(_, _, rca), (_, _, rcb)| rca.partial_cmp(&rcb).unwrap());

        if candidate.is_some() {
            let new_index = index + candidate.unwrap().0;
            if new_index >= out_base.len() {
                (cand_index, cand_id) =
                    (Some(new_index - out_base.len()), Some(candidate.unwrap().1));
                break;
            } else {
                (cand_index, cand_id) = (Some(new_index), Some(candidate.unwrap().1));
                break;
            }
        } else {
            index = if index + block_size >= out_base.len() {
                0
            } else {
                index + block_size
            };
        }
    }
    (cand_index, cand_id)
}

/*
fn _find_start<NUM: CloneableNum>(
    out_base: &Vec<usize>,
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
    index: usize,
) -> Option<usize> {
    out_base[index..]
        .iter()
        .enumerate()
        .find_map(|(pos, &arc)| {
            let rcplus = unsafe {
                *edges.cost.get_unchecked(arc)
                    + *nodes
                        .potential
                        .get_unchecked(*edges.target.get_unchecked(arc))
            };
            let rcminus = unsafe {
                *nodes
                    .potential
                    .get_unchecked(*edges.source.get_unchecked(arc))
            };
            let s: NUM = unsafe { *edges.state.get_unchecked(arc) };
            ((rcplus < rcminus) ^ (s.is_negative())).then_some(index + pos)
        })
        .or_else(|| {
            out_base[..index]
                .iter()
                .enumerate()
                .find_map(|(pos, &arc)| {
                    let rcplus = unsafe {
                        *edges.cost.get_unchecked(arc)
                            + *nodes
                                .potential
                                .get_unchecked(*edges.target.get_unchecked(arc))
                    };
                    let rcminus = unsafe {
                        *nodes
                            .potential
                            .get_unchecked(*edges.source.get_unchecked(arc))
                    };
                    let s: NUM = unsafe { *edges.state.get_unchecked(arc) };

                    ((rcplus < rcminus) ^ (s.is_negative())).then_some(pos)
                })
        })
}*/

//main algorithm function
pub fn min_cost<NUM: CloneableNum>(
    mut graph: DiGraph<u32, CustomEdgeIndices<NUM>>,
    demand: NUM,
) -> DiGraph<u32, CustomEdgeIndices<NUM>> {
    let (mut nodes, mut edges) = initialization::<NUM>(&mut graph, demand);
    let _block_size = (edges.out_base.len() / 10384) as usize;
    let mut _index: Option<usize> = Some(0);
    let mut entering_arc: Option<usize>;
    let mut _iteration = 0;
    (_index, entering_arc) = _best_arc(&edges, &nodes);
    let _thread_nb = 4;
    ThreadPoolBuilder::new()
        .num_threads(_thread_nb)
        .build_global()
        .unwrap();
    while entering_arc.is_some() {
        let (leaving_arc, branch) =
            _compute_flowchange(&mut edges, &mut nodes, entering_arc.unwrap());

        //println!("entering_arc {:?}", (edges.source[entering_arc.unwrap()], edges.target[entering_arc.unwrap()]));
        update_sptree(
            &mut edges,
            &mut nodes,
            entering_arc.unwrap(),
            leaving_arc,
            _index,
            branch,
        );

        update_node_potentials(&mut edges, &mut nodes, entering_arc.unwrap(), leaving_arc);
        (_index, entering_arc) = _block_search_v1(
            &edges.out_base,
            &edges,
            &nodes,
            _index.expect(""),
            _block_size,
        );

        /**/

        /*_parallel_block_search_v1(
            &edges.out_base,
            &edges,
            &nodes,
            _index.unwrap(),
            _block_size,
        );*/

        //_best_arc(&edges, &nodes);

        //_par_best_arc_v1(&edges, &nodes, _thread_nb);

        //println!("leaving arc {:?}", (edges.source[leaving_arc], edges.target[leaving_arc]));

        //_find_first_arc(&edges, &mut nodes, _index);
        _iteration += 1;
    }
    println!("iterations : {:?}", _iteration);
    //println!("{:?}", Dot::new(&graph));
    graph.remove_node(NodeIndex::new(graph.node_count() - 1));
    let mut cost: NUM = zero();
    let mut total_flow: NUM = zero();
    graph.clone().edge_references().for_each(|x| {
        graph.edge_weight_mut(x.id()).expect("found").flow = edges.flow[x.id().index()];
        cost += edges.flow[x.id().index()] * edges.cost[x.id().index()];
    });
    let sink_id = graph.node_count() - 1;
    graph
        .edges_directed(NodeIndex::new(sink_id), Incoming)
        .for_each(|x| total_flow += edges.flow[x.id().index()]);
    println!("total flow = {:?}, with cost = {:?}", total_flow, cost);
    graph
}
