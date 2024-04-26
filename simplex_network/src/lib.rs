use itertools::Itertools;
use num_traits::identities::one;
use num_traits::identities::zero;
use num_traits::Num;
use petgraph::algo::bellman_ford;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::stable_graph::IndexType;
use petgraph::visit::IntoEdges;
use rayon::prelude::*;
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

    let source_id = graph
        .node_indices()
        .find(|&x| (graph.node_weight(x).unwrap() == &0)) // SINK_ID
        .unwrap();

    let mut big_value: NUM = zero();
    for _ in 0..100 {
        big_value += demand;
    }

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
fn compute_flowchange<'a, NUM: CloneableNum>(
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

    let cutting_depth: usize;
    if nodes.predecessor[k] == Some(l) {
        nodes.predecessor[k] = None;
        cutting_depth = nodes.depth[k]
    } else {
        nodes.predecessor[l] = None;
        cutting_depth = nodes.depth[l]
    }
    let mut path_from_i: Vec<usize>;
    let mut path_from_j: Vec<usize>;
    if branch == 1 {
        path_from_i = vec![i; nodes.depth[i] + 1 - cutting_depth];
        path_from_j = vec![j; nodes.depth[j] + 1];
    } else {
        // branch == 2
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
fn _find_best_arc<NUM: CloneableNum>(
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
) -> (Option<usize>, Option<usize>) {
    let mut min = zero();
    let mut entering_arc = None;
    let mut index = None;
    for i in 0..edges.out_base.len() {
        let arc = edges.out_base[i];
        let rc = edges.state[arc]
            * (edges.cost[arc] - nodes.potential[edges.source[arc]]
                + nodes.potential[edges.target[arc]]);
        //println!("testrc = {:?}", rc);
        if rc < min {
            min = rc;
            entering_arc = Some(arc);
            index = Some(i);
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
fn _par_find_best_arc<NUM: CloneableNum>(
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

//Block search
fn _find_block_search<NUM: CloneableNum>(
    out_base: &Vec<usize>,
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
    mut index: Option<usize>,
    block_size: usize,
    //return (arc_index, arc_id)
) -> (Option<usize>, Option<usize>) {
    let mut block_number = 0;
    let mut arc_index = None;
    if index.is_none() {
        index = Some(0);
    }
    while block_size * block_number <= out_base.len() {
        let (index_, entering_arc) = out_base
            .iter()
            .enumerate()
            .cycle()
            .skip(index.unwrap())
            .take(block_size)
            .min_by(|&x, &y| {
                (edges.state[*x.1] * get_reduced_cost_edgeindex(edges, nodes, *x.1))
                    .partial_cmp(
                        &(edges.state[*y.1] * get_reduced_cost_edgeindex(edges, nodes, *y.1)),
                    )
                    .unwrap()
            })
            .unwrap();
        let rc_entering_arc =
            edges.state[*entering_arc] * get_reduced_cost_edgeindex(edges, nodes, *entering_arc);
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

fn _par_block_search<NUM: CloneableNum>(
    out_base: &Vec<usize>,
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
    mut index: Option<usize>,
    block_size: usize,
    thread_nb: usize,
    //return (arc_index, arc_id)
) -> (Option<usize>, Option<usize>) {
    let mut _iteration = 0;

    let mut mins = vec![zero(); thread_nb];
    let mut arcs: Vec<(Option<usize>, Option<usize>)> = vec![(None, None); thread_nb];
    while (thread_nb * block_size * _iteration) <= (out_base.len() + thread_nb * block_size) {
        _iteration += 1;
        let chunks: &Vec<&[usize]> = &out_base[index.unwrap()..].chunks(block_size).collect();
        std::thread::scope(|s| {
            for (i, (rc_cand, candidate)) in std::iter::zip(&mut mins, &mut arcs).enumerate() {
                if i >= chunks.len() {
                    *rc_cand = zero();
                    *candidate = (None, None);
                    continue;
                };
                s.spawn(move || {
                    for (_index, &arc) in chunks[i].iter().enumerate() {
                        let rc = edges.state[arc]
                            * (edges.cost[arc] - nodes.potential[edges.source[arc]]
                                + nodes.potential[edges.target[arc]]);
                        if rc < *rc_cand {
                            *rc_cand = rc;
                            *candidate =
                                (Some(block_size * i + _index + index.unwrap()), Some(arc));
                        }
                    }
                });
            }
        });
        for (i, rc) in mins.iter().enumerate() {
            if *rc < zero::<NUM>() {
                return arcs[i];
            }
        }
        index = Some(index.unwrap() + block_size * thread_nb);
        if index.unwrap() >= out_base.len() {
            index = Some(0);
        }
    }
    (None, None)
}

//main algorithm function
pub fn min_cost<NUM: CloneableNum>(
    mut graph: DiGraph<u32, CustomEdgeIndices<NUM>>,
    demand: NUM,
) -> DiGraph<u32, CustomEdgeIndices<NUM>> {
    let (mut nodes, mut edges) = initialization::<NUM>(&mut graph, demand);
    println!("edge nb = {:?}", edges.out_base.len());
    let _block_size = (edges.out_base.len() / 75) as usize;
    let mut _index: Option<usize> = Some(0);
    let mut entering_arc: Option<usize>;
    let mut _iteration = 0;

    //try heuristic
    /*let heuristics_arcs:Vec<(usize, usize)> = edges.out_base.clone().into_iter().enumerate().filter(|(_, x)| 
        edges.state[*x] * get_reduced_cost_edgeindex(&edges, &nodes, *x) > zero()).collect();

    heuristics_arcs.iter().for_each(|(index, x)| 
        {let (leaving_arc, branch) =  compute_flowchange(&mut edges, &mut nodes, *x);
         update_sptree(&mut edges, &mut nodes, *x, leaving_arc, Some(*index), branch);
         update_node_potentials(&mut edges, &mut nodes, *x, leaving_arc);});*/


    (_index, entering_arc) = _find_best_arc(&edges, &nodes);
    /*let thread_nb = 8;
    ThreadPoolBuilder::new()
    .num_threads(thread_nb)
    .build_global()
    .unwrap();*/
    println!("now");
    while entering_arc.is_some() {
        let (leaving_arc, branch) =
            compute_flowchange(&mut edges, &mut nodes, entering_arc.unwrap());

        update_sptree(
            &mut edges,
            &mut nodes,
            entering_arc.unwrap(),
            leaving_arc,
            _index,
            branch,
        );

        update_node_potentials(&mut edges, &mut nodes, entering_arc.unwrap(), leaving_arc);

        (_index, entering_arc) =
        
        //_find_block_search(&edges.out_base, &edges, &nodes, _index, _block_size);
        //_par_block_search(&edges.out_base, &edges, &nodes, _index, _block_size, thread_nb);

        _find_best_arc(&edges, &nodes);
        //_par_find_best_arc(&edges, &nodes, thread_nb);

        //_find_first_arc(&edges, &mut nodes, _index);

        _iteration += 1;
    }
    println!("iterations : {:?}", _iteration);
    let mut cost: NUM = zero();
    edges
        .flow
        .iter()
        .enumerate()
        .for_each(|(index, &x)| cost += x * edges.cost[index]);
    graph.clone().edge_references().for_each(|x| {
        graph.edge_weight_mut(x.id()).expect("found").flow = edges.flow[x.id().index()]
    });
    println!("total cost = {:?}", cost);
    graph
}
