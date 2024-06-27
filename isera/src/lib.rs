use crate::basetypes::*;

use itertools::Itertools;
use num_traits::identities::one;
use num_traits::identities::zero;
use petgraph::algo::bellman_ford;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::stable_graph::IndexType;
use pivotrules::*;
use rayon::ThreadPoolBuilder;
use std::any::*;

pub mod basetypes;
pub mod pivotrules;

//Initialization with artificial root
fn initialization<'a, NUM: CloneableNum + 'static>(
    graph: &'a mut DiGraph<u32, CustomEdgeIndices<NUM>>,
    sources: Vec<(usize, NUM)>, //(node_id, demand)
    sinks: Vec<(usize, NUM)>,   //(node_id, demand)
) -> (Nodes<NUM>, Edges<NUM>, GraphState<NUM>) {
    //println!("source_id = {:?}", sources);
    //println!("sink_id = {:?}", sinks);
    let mut total_supply_sources: NUM = zero();
    let mut total_supply_sinks: NUM = zero();
    sources
        .iter()
        .for_each(|(_, demand)| total_supply_sources += *demand);
    sinks
        .iter()
        .for_each(|(_, demand)| total_supply_sinks += *demand);

    let mut big_value: NUM;
    if TypeId::of::<NUM>() == TypeId::of::<f32>() || TypeId::of::<NUM>() == TypeId::of::<f64>() {
        big_value = one::<NUM>() + one();
        for _ in 1..52 {
            big_value = big_value * (one::<NUM>() + one());
        }
    } else {
        big_value = num_traits::Bounded::max_value();
        big_value = big_value / ((one::<NUM>() + one()) + (one::<NUM>() + one()));
    }

    //big_value --arbitrary big value
    let mut edge_tree: Vec<usize> = vec![0; graph.node_count() + 1];

    let artificial_root = graph.add_node(graph.node_count() as u32);
    sources.clone().into_iter().for_each(|(index, demand)| {
        let source_id = graph
            .node_indices()
            .find(|&x| (graph.node_weight(x).unwrap() == &(index as u32))) // SOURCE_ID
            .unwrap();
        let arc = graph.add_edge(
            source_id,
            artificial_root,
            CustomEdgeIndices {
                cost: big_value,
                capacity: big_value,
                flow: demand,
            },
        );
        edge_tree[source_id.index()] = arc.index();
    });

    sinks.clone().into_iter().for_each(|(index, demand)| {
        let sink_id = graph
            .node_indices()
            .find(|&x| (graph.node_weight(x).unwrap() == &(index as u32))) // SINK_ID
            .unwrap();
        let arc = graph.add_edge(
            artificial_root,
            sink_id,
            CustomEdgeIndices {
                cost: big_value,
                capacity: big_value,
                flow: zero::<NUM>() - demand,
            },
        );
        edge_tree[sink_id.index()] = arc.index();
    });

    for node in graph.node_indices() {
        if node == artificial_root {
            continue;
        } else if graph.find_edge(artificial_root, node).is_none()
            && graph.find_edge(node, artificial_root).is_none()
        {
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
        source: (source),
        target: (target),
        cost: (cost),
        capacity: (capacity),
    };

    let graphstate: GraphState<NUM> = GraphState {
        out_base: (outbase),
        flow: (flow),
        state: (state),
    };

    (nodes, edges, graphstate)
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
    rest.iter().for_each(|&x| edges.push(x));

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

unsafe fn update_node_potentials<'a, NUM: CloneableNum>(
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
        change += unsafe {
            *edges.cost.get_unchecked(entering_arc)
                - *nodes.potential.get_unchecked(edges.source[entering_arc])
                + *nodes.potential.get_unchecked(edges.target[entering_arc])
        };
        start = k;
    } else {
        change -= unsafe {
            *edges.cost.get_unchecked(entering_arc)
                - *nodes.potential.get_unchecked(edges.source[entering_arc])
                + *nodes.potential.get_unchecked(edges.target[entering_arc])
        };
        start = l;
    }
    let mut current_node = nodes.thread[start];
    let depth = nodes.depth[start];
    nodes.potential[start] += change;
    while *nodes.depth.get_unchecked(current_node) > depth {
        *nodes.potential.get_unchecked_mut(current_node) += change;
        current_node = *nodes.thread.get_unchecked(current_node);
    }
}


fn _compute_flowchange<'a, NUM: CloneableNum>(
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
    graphstate: &mut GraphState<NUM>,
    entering_arc: usize,
) -> (usize, bool) {
    let (i, j) = (edges.source[entering_arc], edges.target[entering_arc]);
    let up_restricted = graphstate.flow[entering_arc] != zero();

    let mut current_i = i;
    let mut current_j = j;

    let mut min_delta = if up_restricted {
        (graphstate.flow[entering_arc], entering_arc)
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
                    delta = (edges.capacity[arc_j] - graphstate.flow[arc_j], arc_j);
                } else {
                    delta = (graphstate.flow[arc_j], arc_j);
                }
                if delta.0 < min_delta_j.0 {
                    min_delta_j = delta
                };
            } else {
                if current_j == edges.source[arc_j] {
                    delta = (edges.capacity[arc_j] - graphstate.flow[arc_j], arc_j);
                } else {
                    delta = (graphstate.flow[nodes.edge_tree[current_j]], arc_j);
                }
                if delta.0 <= min_delta_j.0 {
                    min_delta_j = delta
                };
            }
            current_j = nodes.predecessor[current_j].unwrap();
        } else {
            if up_restricted {
                if current_i == edges.source[arc_i] {
                    delta = (edges.capacity[arc_i] - graphstate.flow[arc_i], arc_i);
                } else {
                    delta = (graphstate.flow[arc_i], arc_i);
                }
                if delta.0 <= min_delta_i.0 {
                    min_delta_i = delta
                };
            } else {
                if current_i == edges.target[arc_i] {
                    delta = (edges.capacity[arc_i] - graphstate.flow[arc_i], arc_i);
                } else {
                    delta = (graphstate.flow[arc_i], arc_i);
                }
                if delta.0 < min_delta_i.0 {
                    min_delta_i = delta
                };
            }
            current_i = nodes.predecessor[current_i].unwrap();
        }
    }

    let mut branch: bool = false;
    if min_delta.0 > min_delta_i.0 {
        min_delta = min_delta_i;
        branch = true;
    }
    if min_delta.0 > min_delta_j.0 {
        min_delta = min_delta_j;
        branch = false;
    }
    if min_delta_j.0 == min_delta_i.0 {
        min_delta = if up_restricted {
            branch = false;
            min_delta_j
        } else {
            branch = true;
            min_delta_i
        }
    }

    if min_delta.0 != zero() {
        current_i = i;
        current_j = j;
        if up_restricted {
            graphstate.flow[entering_arc] -= min_delta.0;
        } else {
            graphstate.flow[entering_arc] += min_delta.0;
        }
        if graphstate.flow[entering_arc] == zero() {
            graphstate.state[entering_arc] = one();
        } else if graphstate.flow[entering_arc] == edges.capacity[entering_arc] {
            graphstate.state[entering_arc] = zero::<NUM>() - one();
        }
        while current_j != current_i {
            let arc_i = nodes.edge_tree[current_i];
            let arc_j = nodes.edge_tree[current_j];
            if nodes.depth[current_i] < nodes.depth[current_j] {
                if up_restricted {
                    if current_j == edges.target[arc_j] {
                        graphstate.flow[arc_j] += min_delta.0;
                    } else {
                        graphstate.flow[arc_j] -= min_delta.0;
                    }
                } else {
                    if current_j == edges.source[arc_j] {
                        graphstate.flow[arc_j] += min_delta.0;
                    } else {
                        graphstate.flow[arc_j] -= min_delta.0;
                    }
                }
                if graphstate.flow[arc_j] == zero() {
                    graphstate.state[arc_j] = one();
                } else if graphstate.flow[arc_j] == edges.capacity[arc_j] {
                    graphstate.state[arc_j] = zero::<NUM>() - one();
                }
                current_j = nodes.predecessor[current_j].unwrap();
            } else {
                if up_restricted {
                    if current_i == edges.source[arc_i] {
                        graphstate.flow[arc_i] += min_delta.0;
                    } else {
                        graphstate.flow[arc_i] -= min_delta.0;
                    }
                } else {
                    if current_i == edges.target[arc_i] {
                        graphstate.flow[arc_i] += min_delta.0;
                    } else {
                        graphstate.flow[arc_i] -= min_delta.0;
                    }
                }
                if graphstate.flow[arc_i] == zero() {
                    graphstate.state[arc_i] = one();
                } else if graphstate.flow[arc_i] == edges.capacity[arc_i] {
                    graphstate.state[arc_i] = zero::<NUM>() - one();
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
    edges: &Edges<NUM>,
    nodes: &mut Nodes<NUM>,
    graphstate: &mut GraphState<NUM>,
    entering_arc: usize,
    leaving_arc: usize,
    position: Option<usize>,
    branch: bool,
) {
    if leaving_arc == entering_arc {
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
    if branch {
        path_from_i = vec![i; nodes.depth[i] + 1 - cutting_depth];
        path_from_j = vec![j; nodes.depth[j] + 1];
    } else {
        path_from_i = vec![i; nodes.depth[i] + 1];
        path_from_j = vec![j; nodes.depth[j] + 1 - cutting_depth];
    }

    //fill vector
    let mut current_node1: Option<usize> = Some(i);
    let mut current_node2: Option<usize> = Some(j);
    for index in 0..path_from_i.len() {
        path_from_i[index] = current_node1.unwrap();
        current_node1 = unsafe { *nodes.predecessor.get_unchecked(current_node1.unwrap()) };
    }
    for index in 0..path_from_j.len() {
        path_from_j[index] = current_node2.unwrap();
        current_node2 = unsafe { *nodes.predecessor.get_unchecked(current_node2.unwrap()) };
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
    unsafe {
        while *nodes.depth.get_unchecked(current_node.index())
            > *nodes
                .depth
                .get_unchecked(path_to_change.last().unwrap().index())
        {
            block_parcour.push(current_node);
            current_node = *nodes.thread.get_unchecked(current_node.index());
        }
    }
    let mut dirty_rev_thread: Vec<usize> = vec![];
    let nodeid_to_block = nodes.revthread[block_parcour[0].index()];
    nodes.thread[nodeid_to_block.index()] = nodes.thread[block_parcour.last().unwrap().index()];
    dirty_rev_thread.push(nodeid_to_block);

    path_to_change
        .iter()
        .take(path_to_change.len() - 1)
        .for_each(|&x| unsafe {
            let mut current = *nodes.thread.get_unchecked(x.index());
            let mut old_last = x;
            while *nodes.depth.get_unchecked(current.index())
                > *nodes.depth.get_unchecked(x.index())
            {
                old_last = current;
                current = *nodes.thread.get_unchecked(current.index());
            }
            nodes.thread[nodes.revthread.get_unchecked(x.index()).index()] = current;
            dirty_rev_thread.push(*nodes.revthread.get_unchecked(x.index()));
            nodes.thread[old_last.index()] = nodes.predecessor.get_unchecked(x.index()).unwrap();
            dirty_rev_thread.push(old_last);
        });

    let mut current = nodes.thread[path_to_change.last().unwrap().index()];
    let mut old = *path_to_change.last().unwrap();
    unsafe {
        while *nodes.depth.get_unchecked(current.index())
            > *nodes
                .depth
                .get_unchecked(path_to_change.last().unwrap().index())
        {
            old = current;
            current = *nodes.thread.get_unchecked(current.index());
        }
    }

    nodes.thread[old.index()] = nodes.thread[path_to_root[0].index()];
    dirty_rev_thread.push(old);
    nodes.thread[path_to_root[0].index()] = path_to_change[0];
    dirty_rev_thread.push(path_to_root[0]);

    dirty_rev_thread.into_iter().for_each(|new_rev| unsafe {
        nodes.revthread[nodes.thread.get_unchecked(new_rev.index()).index()] = new_rev
    });

    //Predecessors update + edge_tree
    if path_from_i[path_from_i.len() - 1] != node_nb - 1 {
        nodes.predecessor[i.index()] = Some(j);
        path_from_i
            .iter()
            .enumerate()
            .skip(1)
            .for_each(|(index, &x)| {
                nodes.predecessor[x.index()] =
                    unsafe { Some(*path_from_i.get_unchecked(index - 1)) }
            });
        path_to_change = &path_from_i;
        path_to_root = &path_from_j;
    } else {
        nodes.predecessor[j.index()] = Some(i);
        path_from_j
            .iter()
            .enumerate()
            .skip(1)
            .for_each(|(index, &x)| {
                nodes.predecessor[x.index()] =
                    unsafe { Some(*path_from_j.get_unchecked(index - 1)) }
            });
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
            nodes.edge_tree[x] = unsafe { *temp.get_unchecked(index - 1) };
        });

    //update depth
    nodes.depth[path_to_change[0].index()] = nodes.depth[path_to_root[0].index()] + 1;
    path_to_change.iter().skip(1).for_each(|x| {
        nodes.depth[x.index()] = unsafe {
            nodes
                .depth
                .get_unchecked(nodes.predecessor.get_unchecked(x.index()).unwrap().index())
                + 1
        }
    });
    block_parcour.iter().for_each(|&x| {
        nodes.depth[x.index()] = unsafe {
            nodes
                .depth
                .get_unchecked(nodes.predecessor.get_unchecked(x.index()).unwrap().index())
                + 1
        }
    });

    graphstate.out_base[position.unwrap()] = leaving_arc;
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

pub fn min_cost<NUM: CloneableNum + 'static, PR: PivotRules<NUM>>(
    mut graph: DiGraph<u32, CustomEdgeIndices<NUM>>,
    sources: Vec<(usize, NUM)>, //(node_id, demand)
    sinks: Vec<(usize, NUM)>,   //(node_id, demand)
    pivotrule: PR,
    thread_nb: usize,
    scaling: usize,
) -> (State<NUM>, Vec<NUM>, Vec<NUM>) {
    let (mut nodes, edges, mut graphstate) =
        initialization::<NUM>(&mut graph, sources, sinks.clone());
    solve(
        &mut graph,
        &edges,
        &mut nodes,
        &mut graphstate,
        sinks,
        pivotrule,
        thread_nb,
        scaling,
    )
}

pub fn min_cost_from_state<NUM: CloneableNum + 'static, PR: PivotRules<NUM>>(
    mut graph: DiGraph<u32, CustomEdgeIndices<NUM>>,
    edges_state: &Edges<NUM>,
    nodes_state: &Nodes<NUM>,
    graph_state: &mut GraphState<NUM>,
    sinks: Vec<(usize, NUM)>, //vec![(node_id, demand)]
    pivotrule: PR,
    thread_nb: usize,
    scaling: usize,
) -> (State<NUM>, Vec<NUM>, Vec<NUM>) {
    let (edges, mut nodes, mut graphstate) =
        (edges_state.clone(), nodes_state.clone(), graph_state);
    solve(
        &mut graph,
        &edges,
        &mut nodes,
        &mut graphstate,
        sinks,
        pivotrule,
        thread_nb,
        scaling,
    )
}

//main algorithm function
fn solve<NUM: CloneableNum + 'static, PR: PivotRules<NUM>>(
    graph: &mut DiGraph<u32, CustomEdgeIndices<NUM>>,
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
    graph_state: &mut GraphState<NUM>,
    sinks: Vec<(usize, NUM)>, //vec![(node_id, demand)]
    pivotrule: PR,
    thread_nb: usize,
    scaling: usize,
) -> (State<NUM>, Vec<NUM>, Vec<NUM>) {
    ThreadPoolBuilder::new()
        .num_threads(thread_nb)
        .build_global()
        .unwrap();

    let (edges, mut nodes, mut graphstate) = (edges.clone(), nodes.clone(), graph_state);

    let multiply_factor = scaling;
    let divide_factor = 1;

    let mut _block_size = multiply_factor
        * std::cmp::min(
            (graphstate.out_base.len() as f64).sqrt() as usize,
            graphstate.out_base.len() / 100,
        )
        / divide_factor as usize;
    let mut iteration = 0;

    let (mut _index, mut entering_arc) =
        pivotrule.find_entering_arc(&edges, &nodes, &graphstate, 0, _block_size);

    while entering_arc.is_some() {
        let (leaving_arc, branch) =
            _compute_flowchange(&edges, &nodes, &mut graphstate, entering_arc.unwrap());

        update_sptree(
            &edges,
            &mut nodes,
            &mut graphstate,
            entering_arc.unwrap(),
            leaving_arc,
            _index,
            branch,
        );

        unsafe { update_node_potentials(&edges, &mut nodes, entering_arc.unwrap(), leaving_arc) };

        (_index, entering_arc) =
            pivotrule.find_entering_arc(&edges, &nodes, &graphstate, _index.unwrap(), _block_size);

        iteration += 1;
    }
    print!(", iterations = {:?}", iteration);
    //graph.remove_node(NodeIndex::new(graph.node_count() - 1));
    let mut cost: NUM = zero();
    let mut total_flow: NUM = zero();
    graph.clone().edge_references().for_each(|x| {
        graph.edge_weight_mut(x.id()).expect("found").flow = graphstate.flow[x.id().index()];
        cost += graphstate.flow[x.id().index()] * edges.cost[x.id().index()];
    });
    sinks.iter().for_each(|(index, _)| {
        graph
            .edges_directed(NodeIndex::new(*index), Incoming)
            .for_each(|x| total_flow += graphstate.flow[x.id().index()])
    });
    sinks.iter().for_each(|(index, _)| {
        graph
            .edges_directed(NodeIndex::new(*index), Outgoing)
            .for_each(|x| total_flow -= graphstate.flow[x.id().index()])
    });
    //println!("{:?}", Dot::new(&graph));
    let mut sink_sum:NUM = zero();
    sinks.into_iter().for_each(|(_, demand)| sink_sum += demand);
    let status: Status;
    if total_flow == (zero::<NUM>() - sink_sum) {
        status = Status::Optimal;
    } else {
        status = Status::DemandGap;
    }

    print!(", flow = {:?}, cost = {:?}, status = {:?}", total_flow, cost, status);
    let state: State<NUM> = State {
        nodes_state: (nodes.clone()),
        graph_state: (graphstate.clone()),
        edges_state: (edges),
        status: (status),
    };
    (state, graphstate.flow.clone(), nodes.potential)
}
