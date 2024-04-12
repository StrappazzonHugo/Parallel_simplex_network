use itertools::Itertools;
use num_traits::identities::one;
use num_traits::identities::zero;
use num_traits::Num;
use petgraph::algo::bellman_ford;
use petgraph::graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::stable_graph::IndexType;
//use rayon::prelude::*;
//use rayon::ThreadPoolBuilder;

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
) -> (Nodes<NUM>, Edges<NUM>) {
    let max_node_id: u32 = (graph.node_count() - 1) as u32;

    let sink_id = graph
        .node_indices()
        .find(|&x| (graph.node_weight(x).unwrap() == &max_node_id))
        .unwrap();

    let mut big_value: NUM = zero();
    for _ in 0..100 {
        big_value += demand;
    }

    let artificial_root = graph.add_node(graph.node_count() as u32);
    for node in graph.node_indices() {
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
                    state: zero(),
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
                    state: zero(),
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
                    state: zero(),
                },
            );
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

fn find_cycle_with_arc<'a, NUM: CloneableNum>(
    edges: &Edges<NUM>,
    nodes: &mut Nodes<NUM>,
    entering_arc: usize,
) -> Vec<usize> {
    let (i, j) = (edges.source[entering_arc], edges.target[entering_arc]);
    let mut path_from_i: Vec<usize> = vec![i];
    let mut path_from_j: Vec<usize> = vec![j];

    let mut current_node_i: usize = i;
    let mut current_node_j: usize = j;

    while current_node_j != current_node_i {
        if nodes.depth[current_node_i] < nodes.depth[current_node_j] {
            current_node_j = nodes.predecessor[current_node_j].expect("found");
            path_from_j.push(current_node_j);
        } else {
            current_node_i = nodes.predecessor[current_node_i].expect("found");
            path_from_i.push(current_node_i);
        }
    }
    path_from_i
        .iter()
        .rev()
        .skip(1)
        .for_each(|&x| path_from_j.push(x));
    path_from_j
}

//computing delta the amount of flow we can augment through the cycle
//returning the leaving edge
fn compute_flowchange<'a, NUM: CloneableNum>(
    edges: &mut Edges<NUM>,
    graph: &DiGraph<u32, CustomEdgeIndices<NUM>>,
    cycle: &mut Vec<usize>,
    entering_arc: usize,
) -> usize {
    let mut edge_in_cycle: Vec<usize> = vec![entering_arc; cycle.len()];
    let mut direction: Vec<NUM> = vec![one(); cycle.len()];
    let mut delta: Vec<NUM> = vec![zero(); cycle.len()];

    cycle.push(cycle[0]);

    if edges.flow[entering_arc] != zero() {
        cycle.reverse();
    };

    cycle
        .iter()
        .tuple_windows::<(&usize, &usize)>()
        .enumerate()
        .for_each(|(index, (i, j))| {
            let edge = graph.find_edge(NodeIndex::new(*i), NodeIndex::new(*j)); // TODO change this
            if edge.is_none() {
                direction[index] = zero::<NUM>() - one();
                edge_in_cycle[index] = graph
                    .find_edge(NodeIndex::new(*j), NodeIndex::new(*i))
                    .expect("found")
                    .index();
            } else {
                edge_in_cycle[index] = edge.unwrap().index();
            }
        });

    edge_in_cycle.iter().enumerate().for_each(|(index, &x)| {
        if direction[index] == one() {
            delta[index] = edges.capacity[x] - edges.flow[x];
        } else {
            delta[index] = edges.flow[x];
        }
    });
    let flowchange = delta
        .iter()
        .enumerate()
        .min_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .unwrap();
    let farthest_blocking_edge: usize = edge_in_cycle[flowchange.0];

    //flow update if needed
    if *flowchange.1 != zero::<NUM>() {
        edge_in_cycle.iter().enumerate().for_each(|(index, &x)| {
            edges.flow[x] += direction[index] * *flowchange.1;
            if edges.flow[x] == zero() {
                edges.state[x] = one();
            }
            if edges.flow[x] == edges.capacity[x] {
                edges.state[x] = zero::<NUM>() - one();
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
    edges: &mut Edges<NUM>,
    nodes: &mut Nodes<NUM>,
    entering_arc: usize,
    leaving_arc: usize,
    position: Option<usize>,
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
    if nodes.predecessor[k] == Some(l) {
        nodes.predecessor[k] = None;
    } else {
        nodes.predecessor[l] = None;
    }
    let mut path_from_i: Vec<usize> = Vec::new();
    let mut path_from_j: Vec<usize> = Vec::new();

    let mut current_node: Option<usize> = Some(i);
    while !current_node.is_none() {
        path_from_i.push(current_node.unwrap());
        current_node = nodes.predecessor[current_node.unwrap()];
    }
    current_node = Some(j);
    while !current_node.is_none() {
        path_from_j.push(current_node.unwrap());
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

    //Predecessors update
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
        if rc < min {
            min = rc;
            entering_arc = Some(arc);
            index = Some(i);
        }
    }
    (index, entering_arc)
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
    edges: &Edges<NUM>,
    nodes: &Nodes<NUM>,
    mut index: Option<usize>,
    block_size: usize,
) -> (Option<usize>, Option<usize>) {
    let mut block_number = 0;
    let mut arc_index = None;
    if index.is_none() {
        index = Some(0);
    }
    while block_size * block_number <= edges.out_base.len() {
        let (index_, rc_entering_arc) = edges
            .out_base
            .iter()
            .enumerate()
            .cycle()
            .skip(index.unwrap())
            .take(block_size)
            .map(|(index, &arc)| {
                (
                    index,
                    edges.state[arc]
                        * (edges.cost[arc] - nodes.potential[edges.source[arc]]
                            + nodes.potential[edges.target[arc]]),
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
        let arc = if arc_index.unwrap() < edges.out_base.len() {
            edges.out_base[arc_index.unwrap()]
        } else {
            edges.out_base[arc_index.unwrap() - edges.out_base.len()]
        };
        (arc_index, Some(arc))
    }
}

//main algorithm function
pub fn min_cost<NUM: CloneableNum>(
    mut graph: DiGraph<u32, CustomEdgeIndices<NUM>>,
    demand: NUM,
) -> DiGraph<u32, CustomEdgeIndices<NUM>> {
    let (mut nodes, mut edges) = initialization::<NUM>(&mut graph, demand);
    let _block_size = (graph.edge_count() / 15) as usize;
    let mut _index: Option<usize> = Some(0);
    let mut entering_arc: Option<usize>;

    for i in 0..edges.out_base.len() {
        entering_arc = Some(edges.out_base[i]);
        if (edges.state[i] * get_reduced_cost_edgeindex(&edges, &nodes, entering_arc.unwrap()))
            >= zero()
        {
            continue;
        }
        let mut cycle = find_cycle_with_arc(&edges, &mut nodes, entering_arc.unwrap());
        let leaving_arc =
            compute_flowchange(&mut edges, &mut graph, &mut cycle, entering_arc.unwrap());
        update_sptree(
            &mut edges,
            &mut nodes,
            entering_arc.unwrap(),
            leaving_arc,
            _index,
        );
        update_node_potentials(&mut edges, &mut nodes, entering_arc.unwrap(), leaving_arc);
    }

    (_index, entering_arc) = _find_first_arc(&edges, &mut nodes, _index);
    //ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();
    //(_index, entering_arc) = _find_first_arc(&edges, &nodes, Some(0));
    let mut _iteration = 0;
    while !entering_arc.is_none() {
        let mut cycle = find_cycle_with_arc(&edges, &mut nodes, entering_arc.unwrap());
        let leaving_arc =
            compute_flowchange(&mut edges, &mut graph, &mut cycle, entering_arc.unwrap());

        update_sptree(
            &mut edges,
            &mut nodes,
            entering_arc.unwrap(),
            leaving_arc,
            _index,
        );

        update_node_potentials(&mut edges, &mut nodes, entering_arc.unwrap(), leaving_arc);

        (_index, entering_arc) =

        //_find_block_search(&edges, &nodes, _index, _block_size);

        //_find_best_arc(&edges, &mut nodes);

        _find_first_arc(&edges, &mut nodes, _index);

        _iteration += 1;
    }
    println!("iterations : {:?}", _iteration);
    let mut cost: NUM = zero();
    edges
        .flow
        .iter()
        .enumerate()
        .for_each(|(index, &x)| cost += x * edges.cost[index]);
    graph.clone().edge_references().for_each(|x| graph.edge_weight_mut(x.id()).expect("found").flow = edges.flow[x.id().index()]);
    println!("total cost = {:?}", cost);
    graph
}
