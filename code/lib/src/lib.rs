use debug_print::debug_println;
use itertools::Itertools;
use petgraph::algo::bellman_ford;
//use num; not used yet
use petgraph::algo::find_negative_cycle;
use petgraph::dot::Dot;
use petgraph::graph::node_index;
use petgraph::graph::DefaultIx;
use petgraph::graph::EdgeReference;
use petgraph::matrix_graph::NodeIndex;
use petgraph::prelude::*;
//use std::any::type_name;
use std::collections::HashMap;

const INF: i32 = 99999999;

//TODO standardize node index type u32/i32 its currently a mess
#[derive(Debug, Clone)]
struct SPTree {
    t: Vec<(u32, u32)>,
    l: Vec<(u32, u32)>,
    u: Vec<(u32, u32)>,
}

//TODO can easily be done with functions getcost(graph, i, j)
//                                       getcapacity(graph, i, j)
//                                       getflow(graph, i, j)
trait EdgeIndexed<NUM> {
    fn cost(i: u32, j: u32) -> NUM;
    fn capacity(i: u32, j: u32) -> NUM;
    fn flow(i: u32, j: u32) -> NUM;
}

#[derive(Debug, Clone)]
pub struct CustomEdgeIndices<NUM> {
    pub cost: NUM,
    pub capacity: NUM,
    pub flow: NUM,
}

impl<NUM> CustomEdgeIndices<NUM> {
    fn update_flow(&mut self, x: NUM) {
        self.flow = x;
    }
}
/*
fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}*/

//TODO function over generics integer : DiGraph<u32, CustomEdgeIndices<INT>>
fn initialization<'a, NUM, Ix>(
    mut graph: DiGraph<u32, CustomEdgeIndices<i32>>,
) -> (SPTree, DiGraph<u32, CustomEdgeIndices<i32>>) {
    let mut N = graph.node_count();
    let mut u = vec![0; N * N];
    //building capacity matrix needed for Push-Relabel algorithm TODO modify PushRelabel such that
    //it take Digraph structure instead of PRstruct
    graph.edge_references().for_each(|x| {
        u[x.source().index() * N + x.target().index()] = x.weight().capacity;
    });
    let g = PRstruct {
        n: N,
        capacity: u.clone(),
        flow: vec![0; N * N],
        label: Vec::new(),
        excess: Vec::new(),
        excess_vertices: Vec::new(),
        seen: Vec::new(),
    };

    //Supposing source_id = 0 and sink_id = lastnode id
    //TODO add a way to specify source/sink instead
    let source_id = 0;
    let sink_id = graph.node_count() - 1;
    let initial_feasible_flow = max_flow(source_id, sink_id, g);

    graph.clone().edge_references().for_each(|x| {
        graph
            .edge_weight_mut(x.id())
            .unwrap()
            .update_flow(initial_feasible_flow[x.source().index() * N + x.target().index()]);
    });

    debug_println!("{:?}", Dot::new(&graph));
    
    //removing orphan nodes from the graph
    find_orphan_nodes(graph.clone(), initial_feasible_flow.clone(), N)
        .iter()
        .for_each(|&x| {
            graph.remove_node(x).unwrap();
        });

    debug_println!("{:?}", Dot::new(&graph));

    //Filling T arc-set with free arc of the initial feasible solution
    //L with restricted at lowerbound arcs
    //u with restricted at upperbound arcs
    let freearcs = graph
        .edge_references()
        .filter(|&x| x.weight().flow > 0 && x.weight().flow < x.weight().capacity);

    let up_bound_restricted = graph
        .edge_references()
        .filter(|&x| x.weight().flow == x.weight().capacity && x.weight().capacity > 0);

    let low_bound_restricted = graph
        .edge_references()
        .filter(|&x| x.weight().flow == 0 && x.weight().capacity > 0);

    let mut T: Vec<(u32, u32)> = freearcs
        .map(|x| (x.source().index() as u32, x.target().index() as u32))
        .collect();

    let mut U: Vec<(u32, u32)> = up_bound_restricted
        .map(|x| (x.source().index() as u32, x.target().index() as u32))
        .collect();

    let L: Vec<(u32, u32)> = low_bound_restricted
        .map(|x| (x.source().index() as u32, x.target().index() as u32))
        .collect();

    debug_println!("initial_feasible_flow = {:?}", initial_feasible_flow);

    debug_println!("T ={:?}", T);
    debug_println!("L = {:?}", L);
    debug_println!("U = {:?}", U);

    //while T isnt a tree : we adding one edge from U to T
    //we cant obtain cycle at iteration n since we necesseraly
    //have a spanning tree the iteration n-1
    while !is_tree(T.clone(), graph.node_count()) {
        T.push(U.pop().unwrap());
    }

    debug_println!("INITIAL SOLUTION : ");
    let tlu_solution = SPTree { t: T, l: L, u: U };
    debug_println!("T = {:?}", tlu_solution.t);
    debug_println!("L = {:?}", tlu_solution.l);
    debug_println!("U = {:?}", tlu_solution.u);

    (tlu_solution, graph)
}

//checking tree using number of edge property in spanning tree
fn is_tree<E>(edges: Vec<E>, n: usize) -> bool {
    edges.len() + 1 == n
}

//iterate over nodes and checking : sum(flow in adjacent edges) > 0
fn find_orphan_nodes<N, E>(
    graph: DiGraph<N, E>,
    flow: Vec<i32>,
    n: usize,
) -> Vec<petgraph::graph::NodeIndex> {
    graph
        .node_indices()
        .skip(1)
        .rev()
        .skip(1)
        .filter(|&u| {
            graph.neighbors(u).fold(true, |check, v| {
                check && flow[u.index() * n + v.index()] == 0
            })
        })
        .collect()
}

//TODO refactor this abomination (it work)
fn compute_node_potential<'a, N>(
    graph: DiGraph<N, CustomEdgeIndices<i32>>,
    sptree: SPTree,
) -> Vec<i32> {
    let mut pi: Vec<i32> = vec![-INF; graph.node_count()]; //vector of node potential
    pi[0] = 0; // init of root potential
    let mut node_computed: Vec<i32> = vec![0];
    let mut current: i32 = 0;

    //iterate over edges in T and computing node potential with the 'current' variable that go from
    //one id of computed node potential to another
    //useless pattern matching
    while node_computed.len() < graph.node_count() && pi.iter().contains(&(-INF)) {
        sptree.t.iter().for_each(|&(u, v)| match (u, v) {
            (u, v) if u as i32 == current => {
                pi[v as usize] = pi[current as usize]
                    - graph
                        .edge_weight(
                            graph
                                .find_edge(node_index(current as usize), node_index(v as usize))
                                .unwrap(),
                        )
                        .unwrap()
                        .cost;
            }
            (u, v) if v as i32 == current => {
                pi[u as usize] = graph
                    .edge_weight(
                        graph
                            .find_edge(node_index(u as usize), node_index(current as usize))
                            .unwrap(),
                    )
                    .unwrap()
                    .cost
                    + pi[current as usize];
            }
            _ => (),
        });
        current = pi
            .clone()
            .iter()
            .enumerate()
            .find_position(|x| *(x.1) != -INF && !node_computed.iter().contains(&(x.0 as i32)))
            .unwrap()
            .0 as i32;
        node_computed.push(current);
        debug_println!("node computed = {:?}", node_computed);
    }
    pi
}

//Updating potential inspired by : NETWORK FLOWS Theory, Algorithms, and Applications (p.419)
//modified such that the difference from T1 and T2 is done with a BellmanFord algorithm
//detecting the single edge cost put to 1 that divide T1 and T2 in T
fn update_potential(
    potential: Vec<i32>,
    sptree: SPTree,
    leaving_arc: (u32, u32),
    reduced_cost: HashMap<(i32, i32), i32>,
) -> Vec<i32> {
    let mut edges: Vec<(u32, u32, f32)> = sptree
        .t
        .iter()
        .filter(|&&x| x != leaving_arc)
        .map(|x| (x.0, x.1, 0.))
        .collect();
    edges.push((leaving_arc.0, leaving_arc.1, 1.)); // <-- edge separating T1 and T2
    let g = Graph::<(), f32, Directed>::from_edges(edges.clone());
    let path_cost = bellman_ford(&g, NodeIndex::new(0)).unwrap().distances;
    let potentials_to_update: Vec<usize> = path_cost
        .iter()
        .enumerate()
        .filter(|&(_, cost)| cost > &0.)
        .map(|x| x.0)
        .collect();
    let mut change: i32 = 0;
    if potentials_to_update.contains(&(leaving_arc.1 as usize)) {
        change -= reduced_cost
            .get(&(leaving_arc.0 as i32, leaving_arc.1 as i32))
            .unwrap();
    } else {
        change += reduced_cost
            .get(&(leaving_arc.0 as i32, leaving_arc.1 as i32))
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
fn compute_reduced_cost<N>(
    pi: Vec<i32>,
    graph: DiGraph<N, CustomEdgeIndices<i32>>,
    sptree: SPTree,
) -> HashMap<(i32, i32), i32> {
    let mut reduced_cost: HashMap<(i32, i32), i32> = HashMap::new();
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
        reduced_cost.insert((u as i32, v as i32), 0);
    });
    reduced_cost
}

//Working but not sure about the way it is... (it work)
//probably better way to manage type Option<(_,_)>
fn find_entering_arc(sptree: SPTree, reduced_cost: HashMap<(i32, i32), i32>) -> Option<(u32, u32)> {
    let min_l = sptree.l.iter().min_by(|a, b| {
        reduced_cost
            .get(&(a.0 as i32, a.1 as i32))
            .unwrap()
            .cmp(reduced_cost.get(&(b.0 as i32, b.1 as i32)).unwrap())
    });
    let max_u = sptree.u.iter().max_by(|a, b| {
        reduced_cost
            .get(&(a.0 as i32, a.1 as i32))
            .unwrap()
            .cmp(reduced_cost.get(&(b.0 as i32, b.1 as i32)).unwrap())
    });
    let rc_min_l = if min_l != None {
        reduced_cost.get(&(min_l.unwrap().0 as i32, min_l.unwrap().1 as i32))
    } else {
        Some(&0i32)
    };
    let rc_max_u = if max_u != None {
        reduced_cost.get(&(max_u.unwrap().0 as i32, max_u.unwrap().1 as i32))
    } else {
        Some(&0i32)
    };
    debug_println!("min_l = {:?}, rc = {:?}", min_l, rc_min_l);
    debug_println!("max_u = {:?} rc = {:?}", max_u, rc_max_u);
    //optimality conditions
    if rc_min_l >= Some(&0i32) && rc_max_u <= Some(&0i32) {
        debug_println!("OPTIMAL");
        return None;
    }
    if rc_max_u <= Some(&0i32) {
        return Some(*min_l.unwrap());
    }
    if rc_min_l >= Some(&0i32) {
        return Some(*max_u.unwrap());
    }
    if rc_min_l.unwrap().abs() >= rc_max_u.unwrap().abs() {
        return Some(*min_l.unwrap());
    } else {
        return Some(*max_u.unwrap());
    }
}

//finding cycle using Bellmanford algorithm to find negative weight cycle
//we build a graph with a cycle and forcing every arc weight to 0 except for one arc that we know
//in the cycle. BF then find negative weight cycle
fn find_cycle(sptree: SPTree, entering_arc: (u32, u32)) -> Vec<u32> {
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
    vec_id_cycle
}

//checking if the specified edge is in forward direction according to a cycle
//needed in function compute_flowchange(...)  
fn is_forward(edgeref: EdgeReference<CustomEdgeIndices<i32>>, cycle: Vec<u32>) -> bool {
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
fn distance_in_cycle(edgeref: EdgeReference<CustomEdgeIndices<i32>>, cycle: Vec<u32>) -> usize {
    let (i, j) = (edgeref.source().index(), edgeref.target().index());
    let mut altered_cycle = cycle.clone();
    altered_cycle.push(*cycle.first().unwrap());
    let tuple_altered_cycle: Vec<(u32, u32)> = altered_cycle
        .into_iter()
        .tuple_windows::<(u32, u32)>()
        .collect();
    let res = tuple_altered_cycle
        .iter()
        .enumerate()
        .find(|&(_, x)| x == &(i as u32, j as u32) || x == &(j as u32, i as u32))
        .unwrap();
    return res.0;
}

//computing delta the amount of unit of flow we can augment through the cycle
fn compute_flowchange<N>(
    graph: DiGraph<N, CustomEdgeIndices<i32>>,
    cycle: Vec<u32>,
) -> ((u32, u32), Vec<((u32, u32), i32)>) {
    let mut altered_cycle = cycle.clone();
    altered_cycle.push(*cycle.first().unwrap());
    let tuple_altered_cycle: Vec<(u32, u32)> = altered_cycle
        .into_iter()
        .tuple_windows::<(u32, u32)>()
        .collect();
    let mut edge_in_cycle: Vec<EdgeReference<CustomEdgeIndices<i32>>> = graph
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
    //debug_println!("edge_in_cycle before reorder = {:?}", edge_in_cycle);
    edge_in_cycle = edge_in_cycle
        .into_iter()
        .sorted_by_key(|&x| distance_in_cycle(x, cycle.clone()))
        .rev()
        .collect();
    //debug_println!("edge_in_cycle after reorder = {:?}", edge_in_cycle);
    let delta: Vec<i32> = edge_in_cycle
        .iter()
        .map(|&x| {
            if is_forward(x, cycle.clone()) {
                debug_println!(
                    "edge {:?} is forward, delta_ij ={:?}",
                    x,
                    x.weight().capacity - x.weight().flow
                );
                x.weight().capacity - x.weight().flow
            } else {
                debug_println!("edge {:?} is backward delta_ij ={:?}", x, x.weight().flow);
                x.weight().flow
            }
        })
        .collect();
    let flowchange = delta.iter().enumerate().min_by_key(|x| x.1).unwrap();
    let farthest_blocking_edge: (u32, u32) = (
        edge_in_cycle[flowchange.0].source().index() as u32,
        edge_in_cycle[flowchange.0].target().index() as u32,
    );
    let edge_flow_change: Vec<((u32, u32), i32)> = edge_in_cycle
        .iter()
        .map(|&x| {
            (
                (x.source().index() as u32, x.target().index() as u32),
                if is_forward(x, cycle.clone()) {
                    *flowchange.1
                } else {
                    -*flowchange.1
                },
            )
        })
        .collect();
    debug_println!(
        "farthest blocking edge = {:?}\namount of flow change = {:?}",
        edge_in_cycle[flowchange.0],
        flowchange.1
    );

    return (farthest_blocking_edge, edge_flow_change);
}

//updating the flow in every edge specified in flow_graph variable
fn update_flow<N>(
    graph: DiGraph<N, CustomEdgeIndices<i32>>,
    flow_change: Vec<((u32, u32), i32)>,
) -> DiGraph<N, CustomEdgeIndices<i32>>
where
    N: Clone,
{
    let mut updated_graph = graph.clone();
    flow_change.iter().for_each(|((i, j), a)| {
        updated_graph
            .edge_weight_mut(
                graph
                    .find_edge(node_index(*i as usize), node_index(*j as usize))
                    .unwrap(),
            )
            .unwrap()
            .flow += a;
    });
    updated_graph
}

//updating sptree structure according to the leaving arc
//removing it from T and putting in L or U depending on the updated flow on the arc
fn update_sptree_with_leaving<N>(
    sptree: SPTree,
    leaving_arc: (u32, u32),
    graph: DiGraph<N, CustomEdgeIndices<i32>>,
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
        == 0;
    if at_min_capacity {
        updated_sptree.l.push(leaving_arc);
    } else {
        updated_sptree.u.push(leaving_arc)
    }
    updated_sptree
}

//adding entering_arc to T arc set of sptree
fn update_sptree_with_entering(sptree: SPTree, entering_arc: Option<(u32, u32)>) -> SPTree {
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

//structure for Push-Relabel algorithm
//TODO : function over generics integer
struct PRstruct {
    capacity: Vec<i32>, //2d vec flatten
    flow: Vec<i32>,     //2d vec flatten
    label: Vec<i32>,
    excess: Vec<i32>,
    excess_vertices: Vec<usize>,
    seen: Vec<usize>,
    n: usize,
}

fn push(u: usize, v: usize, mut g: PRstruct) -> PRstruct {
    let d = std::cmp::min(g.excess[u], g.capacity[g.n * u + v] - g.flow[g.n * u + v]);
    g.flow[g.n * u + v] += d;
    g.flow[g.n * v + u] -= d;
    g.excess[u] -= d;
    g.excess[v] += d;
    if d != 0 && g.excess[v] == d {
        g.excess_vertices.push(v);
    }
    g
}

fn relabel(u: usize, mut g: PRstruct) -> PRstruct {
    let mut d = INF;
    for i in 0..g.n {
        if (g.capacity[g.n * u + i] - g.flow[g.n * u + i]) > 0 {
            d = std::cmp::min(d, g.label[i]);
        }
    }
    if d < INF {
        g.label[u] = d + 1;
    }
    g
}

fn discharge(u: usize, mut g: PRstruct) -> PRstruct {
    while g.excess[u] > 0 {
        if g.seen[u] < g.n {
            let v = g.seen[u];
            if (g.capacity[g.n * u + v] - g.flow[g.n * u + v]) > 0 && g.label[u] > g.label[v] {
                g = push(u, v, g);
            } else {
                g.seen[u] += 1;
            }
        } else {
            g = relabel(u, g);
            g.seen[u] = 0;
        }
    }
    g
}

fn max_flow(s: usize, t: usize, mut g: PRstruct) -> Vec<i32> {
    g.label = vec![0; g.n];
    g.label[s] = g.n as i32;
    g.flow = vec![0; g.n * g.n];
    g.excess = vec![0; g.n];
    g.excess[s] = INF;
    for i in 0..g.n {
        if i != s {
            g = push(s, i, g);
        }
    }
    g.seen = vec![0; g.n];

    while !g.excess_vertices.is_empty() {
        let u = g.excess_vertices.pop();
        if u.unwrap() != s && u.unwrap() != t {
            g = discharge(u.unwrap(), g);
        }
    }
    let mut maxflow = 0;
    for i in 0..g.n {
        maxflow += g.flow[i * g.n + t];
    }
    debug_println!(" MAX FLOW : {:?}", maxflow);
    g.flow.iter().map(|&x| if x < 0 { 0 } else { x }).collect()
}

//main algorithm function
pub fn min_cost(
    graph: DiGraph<u32, CustomEdgeIndices<i32>>,
) -> DiGraph<u32, CustomEdgeIndices<i32>> {
    debug_println!("##################################### INITIALIZATION #####################################");
    let (mut tlu_solution, mut new_graph) = initialization::<i32, DefaultIx>(graph.clone());

    let mut potentials = compute_node_potential(new_graph.clone(), tlu_solution.clone());
    debug_println!("potentials = {:?}", potentials);
    let mut reduced_cost =
        compute_reduced_cost(potentials.clone(), new_graph.clone(), tlu_solution.clone());
    debug_println!("tlu_solution = {:?}", tlu_solution);
    debug_println!("reduced_cost = {:?}", reduced_cost);

    let mut entering_arc = find_entering_arc(tlu_solution.clone(), reduced_cost.clone());
    debug_println!("entering arc = {:?}", entering_arc);

    let mut iteration = 1;
    let max_iteration = 100;

    while entering_arc != None && iteration < max_iteration {
        debug_println!("##################################### ITERATION {:?} #####################################", iteration);

        let cycle = find_cycle(tlu_solution.clone(), entering_arc.unwrap());
        debug_println!("tlu_solution = {:?}", tlu_solution);
        debug_println!("entering arc = {:?}", entering_arc.unwrap());
        debug_println!("cycle found = {:?}", cycle);

        tlu_solution = update_sptree_with_entering(tlu_solution.clone(), entering_arc);

        let (leaving_arc, vec_flow_change) = compute_flowchange(new_graph.clone(), cycle);

        tlu_solution =
            update_sptree_with_leaving(tlu_solution.clone(), leaving_arc, new_graph.clone());

        new_graph = update_flow(new_graph.clone(), vec_flow_change);
        debug_println!("{:?}", Dot::new(&new_graph));

        potentials = update_potential(
            potentials.clone(),
            tlu_solution.clone(),
            entering_arc.unwrap(),
            reduced_cost.clone(),
        );
        //potentials = compute_node_potential(new_graph.clone(), tlu_solution.clone());
        debug_println!("potentials = {:?}", potentials);

        reduced_cost =
            compute_reduced_cost(potentials.clone(), new_graph.clone(), tlu_solution.clone());
        debug_println!("tlu_solution = {:?}", tlu_solution);
        debug_println!("reduced_cost = {:?}", reduced_cost);

        entering_arc = find_entering_arc(tlu_solution.clone(), reduced_cost.clone());
        debug_println!("entering arc = {:?}", entering_arc);
        iteration += 1;
    }
    if iteration == max_iteration {
        debug_println!("MAX iteration reached");
    }
    new_graph
}
