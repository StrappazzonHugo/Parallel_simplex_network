use itertools::Itertools;
use petgraph::algo::find_negative_cycle;
use petgraph::dot::Dot;
use petgraph::graph::node_index;
use petgraph::graph::DefaultIx;
use petgraph::graph::EdgeReference;
use petgraph::matrix_graph::NodeIndex;
use petgraph::prelude::*;
use petgraph::stable_graph::IndexType;
use std::any::type_name;
use std::collections::HashMap;

const INF: i32 = 99999999;

#[derive(Debug, Clone)]
struct SPTree {
    t: Vec<(u32, u32)>,
    l: Vec<(u32, u32)>,
    u: Vec<(u32, u32)>,
}

//TODO
trait EdgeIndexed<INT> {
    fn cost(i: u32, j: u32) -> INT;
    fn capacity(i: u32, j: u32) -> INT;
    fn flow(i: u32, j: u32) -> INT;
}

#[derive(Debug, Clone)]
struct CustomEdgeIndices<INT> {
    pub cost: INT,
    pub capacity: INT,
    pub flow: INT,
}

impl<INT> CustomEdgeIndices<INT> {
    /*fn update_cost(&mut self, c: INT) {
        self.cost = c;
    }*/
    fn update_flow(&mut self, x: INT) {
        self.flow = x;
    }
}

fn type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

//TODO function over generics integer : DiGraph<u32, CustomEdgeIndices<INT>>
fn initialization<'a, INT, Ix>(
    mut graph: DiGraph<u32, CustomEdgeIndices<i32>>,
    mut u: Vec<i32>,
    // c: Vec<i32>,
) -> (SPTree, DiGraph<u32, CustomEdgeIndices<i32>>) {
    let mut N = graph.node_count();
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
    let source_id = 0;
    let sink_id = graph.node_count() - 1;
    let mut initial_feasible_flow = max_flow(source_id, sink_id, g);

    graph.clone().edge_references().for_each(|x| {
        graph
            .edge_weight_mut(x.id())
            .unwrap()
            .update_flow(initial_feasible_flow[x.source().index() * N + x.target().index()]);
    });

    //= initial_feasible_flow[x.source().index() * N + x.target().index()
    println!("{:?}", Dot::new(&graph));

    find_orphan_nodes(graph.clone(), initial_feasible_flow.clone(), N)
        .iter()
        .for_each(|&x| {
            let mut try_to_remove: Vec<usize> = Vec::new();
            for i in 0..N {
                try_to_remove.push(i * N + x.index());
                try_to_remove.push(x.index() * N + i);
            }
            //removing flow value corresponding to oprhan node in u in reverse order so that indices doesnt mess up
            try_to_remove = try_to_remove.into_iter().sorted().rev().dedup().collect();
            try_to_remove.iter().for_each(|&y| {
                u.remove(y);
                initial_feasible_flow.remove(y);
            });
            //removing orphans nodes from the graph
            graph.remove_node(x).unwrap();
        });

    println!("{:?}", Dot::new(&graph));
    N = graph.node_count();
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

    println!("initial_feasible_flow = {:?}", initial_feasible_flow);
    println!("                    u = {:?}\n", u);

    println!("T ={:?}", T);
    println!("L = {:?}", L);
    println!("U = {:?}", U);

    while !is_tree(T.clone(), graph.node_count()) {
        T.push(U.pop().unwrap());
    }

    println!("INITIAL SOLUTION : ");
    let tlu_solution = SPTree { t: T, l: L, u: U };
    println!("T = {:?}", tlu_solution.t);
    println!("L = {:?}", tlu_solution.l);
    println!("U = {:?}", tlu_solution.u);

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
        println!("node computed = {:?}", node_computed);
    }
    pi
}

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

//Working but not sure about the way i did it
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
    println!("min_l = {:?}", min_l);
    println!("max_u = {:?}", max_u);
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
    //optimality conditions
    if rc_min_l >= Some(&0i32) && rc_max_u <= Some(&0i32) {
        println!("OPTIMAL");
        return None;
    }
    if rc_min_l.unwrap().abs() >= rc_max_u.unwrap().abs() && rc_min_l.unwrap() < &0 {
        return Some(*min_l.unwrap());
    } else {
        return Some(*max_u.unwrap());
    }
}

//finding cycle using Bellmanford algorithm
//we build a graph with a cycle and forcing every arc weight to 0 except for one arc that we know
//in the cycle. BF then find negative weight cycle
fn find_cycle(sptree: SPTree, entering_arc: (u32, u32)) -> Vec<u32> {
    println!("sptree.t = {:?}", sptree.t);
    let mut edges: Vec<(u32, u32, f32)> = sptree.t.iter().map(|&x| (x.0, x.1, 0.)).collect();
    let mut edges1: Vec<(u32, u32, f32)> = sptree.t.iter().map(|&x| (x.1, x.0, 0.)).collect();
    edges.append(&mut edges1);
    edges.push((entering_arc.0, entering_arc.1, -1.)); // <--- edge in the cycle we want
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

fn distance_in_cycle(edgeref: EdgeReference<CustomEdgeIndices<i32>>, cycle: Vec<u32>) -> usize {
    let (i, j) = (edgeref.source().index(), edgeref.target().index());
    let mut altered_cycle = cycle.clone();
    altered_cycle.push(*cycle.first().unwrap());
    let tuple_altered_cycle: Vec<(u32, u32)> = altered_cycle
        .into_iter()
        .tuple_windows::<(u32, u32)>()
        .collect();
    //println!("altered cycle = {:?}", tuple_altered_cycle);
    //println!("try to find...{:?}", (i, j));

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
    println!("edge_in_cycle before reorder = {:?}", edge_in_cycle);
    edge_in_cycle = edge_in_cycle
        .into_iter()
        .sorted_by_key(|&x| distance_in_cycle(x, cycle.clone()))
        .rev()
        .collect();
    println!("edge_in_cycle after reorder = {:?}", edge_in_cycle);
    let delta: Vec<i32> = edge_in_cycle
        .iter()
        .map(|&x| {
            if is_forward(x, cycle.clone()) {
                println!(
                    "edge {:?} is forward, delta_ij ={:?}",
                    x,
                    x.weight().capacity - x.weight().flow
                );
                x.weight().capacity - x.weight().flow
            } else {
                println!("edge {:?} is backward delta_ij ={:?}", x, x.weight().flow);
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
    println!(
        "farthest blocking edge = {:?}, amount of flow change = {:?}",
        edge_in_cycle[flowchange.0], flowchange.1
    );

    return (farthest_blocking_edge, edge_flow_change);
}

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

fn update_SPtree_with_leaving(sptree: SPTree, leaving_arc: (u32, u32), clue: i32) -> SPTree {
    let mut updated_sptree = sptree.clone();
    updated_sptree.t = updated_sptree
        .t
        .into_iter()
        .filter(|&x| x != leaving_arc)
        .collect();
    if clue > 0 {
        updated_sptree.u.push(leaving_arc);
    } else {
        updated_sptree.l.push(leaving_arc)
    }
    updated_sptree
}

fn update_SPtree_with_entering(sptree: SPTree, entering_arc: Option<(u32, u32)>) -> SPTree {
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

fn min_cost(
    graph: DiGraph<u32, CustomEdgeIndices<i32>>,
    u: Vec<i32>,
) -> DiGraph<u32, CustomEdgeIndices<i32>> {
    let (mut tlu_solution, mut new_graph) = initialization::<i32, DefaultIx>(graph.clone(), u);

    let mut potentials = compute_node_potential(new_graph.clone(), tlu_solution.clone());
    println!("potentials = {:?}", potentials);
    let mut reduced_cost =
        compute_reduced_cost(potentials, new_graph.clone(), tlu_solution.clone());
    println!("tlu_solution = {:?}", tlu_solution);
    println!("reduced_cost = {:?}", reduced_cost);

    let mut entering_arc = find_entering_arc(tlu_solution.clone(), reduced_cost);
    println!("entering arc = {:?}", entering_arc);
    while entering_arc != None {
        let cycle = find_cycle(tlu_solution.clone(), entering_arc.unwrap());
        println!("cycle found = {:?}", cycle);

        tlu_solution = update_SPtree_with_entering(tlu_solution.clone(), entering_arc);

        let (leaving_arc, vec_flow_change) = compute_flowchange(new_graph.clone(), cycle);

        let clue = vec_flow_change[0].1;

        tlu_solution = update_SPtree_with_leaving(tlu_solution.clone(), leaving_arc, clue);

        new_graph = update_flow(new_graph.clone(), vec_flow_change);
        println!("{:?}", Dot::new(&new_graph));

        potentials = compute_node_potential(new_graph.clone(), tlu_solution.clone());
        println!("potentials = {:?}", potentials);

        reduced_cost = compute_reduced_cost(potentials, new_graph.clone(), tlu_solution.clone());
        println!("tlu_solution = {:?}", tlu_solution);
        println!("reduced_cost = {:?}", reduced_cost);

        entering_arc = find_entering_arc(tlu_solution.clone(), reduced_cost);
        println!("entering arc = {:?}", entering_arc);
    }

    new_graph
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
    println!(" MAX FLOW : {:?}", maxflow);
    g.flow.iter().map(|&x| if x < 0 { 0 } else { x }).collect()
}

fn main() {
    let mut graph = DiGraph::<u32, CustomEdgeIndices<i32>>::new();

    let s = graph.add_node(0);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);
    let n4 = graph.add_node(4);
    let t = graph.add_node(5);
    let N = graph.node_count();
    let u = Vec::from([
        0i32, 9, 0, 8, 0, 0, 0, 0, 5, 0, 2, 0, 0, 0, 0, 0, 0, 7, 0, 0, 3, 0, 6, 0, 0, 0, 0, 0, 0,
        6, 8, 0, 0, 0, 0, 0,
    ]);

    graph.add_edge(
        s,
        n1,
        CustomEdgeIndices {
            cost: (2),
            capacity: (u[s.index() * N + n1.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        s,
        n3,
        CustomEdgeIndices {
            cost: (1),
            capacity: (u[s.index() * N + n3.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n1,
        n2,
        CustomEdgeIndices {
            cost: (2),
            capacity: (u[n1.index() * N + n2.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n1,
        n4,
        CustomEdgeIndices {
            cost: (1),
            capacity: (u[n1.index() * N + n4.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n3,
        n2,
        CustomEdgeIndices {
            cost: (3),
            capacity: (u[n3.index() * N + n2.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n3,
        n4,
        CustomEdgeIndices {
            cost: (1),
            capacity: (u[n3.index() * N + n4.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n2,
        t,
        CustomEdgeIndices {
            cost: (3),
            capacity: (u[n2.index() * N + t.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n4,
        t,
        CustomEdgeIndices {
            cost: (4),
            capacity: (u[n4.index() * N + t.index()]),
            flow: (0),
        },
    );

    /*
    let s = graph.add_node(0);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);
    let n4 = graph.add_node(4);
    let t = graph.add_node(5);
    let N = graph.node_count();
    let u = Vec::from([
        0i32, 0, 7, 6, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 0, 0, 0, 4, 0, 0, 6, 0, 2, 0, 0, 0, 0, 0, 0,
        8, 0, 0, 0, 0, 0, 0,
    ]);

    let c = vec![0; N * N];

    graph.add_edge(
        s,
        n1,
        CustomEdgeIndices {
            cost: (1),
            capacity: (u[s.index() * N + n1.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        s,
        n2,
        CustomEdgeIndices {
            cost: (1),
            capacity: (u[s.index() * N + n2.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        s,
        n3,
        CustomEdgeIndices {
            cost: (1),
            capacity: (u[s.index() * N + n3.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n1,
        n2,
        CustomEdgeIndices {
            cost: (1),
            capacity: (u[n1.index() * N + n2.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n1,
        n4,
        CustomEdgeIndices {
            cost: (1),
            capacity: (u[n1.index() * N + n4.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n3,
        n2,
        CustomEdgeIndices {
            cost: (1),
            capacity: (u[n3.index() * N + n2.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n3,
        n4,
        CustomEdgeIndices {
            cost: (1),
            capacity: (u[n3.index() * N + n4.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n2,
        t,
        CustomEdgeIndices {
            cost: (1),
            capacity: (u[n2.index() * N + t.index()]),
            flow: (0),
        },
    );
    graph.add_edge(
        n4,
        t,
        CustomEdgeIndices {
            cost: (1),
            capacity: (u[n4.index() * N + t.index()]),
            flow: (0),
        },
    );*/

    let min_cost_flow = min_cost(graph, u);
    println!("{:?}", Dot::new(&min_cost_flow));
}
