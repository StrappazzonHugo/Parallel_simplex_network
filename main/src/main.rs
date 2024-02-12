use itertools::Itertools;
use petgraph::dot::Dot;
use petgraph::graph::node_index;
use petgraph::graph::DefaultIx;
use petgraph::prelude::*;
use std::any::type_name;

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
) -> SPTree {
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

    let potentials = compute_node_potential(graph.clone(), tlu_solution.clone());
    println!("potentials = {:?}", potentials);
    let reduced_cost = computing_reduced_cost(potentials, graph.clone(), tlu_solution.clone());
    println!("reduced_cost = {:?}", reduced_cost);
    tlu_solution
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
            .into_iter()
            .find_position(|&x| x != -INF && !node_computed.contains(&x))
            .unwrap()
            .0 as i32;
        node_computed.push(current);
    }
    pi
}

fn computing_reduced_cost<N>(
    pi: Vec<i32>,
    graph: DiGraph<N, CustomEdgeIndices<i32>>,
    sptree: SPTree,
) -> Vec<i32> {
    let mut reduced_cost = Vec::new();
    sptree.l.iter().for_each(|&(u, v)| {
        reduced_cost.push(
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
        reduced_cost.push(
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
    reduced_cost
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
    /*
    //Graph example definition
    let mut graph = DiGraph::<i32, _>::new();
    let root = graph.add_node(0);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let t = graph.add_node(3);
    graph.add_edge(root, n1, 1);
    graph.add_edge(root, n2, 1);
    graph.add_edge(n1, n2, 1);
    graph.add_edge(n2, t, 1);
    graph.add_edge(n1, t, 1);
    graph.add_edge(root, t, 1);

    //let adj = graph.adjacency_matrix();

    //Flow capacities and costs
    let c = Vec::from([0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]);
    let u = Vec::from([0i32, 2, 2, 0, 0, 0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0]);
    */

    let mut graph = DiGraph::<u32, CustomEdgeIndices<i32>>::new();

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
    );

    let _initial_feasible_solution = initialization::<i32, DefaultIx>(graph.clone(), u);
}
