use petgraph::{prelude::*, visit::GetAdjacencyMatrix};

const INF: i32 = 99999999;

struct SPTree {
    t: Vec<(u32, u32)>,
    l: Vec<(u32, u32)>,
    u: Vec<(u32, u32)>,
}

fn main() {
    //Graph definition
    let mut graph = Graph::new();
    let root = graph.add_node("root");
    let n1 = graph.add_node("1");
    let n2 = graph.add_node("2");
    let t = graph.add_node("sink");
    graph.add_edge(root, n1, 1);
    graph.add_edge(root, n2, 1);
    graph.add_edge(n2, n1, 1);
    graph.add_edge(n2, t, 1);
    graph.add_edge(n1, t, 1);
    graph.add_edge(root, t, 1);

    let adj = graph.adjacency_matrix();

    //Flow capacities and costs
    //let c = arr2(&[[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]);
    let N = 4;
    let g = PRstruct{
        n : N, 
        capacity: Vec::from([
        0i32, 2, 3, 0,
        0, 0, 1, 3,
        0, 0, 0, 2,
        0, 0, 0, 0,
    ]),
        flow: vec![0; 4*4],
        label: Vec::new(),
        excess: Vec::new(),
        excess_vertices: Vec::new(),
        seen: Vec::new(),
    };
    
    println!("{:?}", g.capacity);
    //Maxflow
    let feasible_sol:Vec<i32> = max_flow(0, N-1, g);
    


    println!("{:?}", feasible_sol);


    //initialization(graph, x, u, c);
}



/*
fn _initialization<N, E>(
    graph: DiGraph<N, E>,
    x: Array2<u32>,
    u: Array2<u32>,
    c: Array2<u32>,
) -> SPTree {
    







    let mut T: Vec<(u32, u32)> = Vec::new();
    let mut L: Vec<(u32, u32)> = Vec::new();
    let mut U: Vec<(u32, u32)> = Vec::new();

    SPTree { t: T, l: L, u: U }
}*/


///////////////////////////
//Push-Relabel algorithm
///////////////////////////

//structure for Push-Relabel algorithm
struct PRstruct {
    capacity: Vec<i32>,//2d vec flatten
    flow: Vec<i32>,//2d vec flatten
    label: Vec<i32>,
    excess: Vec<i32>,
    excess_vertices: Vec<usize>,
    seen: Vec<usize>,
    n: usize,
}

fn push(u: usize, v: usize, mut g: PRstruct) -> PRstruct {
    let d = std::cmp::min(g.excess[u], g.capacity[g.n*u+v] - g.flow[g.n*u+v]);
    g.flow[g.n*u+v] += d;
    g.flow[g.n*v+u] -= d;
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
        if (g.capacity[g.n*u+i] - g.flow[g.n*u+i]) > 0 {
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
            if (g.capacity[g.n*u+v] - g.flow[g.n*u+v]) > 0 && g.label[u] > g.label[v] {
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
    g.flow = vec![0; g.n*g.n];
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
        maxflow += g.flow[i*g.n + t];
    }
    println!(" MAX FLOW : {:?}", maxflow);
    g.flow.iter().map(|&x| if x < 0 {0} else {x}).collect()
}
