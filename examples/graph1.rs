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
