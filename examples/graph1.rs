    let mut graph = DiGraph::<u32, CustomEdgeIndices<f32>>::new();
    let n0 = graph.add_node(0);
    let n1 = graph.add_node(1);
    let n2 = graph.add_node(2);
    let n3 = graph.add_node(3);
    let n4 = graph.add_node(4);
    let n5 = graph.add_node(99);
    graph.add_edge(n0, n1, CustomEdgeIndices {cost: (1.), capacity: (6.5), flow: (0.), },);
    graph.add_edge(n0, n3, CustomEdgeIndices {cost: (1.), capacity: (5.5), flow: (0.), },);
    graph.add_edge(n1, n2, CustomEdgeIndices {cost: (1.), capacity: (3.5), flow: (0.), },);
    graph.add_edge(n1, n4, CustomEdgeIndices {cost: (1.), capacity: (4.5), flow: (0.), },);
    graph.add_edge(n3, n2, CustomEdgeIndices {cost: (1.), capacity: (1.5), flow: (0.), },);
    graph.add_edge(n3, n4, CustomEdgeIndices {cost: (1.), capacity: (5.5), flow: (0.), },);
    graph.add_edge(n2, n5, CustomEdgeIndices {cost: (1.), capacity: (8.5), flow: (0.), },);
    graph.add_edge(n4, n5, CustomEdgeIndices {cost: (1.), capacity: (7.5), flow: (0.), },);
