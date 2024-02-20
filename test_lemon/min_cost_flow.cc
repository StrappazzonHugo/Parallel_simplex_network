#include <lemon/list_graph.h>
#include <lemon/network_simplex.h>
#include <ostream>

using namespace lemon;
using namespace std;

int main() {
  ListDigraph g;

  ListDigraph::Node n0 = g.addNode();
  ListDigraph::Node n1 = g.addNode();
  ListDigraph::Node n2 = g.addNode();
  ListDigraph::Node n3 = g.addNode();
  ListDigraph::Node n4 = g.addNode();
  ListDigraph::Node n5 = g.addNode();
  ListDigraph::Node n6 = g.addNode();
  ListDigraph::Node n7 = g.addNode();
  ListDigraph::Node n8 = g.addNode();
  ListDigraph::Node n9 = g.addNode();
  ListDigraph::Node n10 = g.addNode();
  ListDigraph::Node n11 = g.addNode();
  ListDigraph::Node n12 = g.addNode();
  ListDigraph::Node n13 = g.addNode();
  ListDigraph::Node n14 = g.addNode();
  ListDigraph::Node n15 = g.addNode();
  ListDigraph::Node n16 = g.addNode();

  ListDigraph::Arc a01 = g.addArc(n0, n1);
  ListDigraph::Arc a02 = g.addArc(n0, n2);
  ListDigraph::Arc a03 = g.addArc(n0, n3);
  ListDigraph::Arc a04 = g.addArc(n0, n4);
  ListDigraph::Arc a05 = g.addArc(n0, n5);

  ListDigraph::Arc a16 = g.addArc(n1, n6);
  ListDigraph::Arc a26 = g.addArc(n2, n6);
  ListDigraph::Arc a23 = g.addArc(n2, n3);
  ListDigraph::Arc a37 = g.addArc(n3, n7);
  ListDigraph::Arc a34 = g.addArc(n3, n4);
  ListDigraph::Arc a49 = g.addArc(n4, n9);
  ListDigraph::Arc a58 = g.addArc(n5, n8);

  ListDigraph::Arc a610 = g.addArc(n6, n10);
  ListDigraph::Arc a67 = g.addArc(n6, n7);
  ListDigraph::Arc a78 = g.addArc(n7, n8);
  ListDigraph::Arc a810 = g.addArc(n8, n10);
  ListDigraph::Arc a811 = g.addArc(n8, n11);
  ListDigraph::Arc a911 = g.addArc(n9, n11);

  ListDigraph::Arc a1012 = g.addArc(n10, n12);
  ListDigraph::Arc a1014 = g.addArc(n10, n14);
  ListDigraph::Arc a1115 = g.addArc(n11, n15);
  ListDigraph::Arc a1113 = g.addArc(n11, n13);
  ListDigraph::Arc a1315 = g.addArc(n13, n15);

  ListDigraph::Arc a1415 = g.addArc(n14, n15);
  ListDigraph::Arc a1416 = g.addArc(n14, n16);
  ListDigraph::Arc a1516 = g.addArc(n15, n16);

  ListDigraph::ArcMap<int> costmap(g);
  costmap.set(a01, 1);
  costmap.set(a02, 1);
  costmap.set(a03, 1);
  costmap.set(a04, 1);
  costmap.set(a05, 1);

  costmap.set(a16, 1);
  costmap.set(a26, 1);
  costmap.set(a23, 1);
  costmap.set(a37, 1);
  costmap.set(a34, 1);
  costmap.set(a49, 1);
  costmap.set(a58, 1);

  costmap.set(a610, 1);
  costmap.set(a67, 1);
  costmap.set(a78, 1);
  costmap.set(a810, 1);
  costmap.set(a811, 1);
  costmap.set(a911, 1);

  costmap.set(a1012, 1);
  costmap.set(a1014, 1);
  costmap.set(a1115, 1);
  costmap.set(a1113, 1);
  costmap.set(a1315, 1);

  costmap.set(a1415, 1);
  costmap.set(a1416, 1);
  costmap.set(a1516, 1);

  ListDigraph::ArcMap<int> capamap(g);
  capamap.set(a01, 2);
  capamap.set(a02, 6);
  capamap.set(a03, 4);
  capamap.set(a04, 2);
  capamap.set(a05, 3);

  capamap.set(a16, 3);
  capamap.set(a26, 8);
  capamap.set(a23, 2);
  capamap.set(a37, 10);
  capamap.set(a34, 2);
  capamap.set(a49, 3);
  capamap.set(a58, 7);

  capamap.set(a610, 11);
  capamap.set(a67, 7);
  capamap.set(a78, 4);
  capamap.set(a810, 7);
  capamap.set(a811, 8);
  capamap.set(a911, 5);

  capamap.set(a1012, 3);
  capamap.set(a1014, 6);
  capamap.set(a1115, 10);
  capamap.set(a1113, 3);
  capamap.set(a1315, 8);

  capamap.set(a1415, 3);
  capamap.set(a1416, 10);
  capamap.set(a1516, 12);

  NetworkSimplex<ListDigraph> ns(g);
  ns.costMap(costmap);
  ns.upperMap(capamap);
  ns.stSupply(n0, n16, 9);

  ListDigraph::ArcMap<int> res(g);
  ns.run(lemon::NetworkSimplex<ListDigraph>::FIRST_ELIGIBLE);
  ns.flowMap(res);
  cout << "total cost : " << ns.totalCost() << endl;
  for (ListDigraph::ArcIt a(g); a != INVALID; ++a)
    cout << g.id(g.source(a)) << "->"  << g.id(g.target(a)) << " : flow = " << ns.flow(a) << endl;
  // testing network simplex..
  return 0;
};
