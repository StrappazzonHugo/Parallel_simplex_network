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

  ListDigraph::Arc a01 = g.addArc(n0, n1);
  ListDigraph::Arc a02 = g.addArc(n0, n2);
  ListDigraph::Arc a03 = g.addArc(n0, n3);

  ListDigraph::Arc a14 = g.addArc(n1, n4);

  ListDigraph::Arc a24 = g.addArc(n2, n4);
  ListDigraph::Arc a25 = g.addArc(n2, n5);

  ListDigraph::Arc a32 = g.addArc(n3, n2);
  ListDigraph::Arc a36 = g.addArc(n3, n6);

  ListDigraph::Arc a45 = g.addArc(n4, n5);

  ListDigraph::Arc a57 = g.addArc(n5, n7);

  ListDigraph::Arc a67 = g.addArc(n6, n7);
  ListDigraph::Arc a62 = g.addArc(n6, n2);

  ListDigraph::ArcMap<int> costmap(g);
  costmap.set(a01, 2);
  costmap.set(a02, 1);
  costmap.set(a03, 1);
  costmap.set(a14, 4);
  costmap.set(a24, 2);
  costmap.set(a25, 1);
  costmap.set(a32, 1);
  costmap.set(a36, 2);
  costmap.set(a45, 2);
  costmap.set(a57, 2);
  costmap.set(a67, 4);
  costmap.set(a62, 1);

  ListDigraph::ArcMap<int> capamap(g);
  capamap.set(a01, 2);
  capamap.set(a02, 3);
  capamap.set(a03, 4);
  capamap.set(a14, 1);
  capamap.set(a24, 5);
  capamap.set(a25, 3);
  capamap.set(a32, 1);
  capamap.set(a36, 6);
  capamap.set(a45, 3);
  capamap.set(a57, 9);
  capamap.set(a67, 7);
  capamap.set(a62, 2);

  NetworkSimplex<ListDigraph> ns(g);
  ns.costMap(costmap);
  ns.upperMap(capamap);
  ns.stSupply(n0, n7, 9);

  ListDigraph::ArcMap<int> res(g);
  ns.run();
  ns.flowMap(res);
  for (ListDigraph::ArcIt a(g); a != INVALID; ++a)
    cout << g.id(g.source(a)) << "->"  << g.id(g.target(a)) << " : flow = " << ns.flow(a) << endl;
  // testing network simplex..
  return 0;
};
