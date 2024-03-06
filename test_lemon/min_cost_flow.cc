#include <cmath>
#include <cstdlib>
#include <lemon/list_graph.h>
#include <lemon/network_simplex.h>
#include <ostream>
#include <random>

using namespace lemon;
using namespace std;

double randomnumber() {
  // Making rng static ensures that it stays the same
  // Between different invocations of the function
  static std::default_random_engine rng;

  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng);
}

int main() {
  ListDigraph g;

  ListDigraph::Node n0 = g.addNode();
  ListDigraph::Node n1 = g.addNode();

  vector<ListDigraph::Node> left;
  vector<ListDigraph::Node> right;

  ListDigraph::ArcMap<int> costmap(g);
  ListDigraph::ArcMap<int> capamap(g);
  for (int i = 0; i < 1000; i++) {
    ListDigraph::Node l = g.addNode();
    left.push_back(l);
    ListDigraph::Arc a1 = g.addArc(n0, l);
    costmap.set(a1, 0);
    capamap.set(a1, 100000000);

    ListDigraph::Node r = g.addNode();
    right.push_back(r);
    ListDigraph::Arc a2 = g.addArc(r, n1);
    costmap.set(a2, 0);
    capamap.set(a2, 100000000);
  }

  for (int i = 0; i < left.size(); i++) {
    int x1 = randomnumber() * 10000;
    int y1 = randomnumber() * 10000;
    for (int j = 0; j < right.size(); j++) {
      int x2 = randomnumber() * 10000;
      int y2 = randomnumber() * 10000;
      int cost = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
      ListDigraph::Arc arc = g.addArc(left[i], right[j]);
      costmap.set(arc, cost);
      capamap.set(arc, 10);
    }
  }

  NetworkSimplex<ListDigraph> ns(g);
  ns.costMap(costmap);
  ns.upperMap(capamap);
  ns.stSupply(n0, n1, 20000);

  ListDigraph::ArcMap<int> res(g);
  cout << "starting..." << endl;
  ns.run(lemon::NetworkSimplex<ListDigraph>::FIRST_ELIGIBLE);
  ns.flowMap(res);
  cout << "total cost : " << ns.totalCost() << endl;
    return 0;
};
