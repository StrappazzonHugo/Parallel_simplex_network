#include <chrono>
#include <cmath>
#include <cstdlib>
#include <lemon/list_graph.h>
#include <lemon/network_simplex.h>
#include <ostream>
#include <random>

using namespace lemon;
using namespace std;
using namespace std::chrono;

double randomnumber() {
  // Making rng static ensures that it stays the same
  // Between different invocations of the function
  static std::default_random_engine rng;

  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(rng);
}

int main() {
  ListDigraph g;
  int node_nb = 5000;

  ListDigraph::Node n0 = g.addNode();
  ListDigraph::Node n1 = g.addNode();

  vector<ListDigraph::Node> left;
  vector<ListDigraph::Node> right;

  ListDigraph::ArcMap<int> costmap(g);
  ListDigraph::ArcMap<int> capamap(g);

  for (int i = 0; i < node_nb; i++) {
    ListDigraph::Node l = g.addNode();
    ListDigraph::Arc a1;
    if (i == 0) {
      a1 = g.addArc(n0, l);

    } else {
      a1 = g.addArc(left.back(), l);
    }
    left.push_back(l);
    costmap.set(a1, 1);
    capamap.set(a1, 1000);
  }

  for (int i = 0; i < node_nb; i++) {
    ListDigraph::Node r = g.addNode();
    ListDigraph::Arc a2;
    if (i == 0) {
      a2 = g.addArc(n0, r);

    } else {
      a2 = g.addArc(right.back(), r);
    }
    right.push_back(r);
    costmap.set(a2, 1);
    capamap.set(a2, 1000);
    
    int rng = randomnumber() * 100;

    int v = randomnumber() * left.size();
    if (rng < 1) {
      ListDigraph::Arc a = g.addArc(r, left[v]);
      costmap.set(a, 1);
      capamap.set(a, 1000);
      cout << "rng = " << rng << " added arc from " << node_nb + i << " to " << v << endl;
    }
    if (rng >= 99) {
      ListDigraph::Arc a = g.addArc(left[v], r);
      costmap.set(a, 1);
      capamap.set(a, 1000);
      cout << "rng = " << rng << " added arc from " << v << " to " << node_nb + i << endl;
    }
  }

  ListDigraph::Arc f1 = g.addArc(left.back(), n1);
  ListDigraph::Arc f2 = g.addArc(right.back(), n1);
  costmap.set(f1, 5);
  capamap.set(f1, 1000);
  costmap.set(f2, 7);
  capamap.set(f2, 1000);

  NetworkSimplex<ListDigraph> ns(g);
  ns.costMap(costmap);
  ns.upperMap(capamap);
  ns.stSupply(n0, n1, 900);

  ListDigraph::ArcMap<int> res(g);
  auto start = high_resolution_clock::now();
  cout << "starting..." << endl;
  ns.run(lemon::NetworkSimplex<ListDigraph>::FIRST_ELIGIBLE);
  auto stop = high_resolution_clock::now();
  ns.flowMap(res);
  cout << "total cost : " << ns.totalCost() << endl;
  auto duration = duration_cast<seconds>(stop - start);
  cout << "time : " << duration.count() << endl;
  return 0;
};
