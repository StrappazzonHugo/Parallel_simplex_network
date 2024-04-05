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

  ListDigraph::Node n0 = g.addNode();
  ListDigraph::Node n1 = g.addNode();

  int grid_size = 150;
  vector<vector<ListDigraph::Node>> nodes(grid_size, vector<ListDigraph::Node> (grid_size));

  ListDigraph::ArcMap<int> costmap(g);
  ListDigraph::ArcMap<int> capamap(g);

  for (int i = 0; i < grid_size; i++) {
    for (int j = 0; j < grid_size; j++) {
      ListDigraph::Node n = g.addNode();
      nodes[i][j] = n;
    }
  }

  for (int i = 0; i < grid_size; i++) {
    for (int j = 0; j < grid_size; j++) {
      if (i != grid_size - 1) {
        ListDigraph::Arc h = g.addArc(nodes[i][j], nodes[i+1][j]);
        costmap.set(h, randomnumber() * grid_size);
        //costmap.set(h, 40);
        capamap.set(h, 10);
      }
      ListDigraph::Arc v = g.addArc(nodes[i][j], nodes[i][(j + 1) % grid_size]);
      costmap.set(v, randomnumber() * grid_size);
      //costmap.set(v, 40);
      capamap.set(v, 10);
    }
  }

  for (int j = 0; j< grid_size; j++) {
    ListDigraph::Arc s = g.addArc(n0, nodes[0][j]);
    capamap.set(s, 100000000);
    costmap.set(s, 0);
    ListDigraph::Arc t = g.addArc(nodes[grid_size-1][j], n1);
    capamap.set(t, 100000000);
    costmap.set(t, 0);
  }

  NetworkSimplex<ListDigraph> ns(g);
  ns.costMap(costmap);
  ns.upperMap(capamap);
  int demand = 5;
  ns.stSupply(n0, n1, demand * grid_size);

  ListDigraph::ArcMap<int> res(g);
  auto start = high_resolution_clock::now();
  cout << "starting..." << endl;
  ns.run(lemon::NetworkSimplex<ListDigraph>::BLOCK_SEARCH);
  auto stop = high_resolution_clock::now();
  ns.flowMap(res);
  cout << "total cost : " << ns.totalCost() << endl;
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "time : " << duration.count() << endl;
  return 0;
};
