#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
using namespace std;

/**
 * @brief 无向图（使用邻接表表示）
 */
class UndirectedGraph {
private:
	int V;
	vector<vector<int>> adj;
public:
	UndirectedGraph(int V) {
		this->V = V;
		adj.resize(V);
	}
	void addEdge(int u, int v) {
		adj[u].push_back(v);
		adj[v].push_back(u);
	}
	void DFS(int v, vector<bool>& visited) {
		visited[v] = true;
		cout << v << " ";
		for (int i : adj[v]) {
			if (!visited[i]) {
				DFS(i, visited);
			}
		}
	}
	void BFS(int start) {
		vector<bool> visited(V, false);
		queue<int> q;
		visited[start] = true;
		q.push(start);
		while (!q.empty()) {
			int v = q.front();
			q.pop();
			cout << v << " ";
			for (int i : adj[v]) {
				if (!visited[i]) {
					visited[i] = true;
					q.push(i);
				}
			}
		}
	}
};

int test_visit_undirectedGraph_using_DFS() {
	int V = 5;
	UndirectedGraph g(V);
	g.addEdge(0, 1);
	g.addEdge(0, 2);
	g.addEdge(1, 3);
	g.addEdge(1, 4);
	vector<bool> visited(V, false);
	g.DFS(0, visited);
	std::cout << std::endl;
	return 0;
}

int test_visit_undirectedGraph_using_BFS() {
	int V = 6;
	UndirectedGraph g(V);
	g.addEdge(0, 1);
	g.addEdge(0, 2);
	g.addEdge(1, 3);
	g.addEdge(1, 4);
	g.addEdge(2, 4);
	g.addEdge(3, 5);
	g.addEdge(4, 5);
	g.BFS(0);
	std::cout << std::endl;
	return 0;
}
int test_visit_graph(std::string& str_err_reason) {
	test_visit_undirectedGraph_using_DFS();
	test_visit_undirectedGraph_using_BFS();
	return 0;
}
